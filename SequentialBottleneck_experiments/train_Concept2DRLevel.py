import os
import torch
import copy
#import pickle
import numpy as np
import pandas as pd
#from PIL import Image
#from CUB.config import BASE_DIR, N_ATTRIBUTES
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

import random
import argparse
from torch import functional
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
#Import the WeightedRandomSampler:
from torch.utils.data.sampler import WeightedRandomSampler
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset
from models import ModelXtoC,ModelXtoChat_ChatToY,ModelOracleCtoY
from analysis import accuracy, AverageMeter



#Device:
# Should us ID = 0 (Vajira will use ID = 1)
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',DEVICE)

#Define dataset class:
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    #CLASSES = ['0','1','2','3','4']
    
    def __init__(
            self, 
            filepaths, 
            concept_df, #Order of the concepts in the df are: MA, HE, SoftEx, HardEx, NV, IRMA
            #augmentation=None, 
            #preprocessing=None,
    ):
        self.filepaths = filepaths
        self.concept_df = concept_df
        #self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        #image_name = os.path.normpath(image_path).split(os.sep)[-1]
        df_row = self.concept_df.loc[self.concept_df['Image_path']==image_path]
        #print('Current row:',df_row)
        #Get the raw predicted concept
        concept_data = df_row.iloc[0,1]
        #Since these are (of unknown causes) interpreted as a string-list
        #We need to convert them to a proper list of float-values:
        concept_data = concept_data.strip('"')
        concept_data = concept_data.strip('[]')
        #print('Concept data:',concept_data)
        concept_data = list(concept_data.split(','))
        concept_data = list(map(float,concept_data))
        #print('Concept data:',concept_data)
        label = df_row.iloc[0,-1]
        #print('Raw concepts:',concept_data)
        return concept_data, label
        
    def __len__(self):
        #return len(self.filepaths)
        return self.concept_df.shape[0]


def run_epoch_simple(model, optimizer, dataloaders, criterion, n_epochs):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0
    
    for _epoch in range(n_epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        for phase, loader in dataloaders.items():
            if phase == 'TRAIN':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0
            
            with torch.set_grad_enabled(phase == "TRAIN"):
                for _, (inputs, y_true) in enumerate(loader):
                #for _, data in enumerate(loader):
                #inputs, labels = data
                    if isinstance(inputs, list):
                        #inputs = [i.long() for i in inputs]
                        inputs = torch.stack(inputs).t().float()
                    #print('The inputs:',inputs)
                    inputs = torch.tensor(np.asarray(inputs))
                    #inputs = torch.stack(list(inputs), dim=0)
                    #print('Inputs as tensor:',inputs)
                    inputs = torch.flatten(inputs, start_dim=1).float()
                    inputs_var = torch.autograd.Variable(inputs).cuda()
                    inputs_var = inputs_var.to(DEVICE)
                    labels_var = torch.autograd.Variable(y_true).cuda()
                    labels_var = labels_var.to(DEVICE)
        
                    outputs = model(inputs_var)
                    loss = criterion(outputs, labels_var)
                    acc = accuracy(outputs, y_true, topk=(1,))
                    #print('Acc[0]:',acc[0])
                    #print('Entire accuracy:',acc)
                    #Since returned as a list of one element, need to just specify 1st element
                    running_acc += acc[0].data.cpu().numpy()
                    if phase == 'TRAIN':
                        train_loss_meter.update(loss.item(), inputs.size(0))
                        train_acc_meter.update(acc[0], inputs.size(0))
                        optimizer.zero_grad() #zero the parameter gradients
                        loss.backward()
                        optimizer.step() #optimizer step to update parameters
                    else:
                        val_loss_meter.update(loss.item(), inputs.size(0))
                        val_acc_meter.update(acc[0], inputs.size(0))
                    running_loss += loss.item()
            mean_loss = running_loss / len(loader)
            mean_acc = running_acc / len(loader)
            if best_val_acc < val_acc_meter.avg:
                print('Update the validation accuracy!')
                best_val_acc = val_acc_meter.avg
                print('New best accuracy:',best_val_acc)
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'VALID':
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,val_loss_meter.avg, val_acc_meter.avg))
            else: 
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,train_loss_meter.avg, train_acc_meter.avg))
    print("Best val Acc: %.4f" % best_val_acc)
    model.load_state_dict(best_model_wts)
    return model

#Main code for training the second part of the sequential model:
output_path = "../../output"
model_save_path = os.path.join(output_path, "BottleneckDensenet121_SequentialModel_part2.pt")
#Get the filenames for the training data:
n_classes = 5
n_concepts = 6
train_folder = '../../Data/CroppedDataKaggle/CroppedTrainFGADR'
small_list = [os.path.join(train_folder, str(class_id)) for class_id in range(n_classes)]
print('Small list training:', small_list)
train_filepath = []
for _list in small_list:
    all_files = os.listdir(_list)
    print('Number of files:',len(all_files))
    all_paths = []
    #For each image in the class folder
    for _img in all_files:
        single_path = os.path.join(_list,_img)
        all_paths.append(single_path)
        #Add the full image path to image_list:
    train_filepath += all_paths
print('Length of training files:',len(train_filepath))
print('First filepath:',train_filepath[0])

#Repeat for validation folder:
valid_folder = '../../Data/CroppedDataKaggle/CroppedValidFGADR'
small_listVal = [os.path.join(valid_folder, str(class_id)) for class_id in range(n_classes)]
print('Small list validation:', small_listVal)
valid_filepath = []
for _list in small_listVal:
    all_files = os.listdir(_list)
    print('Number of files:',len(all_files))
    all_paths = []
    #For each image in the class folder
    for _img in all_files:
        single_path = os.path.join(_list,_img)
        all_paths.append(single_path)
    #Add the full image path to image_list:
    valid_filepath += all_paths
print('Length of validation files:',len(valid_filepath))

########### Use the weighted random sampler #################
#Expects a tensor weight for each sample
#Inspired by this code: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
#Get the number of observations for each class:
class0_observations = len(os.listdir('../../Data/CroppedDataKaggle/CroppedTrainFGADR/0'))
class1_observations = len(os.listdir('../../Data/CroppedDataKaggle/CroppedTrainFGADR/1'))
class2_observations = len(os.listdir('../../Data/CroppedDataKaggle/CroppedTrainFGADR/2'))
class3_observations = len(os.listdir('../../Data/CroppedDataKaggle/CroppedTrainFGADR/3'))
class4_observations = len(os.listdir('../../Data/CroppedDataKaggle/CroppedTrainFGADR/4'))
class_sample_count = np.array([class0_observations,class1_observations,class2_observations,class3_observations,class4_observations])
class_weights = 1. / class_sample_count
targets = [0,1,2,3,4]
sample_weights = []
for _t in targets:
    #Get X number of class weights, where X is number of obs for that given class
    sample_weigths_targetList = [class_weights[_t] for i in range(list(class_sample_count)[_t])]
    print('Weights for class',_t)
    print(sample_weigths_targetList[0])
    sample_weights +=  sample_weigths_targetList
#The length of sample_weights must equal the total number of obs in training dataset:
#print('Length of the sample weight list:',len(sample_weights))
#print('Entire sample weight list:')
#print(sample_weights[0])
sample_weights = np.array(sample_weights)
class_weights = torch.from_numpy(sample_weights)
my_sampler = WeightedRandomSampler(class_weights,len(class_weights))    
print('Length of class weights:',len(class_weights))

#Load in the raw concept df predictions:
conceptPredictions_train = pd.read_csv('SequentialModelOutput/rawDensenet121_conceptPredictions_train.csv',index_col = 'Unnamed: 0')
conceptPredictions_valid = pd.read_csv('SequentialModelOutput/rawDensenet121_conceptPredictions_valid.csv',index_col = 'Unnamed: 0')

#Create the dataset:
train_dataset = Dataset(filepaths = train_filepath,concept_df = conceptPredictions_train)
valid_dataset = Dataset(filepaths = valid_filepath,concept_df = conceptPredictions_valid)
print('Shape of the concept DF:',conceptPredictions_train.shape)
#Create the dataloader:
#NB! Want to use weighted random sampler to take DR class imbalance into account
#This is not implemented in the calculation of the loss for this model
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=8, sampler = my_sampler)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle= False ,num_workers=4)
#next(iter(train_loader))

#Get the model:
model = ModelXtoChat_ChatToY(n_class_attr=2, n_attributes=n_concepts,
                                 num_classes=n_classes, expand_dim=0)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters()) 
criterion = torch.nn.CrossEntropyLoss()

best_model = run_epoch_simple(
        model=model,
        optimizer = optimizer, 
        dataloaders={
            "TRAIN": train_loader,
            "VALID": valid_loader
        }, 
        criterion=criterion, 
        n_epochs=100
        )
#Save best model (based on validation set)
torch.save(best_model.state_dict(), model_save_path)
