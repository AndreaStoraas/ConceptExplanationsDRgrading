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

#####################
# This code trains a logistic regression model to classify DR level from predicted concepts (provided by the bottleneck model)
# NB! If trained on 4 concepts only, the num_concepts should be changed from 6 to 4
#####################

#Device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define dataset class:
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        filepaths (list): list of paths to images folder
        concept_df (DataFrame): DF with image name, 6 concept annotations and DR level
    """
    
    def __init__(
            self, 
            filepaths, 
            concept_df, #Order of the concepts in the df are: MA, HE, SoftEx, HardEx, NV, IRMA
    ):
        self.filepaths = filepaths
        self.concept_df = concept_df

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        df_row = self.concept_df.loc[self.concept_df['Image_path']==image_path]
        #Get the raw predicted concept
        concept_data = df_row.iloc[0,1]
        #Since these are (of unknown causes) interpreted as a string-list
        #We need to convert them to a proper list of float-values:
        concept_data = concept_data.strip('"')
        concept_data = concept_data.strip('[]')
        concept_data = list(concept_data.split(','))
        concept_data = list(map(float,concept_data))
        label = df_row.iloc[0,-1]
        return concept_data, label
        
    def __len__(self):
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
                    if isinstance(inputs, list):
                        inputs = torch.stack(inputs).t().float()
                    inputs = torch.tensor(np.asarray(inputs))
                    inputs = torch.flatten(inputs, start_dim=1).float()
                    inputs_var = torch.autograd.Variable(inputs).cuda()
                    inputs_var = inputs_var.to(DEVICE)
                    labels_var = torch.autograd.Variable(y_true).cuda()
                    labels_var = labels_var.to(DEVICE)
        
                    outputs = model(inputs_var)
                    loss = criterion(outputs, labels_var)
                    acc = accuracy(outputs, y_true, topk=(1,))
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
            if best_val_acc < val_acc_meter.avg:
                best_val_acc = val_acc_meter.avg
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'VALID':
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,val_loss_meter.avg, val_acc_meter.avg))
            else: 
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,train_loss_meter.avg, train_acc_meter.avg))
    print("Best val Acc: %.4f" % best_val_acc)
    model.load_state_dict(best_model_wts)
    return model

#Main code for training the second part of the sequential model:
output_path = "../output"
#Define the name we want to save the new DR level classifier as:
model_save_path = os.path.join(output_path, "BottleneckDensenet121_SequentialModel_part2.pt")
#Get the filenames for the training data:
n_classes = 5
n_concepts = 6
train_folder = '../Data/CroppedDataKaggle/CroppedTrainFGADR'
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
valid_folder = '../Data/CroppedDataKaggle/CroppedValidFGADR'
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
class0_observations = len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainFGADR/0'))
class1_observations = len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainFGADR/1'))
class2_observations = len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainFGADR/2'))
class3_observations = len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainFGADR/3'))
class4_observations = len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainFGADR/4'))
class_sample_count = np.array([class0_observations,class1_observations,class2_observations,class3_observations,class4_observations])
class_weights = 1. / class_sample_count
#5 DR level classes:
targets = [0,1,2,3,4]
sample_weights = []
for _t in targets:
    #Get X number of class weights, where X is number of obs for that given class
    sample_weigths_targetList = [class_weights[_t] for i in range(list(class_sample_count)[_t])]
    print('Weights for class',_t)
    print(sample_weigths_targetList[0])
    sample_weights +=  sample_weigths_targetList
#The length of sample_weights must equal the total number of obs in training dataset:
sample_weights = np.array(sample_weights)
class_weights = torch.from_numpy(sample_weights)
my_sampler = WeightedRandomSampler(class_weights,len(class_weights))    

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
