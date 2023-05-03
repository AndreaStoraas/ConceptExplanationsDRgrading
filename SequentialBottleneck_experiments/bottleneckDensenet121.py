import os
import types
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models
from template_model import FC

import copy
import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler, DataLoader
from torchvision import datasets, models, transforms
#Import the WeightedRandomSampler:
from torch.utils.data.sampler import WeightedRandomSampler
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset
from models import ModelXtoC,ModelXtoChat_ChatToY,ModelOracleCtoY
from analysis import binary_accuracy, AverageMeter

#Code for training the model on all 6 concepts using FGADR!

#Device:
# Should us ID = 0 (Vajira will use ID = 1)
torch.cuda.set_device(1)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Device:',DEVICE)


################# Dataset and helper functions -------------------
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
            augmentation=None, 
            #preprocessing=None,
    ):
        self.filepaths = filepaths
        self.concept_df = concept_df
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #print('Checking the class:')
        #print(os.path.normpath(image_path).split(os.sep)[-2])
        #Check the class:
        if os.path.normpath(image_path).split(os.sep)[-2]=='0':
            label = 0
        elif os.path.normpath(image_path).split(os.sep)[-2]=='1':
            label = 1
        elif os.path.normpath(image_path).split(os.sep)[-2]=='2':
            label = 2
        elif os.path.normpath(image_path).split(os.sep)[-2]=='3':
            label = 3
        elif os.path.normpath(image_path).split(os.sep)[-2]=='4':
            label = 4
        else:
            print('Something is wrong with the classes...')
        # apply augmentations
        if self.augmentation:
            image = self.augmentation(image=image)['image']    
        # get the image name:
        image_name = os.path.normpath(image_path).split(os.sep)[-1]
        # get the corresponding presence/absence of concepts
        df_row = self.concept_df.loc[self.concept_df['image_name']==image_name]
        concept_annotations = df_row.iloc[0,1:7].values.tolist()
        #print('Correct row:',df_row)
        #print('Concept annotations:',concept_annotations)
        return image, label, concept_annotations
        
    def __len__(self):
        return len(self.filepaths)


#Define a function to train the first model of the two sequential models:
def run_epoch(model, optimizer, dataloaders, attr_criterion, n_epochs):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0
    
    for _epoch in range(n_epochs):
        for phase, loader in dataloaders.items():
            if phase == 'TRAIN':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0
            #best_val_acc = 0
            train_loss_meter = AverageMeter()
            train_acc_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            #running_fscore = 0.0

            with torch.set_grad_enabled(phase == "TRAIN"):
                for _, (inputs, y_true, concept_labels) in enumerate(loader):
                #if attr_criterion is None:
                #    inputs, labels = data
                #    attr_labels, attr_labels_var = None, None
                #else:
                    #inputs, labels, attr_labels = data
                    if n_concepts > 1:
                        concept_labels = [i.long() for i in concept_labels]
                        concept_labels = torch.stack(concept_labels).t()#.float() #N x 312
                    #else:
                    #    if isinstance(concept_labels, list):
                    #        concept_labels = concept_labels[0]
                    concept_labels = concept_labels.unsqueeze(1)
                    concept_labels_var = torch.autograd.Variable(concept_labels).float()
                    concept_labels_var = concept_labels_var.to(DEVICE)
                    #Must transpose to get the correct format of the concept labels:
                    transposed_concepts = concept_labels_var[:,0].t()

                    inputs_var = torch.autograd.Variable(inputs)
                    inputs_var = inputs_var.to(DEVICE)
                    labels_var = torch.autograd.Variable(y_true)
                    labels_var = labels_var.to(DEVICE)
                    #Since Densenet does not have aux_outputs as Inception V3, train and validation losses are equal:
                    outputs = model(inputs_var)
                    #print('Outputs:',outputs)
                    #print('Transposed concepts:',transposed_concepts)
                    losses = []
                    out_start = 0
                    if phase=='TRAIN':
                        for i in range(len(attr_criterion)): #Andrea: Added transposed concepts
                            losses.append(1 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), transposed_concepts[i]))
                    else: #Training/validation where batch size = 1
                        for i in range(len(attr_criterion)): #Andrea: Replaced .squeeze() with [0] for outputs since batch-size = 1
                            losses.append(1 * attr_criterion[i](outputs[i+out_start][0].type(torch.cuda.FloatTensor), transposed_concepts[i]))
                    #if args.bottleneck: #attribute/concept accuracy
                    sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
                    #Andrea: Divided by the number of predictions to avoid accuracies above 100
                    acc = binary_accuracy(sigmoid_outputs, concept_labels)/sigmoid_outputs.shape[0]
                    running_acc += acc.data.cpu().numpy()
                    
                    total_loss = sum(losses)/ n_concepts
                    if phase == 'TRAIN':
                        train_acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
                        train_loss_meter.update(total_loss.item(), inputs.size(0))
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                    else:
                        val_acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
                        val_loss_meter.update(total_loss.item(), inputs.size(0))

                    labels_var = labels_var.detach().cpu().numpy()
                    #Predict the most probable class:
                    #y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    running_loss += total_loss.item()
                    #running_acc += metrics.accuracy_score(y_true, y_pred) 
                    #running_fscore += metrics.f1_score(y_true, y_pred, average='macro')
            
            mean_loss = running_loss / len(loader)
            mean_acc = running_acc / len(loader)
            #mean_fscore = running_fscore / len(dataloader)    

            #Fix this to fit:
            if best_val_acc < val_acc_meter.avg:
            #if phase == "VALID" and mean_acc > best_acc:
                best_val_acc = val_acc_meter.avg
                best_model_wts = copy.deepcopy(model.state_dict())
            
            #print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch, mean_loss, mean_acc))
            if phase == 'VALID':
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,val_loss_meter.avg, val_acc_meter.avg))
            else: 
                print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, _epoch,train_loss_meter.avg, train_acc_meter.avg))
    #The best model on the validation set 
    #after all epochs (total epochs) is saved:
    print("Best val Acc: %.4f" % best_val_acc)
    model.load_state_dict(best_model_wts)
    return model

#Check if the dataset works:
output_path = "../../output"
model_save_path = os.path.join(output_path, "Bottleneck_SequentialModelDensenet121_Imagenet.pt")
    
transform_train_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1),
    albu.Resize(620,620),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.augmentations.geometric.rotate.RandomRotate90(),
    albu.ColorJitter (brightness=1, contrast=(0.8,1.2), saturation=1, hue=0.1, p=0.5),
    albu.Perspective(p=0.5),
    albu.AdvancedBlur(blur_limit=(7,13)),
    albu.augmentations.crops.transforms.RandomResizedCrop(620,620,scale = (0.9, 1.0),p=0.5), #Resized to 299, 299 here as well!
    albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
    ToTensorV2()    
    ])

transform_val_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1),
    albu.Resize(620,620),
    albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
    ToTensorV2()
    ])


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
#Import the overview df
#Order of the concepts in the df are: MA, HE, SoftEx, HardEx, NV, IRMA
overview_conceptDF = pd.read_csv('FGADR_Concept_DR_annotations.csv',index_col = 'Unnamed: 0')

############# Count number of observations per concept ######################
#Can then weight the loss functions for each concept:
concept0_observations = 0
concept1_observations = 0
concept2_observations = 0
concept3_observations = 0
concept4_observations = 0
concept5_observations = 0
#NB! Only use the files for training!
for _file in train_filepath:
    #print('Name of file:', _file)
    image_name = os.path.normpath(_file).split(os.sep)[-1]
    my_row = overview_conceptDF.loc[overview_conceptDF['image_name']==image_name]
    #If the values is 1, then the concept is present
    if my_row.iloc[0,1]==1:
        concept0_observations += 1
    if my_row.iloc[0,2]==1:
        concept1_observations += 1
    if my_row.iloc[0,3]==1:
        concept2_observations += 1
    if my_row.iloc[0,4]==1:
        concept3_observations += 1
    if my_row.iloc[0,5]==1:
        concept4_observations += 1
    if my_row.iloc[0,6]==1:
        concept5_observations += 1
concept_sample_count = np.array([concept0_observations,concept1_observations,concept2_observations,concept3_observations,concept4_observations,concept5_observations])
concept_imbalance = 1. / concept_sample_count
#Try to increase the values for higher loss and effect during training...
concept_imbalance = len(train_filepath) * concept_imbalance
print('Concept ratios:',concept_imbalance)



train_dataset = Dataset(train_filepath,concept_df = overview_conceptDF,augmentation = transform_train_clahe)
valid_dataset = Dataset(valid_filepath,concept_df = overview_conceptDF,augmentation = transform_val_clahe)
#Since the loss are weighted by the concept imbalance, we do not need weighted sampling in the dataloader
train_loader = DataLoader(train_dataset, batch_size=8,shuffle = True, num_workers=8)
#NB! If batch-size increases, the attribution loss calculation must be modified!
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

########### New code specifically for running Densenet (not Inception V3) --------------------
#Customized forward function for Densenet:
def forward(self,x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    #Add an out_list for getting output from all the concepts:
    out_list = []
    for fc in self.classifier:
        out_list.append(fc(out))
    return out_list


print('Loading original model with imagenet weights...')
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
#Then load in the weights for the best performing model on the DR datasets (with 5 DR classes):
#n_classes = 5 
num_in_features = model.classifier.in_features
#model.classifier = nn.Linear(num_in_features, n_classes)
#chkpoint_path = '../../output/CroppedKaggle_Densenet121_100epochs.pt'
#chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
#print('Loading the best model weights for DR detection...')
#model.load_state_dict(chkpoint)

#Create the bottleneck layer with one output per concept
model.classifier = nn.ModuleList()
for i in range(n_concepts):
    model.classifier.append(FC(num_in_features, 1, expand_dim=False))
#print(model)

model.forward = types.MethodType(forward, model)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters()) 
#criterion = nn.CrossEntropyLoss() #This one is not really applied
concept_criterion = [] #separate criterion (loss function) for each attribute/concept
for ratio in concept_imbalance:
    concept_criterion.append(nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(DEVICE)))

best_model = run_epoch(
        model=model,
        optimizer = optimizer, 
        dataloaders={
            "TRAIN": train_loader,
            "VALID": valid_loader
        }, 
        #criterion=criterion, 
        attr_criterion=concept_criterion, 
        n_epochs=100
        )
#Save best model (based on validation set)
torch.save(best_model.state_dict(), model_save_path)

