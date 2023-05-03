import os
import torch
import copy
import types
#import pickle
import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

import random
import argparse
from torch import functional
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
#Import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset
from models import ModelXtoC,ModelXtoChat_ChatToY,ModelOracleCtoY
from analysis import binary_accuracy, AverageMeter
#For Densenet bottleneck model:
from template_model import FC

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
        if self.augmentation:
            image = self.augmentation(image=image)['image']    
        # get the image name:
        image_name = os.path.normpath(image_path).split(os.sep)[-1]
        # get the corresponding presence/absence of concepts
        df_row = self.concept_df.loc[self.concept_df['image_name']==image_name]
        concept_annotations = df_row.iloc[0,1:5].values.tolist()
        #print('Correct row:',df_row)
        #print('Concept annotations:',concept_annotations)
        return image, concept_annotations
        
    def __len__(self):
        return len(self.filepaths)

def test_model(model, dataloader):
    all_predicted = []
    all_true = []
    #Make sure the params are freezed:
    model.eval()
    running_acc = 0.0
    running_f1 = 0.0
    with torch.set_grad_enabled(False):
        for param in model.parameters():
            param.requires_grad = False
        for i, (inputs, concept_labels) in enumerate(dataloader):
            if n_concepts > 1:
                concept_labels = [i.long() for i in concept_labels]
                concept_labels = torch.stack(concept_labels).t()#.float() #N x 312
            concept_labels = concept_labels.unsqueeze(1)
            concept_labels_var = torch.autograd.Variable(concept_labels).float()
            concept_labels_var = concept_labels_var.to(DEVICE)
            #Must transpose to get the correct format of the concept labels:
            transposed_concepts = concept_labels_var[:,0].t()

            inputs_var = torch.autograd.Variable(inputs)
            inputs_var = inputs_var.to(DEVICE)
            
            #inputs = inputs.to(DEVICE)
            #y_true = y_true.to(DEVICE)
            y_pred = model(inputs_var)
            #print('Y_pred without sigmoid:',y_pred)
            concept_labels_var = concept_labels_var.detach().cpu().numpy()
            #Predict the most probable class:
            #NB! This is only to test the concept accuracy, 
            #NOT when we want to pass the values on to the last model predicting the DR classes!
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(y_pred, dim=1)) #<- Get the predictions to lie between 0 and 1
            #Andrea: Divided by the number of predictions to avoid accuracies above 100
            acc = binary_accuracy(sigmoid_outputs, concept_labels)/sigmoid_outputs.shape[0]
            running_acc += acc.data.cpu().numpy()
            #print('Sigmoid outputs:',sigmoid_outputs)
            #all_predicted.append(y_pred)
            #Convert from sigmoid to True/False:
            binary_pred = sigmoid_outputs.cpu() >= 0.5
            all_predicted.append(binary_pred)
            #all_predicted.append(sigmoid_outputs)
            all_true.append(concept_labels_var)

    mean_acc = running_acc / len(dataloader)
    #mean_f1 = running_f1 / len(dataloader)
    print('Overall accuracy test set:',mean_acc)
    #print('Overall F1 score test set:',mean_f1)
    #Flatten the list
    all_predicted = [a.squeeze() for a in all_predicted]
    all_true = [a.squeeze() for a in all_true]
    #print('Predicted values:')
    #print(all_predicted)
    return all_predicted, all_true


#NB! If training data, we use the full transformation
#If testing/validation data, we use only CLAHE + normalization
transform_test_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1),
        albu.Resize(620,620),
        #albu.Resize(299,299),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])

transform_train_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1),
    albu.Resize(620,620),
    #albu.Resize(299,299),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.augmentations.geometric.rotate.RandomRotate90(),
    albu.ColorJitter (brightness=1, contrast=(0.8,1.2), saturation=1, hue=0.1, p=0.5),
    albu.Perspective(p=0.5),
    albu.AdvancedBlur(blur_limit=(7,13)),
    albu.augmentations.crops.transforms.RandomResizedCrop(620,620,scale = (0.9, 1.0),p=0.5),
    albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
    ToTensorV2()    
    ])

n_concepts = 4
n_classes = 5
test_folder = '../../Data/CroppedDataKaggle/CroppedTestBottleneckCombined'
#Add all filepaths for the test dataset to a list
test_filepath = []
all_files = os.listdir(test_folder)
for _img in all_files:
    single_path = os.path.join(test_folder,_img)
    test_filepath.append(single_path)
print('Number of files:',len(all_files))
print('Length of test files:',len(test_filepath))

overview_conceptDF = pd.read_csv('../FGADR_DDR_IDRiD_DiaretDB1_conceptOverview.csv',index_col = 'Unnamed: 0')
test_dataset = Dataset(test_filepath,concept_df = overview_conceptDF,augmentation = transform_test_clahe)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)

#model1 = ModelXtoC(pretrained=True, freeze=False, num_classes=n_classes, use_aux=True,
#                      n_attributes=n_concepts,  expand_dim=0, three_class=False) #three_class = False since we want binary classifications of each concept

#For densenet 121:
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
model1 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
#Then customize the last layer:
num_in_features = model1.classifier.in_features
#Create the bottleneck layer with one output per concept for Densenet
model1.classifier = nn.ModuleList()
for i in range(n_concepts):
    model1.classifier.append(FC(num_in_features, 1, expand_dim=False))


print('Loading in the weights for the trained model...')
chkpoint_path = '../../output/BottleneckCombined_SequentialModel.pt'
chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
model1.load_state_dict(chkpoint)
#For Densenet: Apply the modified forward function:
model1.forward = types.MethodType(forward, model1)
model1 = model1.to(DEVICE)

if __name__ == "__main__":
    predictions, y_true = test_model(model1,test_loader)
    #Again, flatten the list
    predictions = [item.tolist() for item in predictions]
    y_true = [item.tolist() for item in y_true]
    #Create one large list for appropriate accuracies...
    flattened_pred = [item for sublist in predictions for item in sublist]
    flattened_true = [item for sublist in y_true for item in sublist]
    #flattened_predSoftEx = [item[5] for item in predictions]
    #flattened_trueSoftEx = [item[5] for item in y_true]
    #print('All predicted values:')
    #print(predictions)
    print('Number of predictions:',len(predictions))
    print('If not complete match for all concepts, accuracy = 0:')
    print('Overall accuracy:',metrics.accuracy_score(y_true,predictions))
    print(metrics.multilabel_confusion_matrix(y_true, predictions))
    print(metrics.classification_report(y_true, predictions))
    print('Inspecting metrics for overall predictions (six concepts * 189 observations):')
    print('Accuracy overall predictions:',metrics.accuracy_score(flattened_true,flattened_pred))
    print(metrics.classification_report(flattened_true, flattened_pred))