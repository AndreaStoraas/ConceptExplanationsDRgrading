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

from sklearn import metrics

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

########################
# This code extracts the mislclassified images for DR level on the FGADR test set
# And saved them as a csv-file with name of image, predicted DR level and ground truth DR level
# This csv-file is then used for test time intervention (TTI) in EntireTestset_TTI_onIncorrectImages.py
########################

#Device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',DEVICE)

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


def test_model(model, dataloader):
    all_predicted = []
    all_true = []
    #Make sure the params are freezed:
    model.eval()
    running_acc = 0.0
    with torch.set_grad_enabled(False):
        for param in model.parameters():
            param.requires_grad = False
        for i, (inputs, y_true) in enumerate(dataloader):
            if isinstance(inputs, list):
                inputs = torch.stack(inputs).t().float()
            inputs = torch.tensor(np.asarray(inputs))
            inputs = torch.flatten(inputs, start_dim=1).float()
            inputs_var = torch.autograd.Variable(inputs).cuda()
            inputs_var = inputs_var.to(DEVICE)
            labels_var = torch.autograd.Variable(y_true).cuda()
            labels_var = labels_var.to(DEVICE)
        
            outputs = model(inputs_var)
            acc = accuracy(outputs, y_true, topk=(1,))
            running_acc += acc[0].data.cpu().numpy()
            #Predict the most probable class:
            y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            all_predicted.append(y_pred)
            all_true.append(y_true)
    mean_acc = running_acc / len(dataloader)
    print('Overall accuracy test set:',mean_acc)
    #Flatten the list
    all_predicted = [a.squeeze() for a in all_predicted]
    all_true = [a.squeeze() for a in all_true]
    print('Predicted values:')
    print(all_predicted)
    return all_predicted, all_true

#Get the filenames for the test data:
n_classes = 5
n_concepts = 6
test_folder = '../../Data/CroppedDataKaggle/CroppedTestFGADR'
#Test on entire Combined XL test set:
#test_folder = '../../Data/CroppedDataKaggle/CroppedTestCombinedXL'

small_list = [os.path.join(test_folder, str(class_id)) for class_id in range(n_classes)]
print('Small list testing:', small_list)
test_filepath = []
for _list in small_list:
    all_files = os.listdir(_list)
    print('Number of files:',len(all_files))
    all_paths = []
    #For each image in the class folder
    for _img in all_files:
        single_path = os.path.join(_list,_img)
        #NB! Since we are in a subfolder, we need to remove the first '../'
        #in order for the filepath to match the filepaths in the overview_df:
        single_path = single_path[3:] 
        all_paths.append(single_path)
        #Add the full image path to image_list:
    test_filepath += all_paths
print('Length of testing files:',len(test_filepath))
print('First filepath:',test_filepath[0])

#Load in the raw concept df predictions:
conceptPredictions_test = pd.read_csv('../SequentialModelOutput/rawDensenet121_conceptPredictions_test.csv',index_col = 'Unnamed: 0')

#Create the dataset:
test_dataset = Dataset(filepaths = test_filepath,concept_df = conceptPredictions_test)
print('Shape of the concept DF:',conceptPredictions_test.shape)
#Create the dataloader:
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle = False)

#Get the model:
model = ModelXtoChat_ChatToY(n_class_attr=2, n_attributes=n_concepts,
                                 num_classes=n_classes, expand_dim=0)
print('Loading in the weights for the trained model...')
chkpoint_path = '../../output/BottleneckDensenet121_SequentialModel_part2.pt'
chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
model.load_state_dict(chkpoint)
model.to(DEVICE)

if __name__ == "__main__":
    predictions, y_true= test_model(model,test_loader)
    #Again, flatten the list
    predictions = [item.tolist() for item in predictions]
    y_true = [item.tolist() for item in y_true]
    print('Number of predictions:',len(predictions))
    #For Test time intervention (TTI):
    #Create a df with image name, predicted class and true class:
    my_df = pd.DataFrame(test_filepath)
    my_df['Predicted class'] = predictions
    my_df['Real class'] = y_true
    print(my_df.shape)
    #Select the images where the predictions were wrong
    wrong_df = my_df.loc[my_df['Predicted class']!=my_df['Real class']]
    print(wrong_df.shape)
    print(wrong_df.head())
    wrong_df.to_csv('WrongPredictions_FGADR_testset_Densenet121.csv',index=False)
