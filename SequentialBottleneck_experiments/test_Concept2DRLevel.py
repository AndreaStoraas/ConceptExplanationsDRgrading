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
        all_paths.append(single_path)
        #Add the full image path to image_list:
    test_filepath += all_paths
print('Length of testing files:',len(test_filepath))
print('First filepath:',test_filepath[0])

#Load in the raw concept df predictions for the combined XL testset:
conceptPredictions_test = pd.read_csv('SequentialModelOutput/rawDensenet121_conceptPredictions_test.csv',index_col = 'Unnamed: 0')

#Create the dataset:
test_dataset = Dataset(filepaths = test_filepath,concept_df = conceptPredictions_test)
print('Shape of the concept DF:',conceptPredictions_test.shape)
#Create the dataloader:
#NB! Want to use weighted random sampler to take DR class imbalance into account
#This is not implemented in the calculation of the loss for this model
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
    predictions, y_true = test_model(model,test_loader)
    #Again, flatten the list
    predictions = [item.tolist() for item in predictions]
    y_true = [item.tolist() for item in y_true]
    #print('All predicted values:')
    #print(predictions)
    print('Number of predictions:',len(predictions))
    #Calculating performance metrics (macro = unweighted mean across all samples):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true,predictions, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, predictions)
    print('Showing results on:',test_folder)
    print('Precision:',precision)
    print('Recall:',recall)
    print('F1 score:',fscore)
    print('Support:',support)
    print('Balanced accuracy',metrics.balanced_accuracy_score(y_true,predictions))
    print('MCC:', mcc)
    print('Overall accuracy:',metrics.accuracy_score(y_true,predictions))
    print('Results for each class separately:')
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true,predictions)
    print('Precision separate:',precision)
    print('Recall separate:',recall)
    print('F1 score separate:',fscore)
    print('Support separate:',support)
    #Try to plot all predictions in one single confusion matrix:
    print(metrics.confusion_matrix(y_true, predictions))
    #print(metrics.multilabel_confusion_matrix(y_true, predictions))
    print(metrics.classification_report(y_true, predictions))