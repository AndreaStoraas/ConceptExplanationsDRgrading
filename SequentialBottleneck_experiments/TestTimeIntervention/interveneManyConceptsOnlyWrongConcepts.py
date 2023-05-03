import os
import torch
import copy
import numpy as np
import pandas as pd
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
            intervention_concept_list #Which concepts to correct for
    ):
        self.filepaths = filepaths
        self.concept_df = concept_df
        self.intervention_concept_list = intervention_concept_list

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
        #For concept intervention, need to replace the predicted concept with 95th/5th percentile from training set:
        #If true concept = 1, use 95th, if true concept = 0, use 5th percentile
        true_concepts = df_row.iloc[0,2]
        #Again, the concepts are interpreted as a string-list
        #We need to convert them to a proper list of float-values:
        true_concepts = true_concepts.strip('"')
        true_concepts = true_concepts.replace('tensor([','')
        true_concepts = true_concepts.replace('])','')
        true_concepts = true_concepts.strip('[]')
        true_concepts = list(true_concepts.split(','))
        true_concepts = list(map(int,true_concepts))
        #print('True concepts converted to integer:',true_concepts)
        #print('Concept data before intervention:',concept_data)
        #Replace predicted concept value with percentile from training set predictions
        if 'MA' in self.intervention_concept_list:
            #Added: Check if predicted concept is correct
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[0]
            concept_presence = true_concepts[0]
            print('Is the concept present?',concept_presence)
            #print('Image path:',image_path)
            print('Predicted concept value:',pred_concept)
            #If the concept is present, pick the 99th percentile value
            if concept_presence == 1:
                #If predicted concept = 0, but the concept is present, 
                #we correct the predicted value:
                if pred_concept<0:
                    print('Predicted not present: must correct the predicted value!')
                    concept_data[0] = MA_percentile99
            else:
                #No concept present, but predicted concept = 1
                if pred_concept>=0:
                    concept_data[0] = MA_percentile1
        if 'hemorrhages' in self.intervention_concept_list:
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[1]
            concept_presence = true_concepts[1]
            if concept_presence == 1:
                if pred_concept<0:
                    concept_data[1] = hemorrhages_percentile99
            else:
                if pred_concept>=0:
                    concept_data[1] = hemorrhages_percentile1
        if 'softEx' in self.intervention_concept_list:
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[2]
            concept_presence = true_concepts[2]
            if concept_presence == 1:
                if pred_concept<0:
                    concept_data[2] = softEx_percentile99
            else:
                if pred_concept>=0:
                    concept_data[2] = softEx_percentile1
        if 'hardEx' in self.intervention_concept_list:
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[3]
            concept_presence = true_concepts[3]
            if concept_presence == 1:
                if pred_concept<0:
                    concept_data[3] = hardEx_percentile99
            else:
                if pred_concept>=0:
                    concept_data[3] = hardEx_percentile1
        if 'NV' in self.intervention_concept_list:
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[4]
            concept_presence = true_concepts[4]
            print('Is the concept present?',concept_presence)
            print('Predicted concept value:',pred_concept)
            #print('Image path:',image_path)
            if concept_presence == 1:
                if pred_concept<0:
                    print('Predicted to be not present: must correct the predicted value!')
                    concept_data[4] = NV_percentile99
            else:
                if pred_concept>=0:
                    concept_data[4] = NV_percentile1
        if 'IRMA' in self.intervention_concept_list:
            #Negative predicted value = no concept, 0 or above = concept is present
            pred_concept = concept_data[5]
            concept_presence = true_concepts[5]
            print('Is the concept present?',concept_presence)
            print('Predicted concept value:',pred_concept)
            if concept_presence == 1:
                if pred_concept<0:
                    print('Predicted to be not present: must correct the predicted value!')
                    concept_data[5]=IRMA_percentile99
            else:
                if pred_concept>=0:
                    concept_data[5]=IRMA_percentile1
        #else:
        #    print('No concept intervention...')
        #print('Concept data:',concept_data)
        label = df_row.iloc[0,-1]
        #print('Raw concepts:',concept_data)
        return concept_data, label
        
    def __len__(self):
        #When only looking at the wrongly predicted files:
        return len(self.filepaths)
        #return self.concept_df.shape[0]


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
    #print('Predicted values:')
    #print(all_predicted)
    return all_predicted, all_true

#Want the percentiles for predicted concepts from the FGADR training dataset
#Extracted from the best performing bottleneck model (Densenet-121 on six concepts):

#The predicted concepts on the training dataset are already available, since this was used to train the
#DR classification part of the model

predicted_trainConcepts = pd.read_csv('../SequentialModelOutput/rawDensenet121_conceptPredictions_train.csv',index_col='Unnamed: 0')
#print(predicted_trainConcepts.head())
#The order of the concepts are: MA, hemorrhages, soft exudates, hard exudates, NV and IRMA
#Get the predicted concepts
MA_predictions = []
hemorrhages_predictions = []
softEx_preditions = []
hardEx_predictions = []
NV_predictions = []
IRMA_predictions = []
for i in range(predicted_trainConcepts.shape[0]):
    concept_data = predicted_trainConcepts.iloc[i,1]
    #Since these are (of unknown causes) interpreted as a string-list
    #We need to convert them to a proper list of float-values:
    concept_data = concept_data.strip('"')
    concept_data = concept_data.strip('[]')
    concept_data = list(concept_data.split(','))
    concept_data = list(map(float,concept_data))
    #print('All predicted concepts:',concept_data)
    #print('Predicted MA:',concept_data[0])
    MA_predictions.append(concept_data[0])
    hemorrhages_predictions.append(concept_data[1])
    softEx_preditions.append(concept_data[2])
    hardEx_predictions.append(concept_data[3])
    NV_predictions.append(concept_data[4])
    IRMA_predictions.append(concept_data[5])

#Get the 95th percentile and 5th percentile for the predicted concepts:
MA_percentile1 = np.percentile(MA_predictions,1)
MA_percentile99 = np.percentile(MA_predictions,99)
hemorrhages_percentile1 = np.percentile(hemorrhages_predictions,1)
hemorrhages_percentile99 = np.percentile(hemorrhages_predictions,99)
softEx_percentile1 = np.percentile(softEx_preditions,1)
softEx_percentile99 = np.percentile(softEx_preditions,99) 
hardEx_percentile1 = np.percentile(hardEx_predictions,1)
hardEx_percentile99 = np.percentile(hardEx_predictions,99)
NV_percentile1 = np.percentile(NV_predictions,1)
NV_percentile99 = np.percentile(NV_predictions,99)
IRMA_percentile1 = np.percentile(IRMA_predictions,1)
IRMA_percentile99 = np.percentile(IRMA_predictions,99)

print('MA 1st percentile:',MA_percentile1)
print('MA 99th percentile:',MA_percentile99)
print('Hemorrhages 1st percentile:',hemorrhages_percentile1)
print('Hemorrhages 99th percentile:',hemorrhages_percentile99)
print('SoftEx 1st percentile:',softEx_percentile1)
print('SoftEx 99th percentile:',softEx_percentile99)
print('HardEx 1st percentile:',hardEx_percentile1)
print('HardEx 99th percentile:',hardEx_percentile99)
print('NV 1st percentile:',NV_percentile1)
print('NV 99th percentile:',NV_percentile99)
print('IRMA 1st percentile:',IRMA_percentile1)
print('IRMA 99th percentile:',IRMA_percentile99)
#print('NV 100 percentile:',np.percentile(NV_predictions,99))
#print('IRMA 100 percentile:',np.percentile(IRMA_predictions,99))

#Next, we want to replace predicted concepts on the test set with the "true" concept value,
#If concept is present, then insert the 99th percentile value, 
#if not present, then insert the 1st percentile value
#This is implemented in the Dataset class...

#Get the filenames for the test data:
n_classes = 5
n_concepts = 6
test_folder = '../../../Data/CroppedDataKaggle/CroppedTestFGADR'
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
#print('Length of testing files:',len(test_filepath))
#print('First filepath:',test_filepath[0])

#Instead of predicting on all test images, 
#get the DR predictions in the test set that were wrong (and intervene on the predicted concepts):
wrongPredictions_test = pd.read_csv('WrongPredictions_FGADR_testset_Densenet121.csv')
wrongPredictions_files = wrongPredictions_test['0'].tolist()
print('First wrongly predicted file:',wrongPredictions_files[0])
print('Number of wrong predicted files:',len(wrongPredictions_files))


#Load in the raw concept df predictions:
conceptPredictions_test = pd.read_csv('../SequentialModelOutput/rawDensenet121_conceptPredictions_test.csv',index_col = 'Unnamed: 0')


#Create the dataset:
#Only intervene and predict on the wrong cases. NB! Also changed the length of Dataset class!!!
#The concepts to correct for are passed as a list
test_dataset = Dataset(filepaths = wrongPredictions_files,concept_df = conceptPredictions_test, 
    intervention_concept_list=['MA','softEx','hardEx','hemorrhages','NV','IRMA'])#
#Uncomment code below if you want to correct for entire test set instead:
#test_dataset = Dataset(filepaths = test_filepath,concept_df = conceptPredictions_test, 
#    intervention_concept_list=['MA','NV','hardEx','IRMA','softEx','hemorrhages'])

print('Shape of the concept DF:',conceptPredictions_test.shape)
#Create the dataloader:
#NB! Want to use weighted random sampler to take DR class imbalance into account
#This is not implemented in the calculation of the loss for this model
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle = False)

#Get the model:
model = ModelXtoChat_ChatToY(n_class_attr=2, n_attributes=n_concepts,
                                 num_classes=n_classes, expand_dim=0)
print('Loading in the weights for the trained model...')
chkpoint_path = '../../../output/BottleneckDensenet121_SequentialModel_part2.pt'
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

