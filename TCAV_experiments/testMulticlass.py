import os
import random
import argparse
import torch
import copy

import numpy as np
import pandas as pd
from torch import functional
import torch.nn as nn

from sklearn import metrics

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset


random.seed(0)
np.random.seed(0)
#Should use torch.manual_seed: https://pytorch.org/vision/stable/transforms.html
torch.manual_seed(0)

#Path to the test data folder:
test_path = 'Data/CroppedDataKaggle/CroppedTestCombinedXL'
print('This is the test path:',test_path)
#Define the device:
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#=========================================
# Helper functions and Datasets
#=========================================


#Define dataset class:
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        filepaths (list): list of paths to images
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """

    
    def __init__(
            self, 
            filepaths, 
            augmentation=None, 
    ):
        self.filepaths = filepaths
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        image_path = self.filepaths[i]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
        return image, label
        
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
        for i, (inputs, y_true) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)
            y_pred = model(inputs)
            #print('Y_pred:',y_pred)
            y_true = y_true.detach().cpu().numpy()
            #Predict the most probable class:
            y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            running_acc += metrics.accuracy_score(y_true, y_pred) 
            running_f1 += metrics.f1_score(y_true, y_pred, average = 'macro')
            all_predicted.append(y_pred)
            all_true.append(y_true)

    mean_acc = running_acc / len(dataloader)
    mean_f1 = running_f1 / len(dataloader)
    print('Overall accuracy test set:',mean_acc)
    print('Overall F1 score test set:',mean_f1)
    #Flatten the list
    all_predicted = [a.squeeze() for a in all_predicted]
    all_true = [a.squeeze() for a in all_true]
    print('Predicted values:')
    print(all_predicted)
    return all_predicted, all_true


#Define the dataloader:

transform_test_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1),
        albu.Resize(620,620),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])

n_classes = 5
#Add all filepaths for the test dataset to a list
small_list = [os.path.join(test_path, str(class_id)) for class_id in range(n_classes)]
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
print('Length of test files:',len(test_filepath))

test_dataset = Dataset(test_filepath,augmentation = transform_test_clahe)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
#Load the trained model:
model = models.densenet121()
model.classifier = nn.Linear(model.classifier.in_features, n_classes)
#Load the checkpoints:
print('Loading in the weights for the trained model...')
chkpoint_path = 'output/CroppedKaggle_Densenet121_100epochs.pt'
chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
model.load_state_dict(chkpoint)
model = model.to(DEVICE)

if __name__ == "__main__":
    predictions, y_true = test_model(model,test_loader)
    #Again, flatten the list
    predictions = [item.tolist() for item in predictions]
    y_true = [item.tolist() for item in y_true]
    print('Number of predictions:',len(predictions))
    #Calculating performance metrics (macro = unweighted mean across all samples):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true,predictions, average='macro')
    mcc = metrics.matthews_corrcoef(y_true, predictions)
    print('Showing results on:',test_path)
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
    print(metrics.classification_report(y_true, predictions))
    #For TCAV experiments:
    #Create a df with image name, predicted class and true class for the representative test set:
    #my_df = pd.DataFrame(test_filepath)
    #my_df['Predicted class'] = predictions
    #my_df['Real class'] = y_true
    #my_df.to_csv('./classificationOverviewRepresentativeTestSet.csv',index=False)

