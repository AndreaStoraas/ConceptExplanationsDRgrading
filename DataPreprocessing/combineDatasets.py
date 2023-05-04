import pandas as pd
import numpy as np
import shutil
import os
import random

#Combine the train, valid and test parts for the three following 
#DR graded datasets:
#APTOS2019, DRDetection and MESSIDOR-2

#NB! First, the images must be split into train, validation and test sets
#By running sortRawdataAPTOS.py, sortRawdataDiabetic.py and sortRawdataMESSIDOR2.py
# AND then crop out black parts of the images by running preprocessImages.py
aptos_train = '../Data/CroppedDataKaggle/CroppedTrainAPTOS'
aptos_valid = '../Data/CroppedDataKaggle/CroppedValidAPTOS'
aptos_test = '../Data/CroppedDataKaggle/CroppedTestAPTOS'

drdetection_train = '../Data/CroppedDataKaggle/CroppedTrainDRDetection'
drdetection_valid = '../Data/CroppedDataKaggle/CroppedValidDRDetection'
drdetection_test = '../Data/CroppedDataKaggle/CroppedTestDRDetection'

messidor_train = '../Data/CroppedDataKaggle/CroppedTrainMESSIDOR'
messidor_valid = '../Data/CroppedDataKaggle/CroppedValidMESSIDOR'
messidor_test = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR'

#First list class distribution in the training sets for the 
#three datasets separately:
print('APTOS dataset:')
for i in range(5):
    class_folder = os.path.join(aptos_train,str(i))
    class_list = os.listdir(class_folder)
    print('Looking at class:',i)
    print('NUmber of images:',len(class_list))


print('DRDetection dataset:')
for i in range(5):
    class_folder = os.path.join(drdetection_train,str(i))
    class_list = os.listdir(class_folder)
    print('Looking at class:',i)
    print('NUmber of images:',len(class_list))


print('MESSIDOR-2 dataset:')
for i in range(5):
    class_folder = os.path.join(messidor_train,str(i))
    class_list = os.listdir(class_folder)
    print('Looking at class:',i)
    print('NUmber of images:',len(class_list))

#Next, for each dataset, copy images for one class over to the combined dataset folder
#Start with combined training folder:
#APTOS:
#For each class:
for i in range(5):
    #First for training data:
    class_folder_train = os.path.join(aptos_train,str(i))
    class_list_train = os.listdir(class_folder_train)
    target_folder_train = os.path.join('../Data/CroppedDataKaggle/CroppedTrainCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_train:
        source_path = os.path.join(class_folder_train,_img)
        target_path = os.path.join(target_folder_train, _img)
        shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(aptos_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('../Data/CroppedDataKaggle/CroppedValidCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(aptos_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('../Data/CroppedDataKaggle/CroppedTestCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        shutil.copy(source_path, target_path)
    
#Repeat for DRDetection:
for i in range(5):
    class_folder = os.path.join(drdetection_train,str(i))
    class_list = os.listdir(class_folder)
    target_folder = os.path.join('../Data/CroppedDataKaggle/CroppedTrainCombinedXL',str(i))
    for _img in class_list:
        source_path = os.path.join(class_folder,_img)
        target_path = os.path.join(target_folder, _img)
        shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(drdetection_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('../Data/CroppedDataKaggle/CroppedValidCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(drdetection_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('../Data/CroppedDataKaggle/CroppedTestCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        shutil.copy(source_path, target_path)
    
#And for MESSIDOR:
for i in range(5):
    class_folder = os.path.join(messidor_train,str(i))
    class_list = os.listdir(class_folder)
    target_folder = os.path.join('../Data/CroppedDataKaggle/CroppedTrainCombinedXL',str(i))
    for _img in class_list:
        source_path = os.path.join(class_folder,_img)
        target_path = os.path.join(target_folder, _img)
        shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(messidor_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('../Data/CroppedDataKaggle/CroppedValidCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(messidor_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('../Data/CroppedDataKaggle/CroppedTestCombinedXL',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        shutil.copy(source_path, target_path)

#Check the number of images for each class for combined training
for i in range(5):
    class_folder = os.path.join('../Data/CroppedDataKaggle/CroppedTrainCombinedXL',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at TRAINING class:',i)
    print('Number of images:',len(class_list))

for i in range(5):
    class_folder = os.path.join('../Data/CroppedDataKaggle/CroppedValidCombinedXL',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at VALIDATION class:',i)
    print('Number of images:',len(class_list))

for i in range(5):
    class_folder = os.path.join('../Data/CroppedDataKaggle/CroppedTestCombinedXL',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at TESTING class:',i)
    print('Number of images:',len(class_list))
