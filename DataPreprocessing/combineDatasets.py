import pandas as pd
import numpy as np
import shutil
import os
import random

#Combine the train, valid and test parts for all the three 
#graded datasets:
#APTOS2019, DRDetection and MESSIDOR-2

aptos_train = 'Data/TrainAPTOS'
aptos_valid = 'Data/ValidAPTOS'
aptos_test = 'Data/TestAPTOS'

drdetection_train = 'Data/TrainDRDetection'
drdetection_valid = 'Data/ValidDRDetection'
drdetection_test = 'Data/TestDRDetection'

messidor_train = 'Data/TrainMESSIDOR'
messidor_valid = 'Data/ValidMESSIDOR'
messidor_test = 'Data/TestMESSIDOR'

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
'''
#Next, for each dataset, copy images for one class over to the combined dataset folder
#Start with combined training folder:
#APTOS:
#For each class:
for i in range(5):
    #First for training data:
    class_folder_train = os.path.join(aptos_train,str(i))
    class_list_train = os.listdir(class_folder_train)
    target_folder_train = os.path.join('Data/TrainCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_train:
        source_path = os.path.join(class_folder_train,_img)
        target_path = os.path.join(target_folder_train, _img)
        #shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(aptos_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('Data/ValidCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        #shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(aptos_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('Data/TestCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        #shutil.copy(source_path, target_path)
    
#Repeat for DRDetection:
for i in range(5):
    class_folder = os.path.join(drdetection_train,str(i))
    class_list = os.listdir(class_folder)
    target_folder = os.path.join('Data/TrainCombined',str(i))
    for _img in class_list:
        source_path = os.path.join(class_folder,_img)
        target_path = os.path.join(target_folder, _img)
        #shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(drdetection_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('Data/ValidCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        #shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(drdetection_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('Data/TestCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        #shutil.copy(source_path, target_path)
    
#And for MESSIDOR:
for i in range(5):
    class_folder = os.path.join(messidor_train,str(i))
    class_list = os.listdir(class_folder)
    target_folder = os.path.join('Data/TrainCombined',str(i))
    for _img in class_list:
        source_path = os.path.join(class_folder,_img)
        target_path = os.path.join(target_folder, _img)
        #shutil.copy(source_path, target_path)
    #Then for validation data:
    class_folder_valid = os.path.join(messidor_valid,str(i))
    class_list_valid = os.listdir(class_folder_valid)
    target_folder_valid = os.path.join('Data/ValidCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_valid:
        source_path = os.path.join(class_folder_valid,_img)
        target_path = os.path.join(target_folder_valid, _img)
        #shutil.copy(source_path, target_path)
    #...And finally for test data:
    class_folder_test = os.path.join(messidor_test,str(i))
    class_list_test = os.listdir(class_folder_test)
    target_folder_test = os.path.join('Data/TestCombined',str(i))
    #For each image belonging to this class:
    for _img in class_list_test:
        source_path = os.path.join(class_folder_test,_img)
        target_path = os.path.join(target_folder_test, _img)
        #shutil.copy(source_path, target_path)

#Check the number of images for each class for combined training
for i in range(5):
    class_folder = os.path.join('Data/TrainCombined',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at TRAINING class:',i)
    print('Number of images:',len(class_list))

for i in range(5):
    class_folder = os.path.join('Data/ValidCombined',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at VALIDATION class:',i)
    print('Number of images:',len(class_list))

for i in range(5):
    class_folder = os.path.join('Data/TestCombined',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at TESTING class:',i)
    print('Number of images:',len(class_list))
'''