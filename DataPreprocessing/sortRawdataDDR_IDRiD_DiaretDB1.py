import pandas as pd
import numpy as np
import shutil
import os
import random

all_images_path = 'Data/CroppedDataKaggle/CroppedDDR_noAbnormalities'

def splitTrainValTest():
    allFiles = os.listdir(all_images_path)
    #Select 80% for training, 10% for validation, 10% for testing
    num_train = int(len(allFiles)*0.8)
    num_valid = int(len(allFiles)*0.1)
    print('Total number of images:',len(allFiles))
    print('Number of images for training:',num_train)
    print('Number of images for validation:',num_valid)
    print('Number of images for testing:',  len(allFiles)-(num_train+num_valid))
    #Pick 80% randomly from the list
    #Convert to set for easier substraction
    trainFiles = set(random.sample(allFiles,num_train))
    #The rest is for validation and testing
    validTestFiles = set(allFiles) - trainFiles
    #Pick the validation files:
    validFiles = set(random.sample(validTestFiles,num_valid))
    #The rest will be for testing
    testFiles = validTestFiles - validFiles
    #Convert back to list again:
    trainFiles = list(trainFiles)
    validFiles = list(validFiles)
    testFiles = list(testFiles)
    print(len(trainFiles))
    print(len(validFiles))
    print(len(testFiles))
    #Check that no duplicates in the train and validation/test files:
    for tFile in trainFiles:
        if tFile in validFiles:
            print('Same image in valid and training list! Something went wrong here!')
        elif tFile in testFiles:
            print('Same image in TEST and training list! Something went wrong here!')
    #Move training files to Train
    for tFile in trainFiles:
        sourceFile = os.path.join(all_images_path,tFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedTrainIDRiD',tFile)
        shutil.copy(sourceFile,targetPath)
    #Check that no duplicates in validation and test files:
    for vFile in validFiles:
        if vFile in testFiles:
            print('Same image in valid and testing list! Something went wrong!')
    #Move validation files to Valid
    for vFile in validFiles:
        sourceFile = os.path.join(all_images_path,vFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedValidIDRiD',vFile)
        shutil.copy(sourceFile,targetPath)
    #Move test files to Test
    for testFile in testFiles:
        sourceFile = os.path.join(all_images_path,testFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedTestIDRiD',testFile)
        shutil.copy(sourceFile,targetPath)

def pick1000_splitTrainValTest():
    allFiles = os.listdir(all_images_path)
    #Select 80% for training, 10% for validation, 10% for testing
    num_train = 800
    num_valid = 100
    num_test = 100
    print('Total number of images:',len(allFiles))
    print('Number of images for training:',num_train)
    print('Number of images for validation:',num_valid)
    print('Number of images for testing:',  num_test)
    #First pick 1000 images randomly from all the files:
    selectedFiles = random.sample(allFiles, 1000)
    #Pick 80% randomly from the list
    #Convert to set for easier substraction
    trainFiles = set(random.sample(selectedFiles,num_train))
    #The rest is for validation and testing
    validTestFiles = set(selectedFiles) - trainFiles
    #Pick the validation files:
    validFiles = set(random.sample(validTestFiles,num_valid))
    #The rest will be for testing
    testFiles = validTestFiles - validFiles
    #Convert back to list again:
    trainFiles = list(trainFiles)
    validFiles = list(validFiles)
    testFiles = list(testFiles)
    print(len(trainFiles))
    print(len(validFiles))
    print(len(testFiles))
    #Check that no duplicates in the train and validation/test files:
    for tFile in trainFiles:
        if tFile in validFiles:
            print('Same image in valid and training list! Something went wrong here!')
        elif tFile in testFiles:
            print('Same image in TEST and training list! Something went wrong here!')
    #Move training files to Train
    for tFile in trainFiles:
        sourceFile = os.path.join(all_images_path,tFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedTrainDDR',tFile)
        shutil.copy(sourceFile,targetPath)
    #Check that no duplicates in validation and test files:
    for vFile in validFiles:
        if vFile in testFiles:
            print('Same image in valid and testing list! Something went wrong!')
    #Move validation files to Valid
    for vFile in validFiles:
        sourceFile = os.path.join(all_images_path,vFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedValidDDR',vFile)
        shutil.copy(sourceFile,targetPath)
    #Move test files to Test
    for testFile in testFiles:
        sourceFile = os.path.join(all_images_path,testFile)
        targetPath = os.path.join('Data/CroppedDataKaggle/CroppedTestDDR',testFile)
        shutil.copy(sourceFile,targetPath)

def copy_toCombinedFolder():
    source_folder = 'Data/CroppedDataKaggle/CroppedTestFGADR'
    class_folders = os.listdir(source_folder)
    target_folder = 'Data/CroppedDataKaggle/CroppedTestBottleneckCombined'
    for _class in class_folders:
        for _file in os.listdir(os.path.join(source_folder,_class)):
            file_path = os.path.join(source_folder,_class,_file)
            shutil.copy(file_path,target_folder)
    #print('Number of files in source folder:',len(os.listdir(source_folder)))
    print('Number of files in target folder:',len(os.listdir(target_folder)))

#splitTrainValTest()
#pick1000_splitTrainValTest()
#copy_toCombinedFolder()
print('Total number of test images FGADR:',len(os.listdir('Data/CroppedDataKaggle/CroppedTestFGADR/0'))+len(os.listdir('Data/CroppedDataKaggle/CroppedTestFGADR/1'))+len(os.listdir('Data/CroppedDataKaggle/CroppedTestFGADR/2'))+len(os.listdir('Data/CroppedDataKaggle/CroppedTestFGADR/3'))+len(os.listdir('Data/CroppedDataKaggle/CroppedTestFGADR/4')))
print('Total number of test images DDR:',len(os.listdir('Data/CroppedDataKaggle/CroppedTestDDR')))
print('Total number of test images IDRiD:',len(os.listdir('Data/CroppedDataKaggle/CroppedTestIDRiD')))
print('Total number of test images DiaretDB1:',len(os.listdir('Data/CroppedDataKaggle/CroppedTestDiaretDB1')))

#print('Total number of valid images DDR:',len(os.listdir('Data/CroppedDataKaggle/CroppedValidDDR')))
#print('Total number of test images DDR:',len(os.listdir('Data/CroppedDataKaggle/CroppedTestDDR')))