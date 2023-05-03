import pandas as pd
import numpy as np
import shutil
import os
import random

image_path = 'Data/FGADR-Seg-set/Original_Images'

#Assume that the gradings are the right column in the .csv file:
#The resulting class distribution is at least consistent with 
#the dataset paper 
annotated_df = pd.read_csv('Data/FGADR-Seg-set/DR_Seg_Grading_Label.csv', header = None,
names = ['Image','DR-grade'])
image_list = os.listdir(image_path)
#print('Number of annotated images:',annotated_df.shape[0])
#print('Number of original images in folder:', len(image_list))
#print(annotated_df['DR-grade'].value_counts())


#Check that all images in folder have been annotated:
for _img in image_list:
    if _img not in annotated_df['Image'].tolist():
        print('This image has not been annotated!')
        print(_img)

#Sort the images based on DR grade:
#Can then be used for testing the final DR model
def sortImages():
    for i in range(annotated_df.shape[0]):
        img_name = annotated_df.iloc[i,0]
        img_class = annotated_df.iloc[i,1]
        source_path = os.path.join(image_path,img_name)
        if img_class == 0:
            target_path = os.path.join('Data/FGADR-Seg-set/FGADR-Sorted/0',img_name)
        elif img_class == 1:
            target_path = os.path.join('Data/FGADR-Seg-set/FGADR-Sorted/1',img_name)
        elif img_class == 2:
            target_path = os.path.join('Data/FGADR-Seg-set/FGADR-Sorted/2',img_name)
        elif img_class == 3:
            target_path = os.path.join('Data/FGADR-Seg-set/FGADR-Sorted/3',img_name)
        elif img_class == 4:
            target_path = os.path.join('Data/FGADR-Seg-set/FGADR-Sorted/4',img_name)
        else:
            print('Class number is not defined correctly!')
        shutil.copy(source_path,target_path)

labeled_folder = 'Data/FGADR-Seg-set/FGADR-Sorted'
class_0 = os.path.join(labeled_folder,'0')
class_1 = os.path.join(labeled_folder,'1')
class_2 = os.path.join(labeled_folder,'2')
class_3 = os.path.join(labeled_folder,'3')
class_4 = os.path.join(labeled_folder,'4')

def splitTrainValTest():
    for _class in range(5):
        #Get the correct folder, matching with the class:
        folderPath = os.path.join(labeled_folder,str(_class))
        #list all files in folder:
        classFiles = os.listdir(folderPath)
        #Select 80% for training, 10% for validation, 10% for testing
        num_train = int(len(classFiles)*0.8)
        num_valid = int(len(classFiles)*0.1)
        print('Looking at DR class:',_class)
        print('Number of images in class:',len(classFiles))
        print('Number of images for training:',num_train)
        print('Number of images for validation:',num_valid)
        print('Number of images for testing:',  len(classFiles)-(num_train+num_valid))
        #Pick 80% randomly from the list
        #Convert to set for easier substraction
        trainFiles = set(random.sample(classFiles,num_train))
        #The rest is for validation and testing
        validTestFiles = set(classFiles) - trainFiles
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
            sourceFile = os.path.join(folderPath,tFile)
            targetPath = os.path.join('Data/TrainFGADR',str(_class),tFile)
            shutil.copy(sourceFile,targetPath)
        #Check that no duplicates in validation and test files:
        for vFile in validFiles:
            if vFile in testFiles:
                print('Same image in valid and testing list! Something went wrong!')
        #Move validation files to Valid
        for vFile in validFiles:
            sourceFile = os.path.join(folderPath,vFile)
            targetPath = os.path.join('Data/ValidFGADR',str(_class),vFile)
            shutil.copy(sourceFile,targetPath)
        #Move test files to Test
        for testFile in testFiles:
            sourceFile = os.path.join(folderPath,testFile)
            targetPath = os.path.join('Data/TestFGADR',str(_class),testFile)
            shutil.copy(sourceFile,targetPath)

def moveToCombined():
    for _class in range(5):
        #Get the correct folder, matching with the class:
        folderPath = os.path.join('Data/CroppedData/CroppedTestFGADR',str(_class))
        #list all files in folder:
        classFiles = os.listdir(folderPath)
        for _file in classFiles:
            source_path = os.path.join(folderPath,_file)
            target_path = os.path.join('Data/CroppedData/CroppedTestCombinedXL',str(_class),_file)
            shutil.copy(source_path,target_path)

#sortImages()
#splitTrainValTest()
#moveToCombined()
#List the number of images in each class
#to make sure it is correct:
for i in range(5):
    class_folder = os.path.join('Data/CroppedData/CroppedTestCombinedXL',str(i))
    class_list = os.listdir(class_folder)
    print('Looking at class:',i)
    print('Number of images:',len(class_list))


        

