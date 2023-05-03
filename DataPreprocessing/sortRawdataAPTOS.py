import pandas as pd
import numpy as np
import shutil
import os
import random

labelFolderName = 'aptos2019-blindness-detection'
imageFolderName = 'aptos2019-blindness-detection/train_images'
#Read in the datalabels and ids:
#label_data = pd.read_csv(os.path.join(labelFolderName,'train.csv'))
#print(label_data.head(20))
#print('Distribution of the diagnoses:')
#print(label_data['diagnosis'].value_counts())
#print('Number of images:',label_data.shape)

#Want to move images into correct folders according to DR diagnose
#id_code in the csv file = filename for corresponding image
def sortRawdata():
    for i in range(label_data.shape[0]):
        diagnose = label_data.iloc[i,-1]
        if diagnose == 0:
            diagnoseName = 'noDR'
        elif diagnose == 1:
            diagnoseName = 'Mild'
        elif diagnose == 2:
            diagnoseName='Moderate'
        elif diagnose == 3:
            diagnoseName='Severe'
        elif diagnose==4:
            diagnoseName='Proliferative'
        else:
            print('No suitable diagnose!')
        filename = label_data.iloc[i,0]
        filename = str(filename)+'.png'
        print('Name of file:',filename)
        print('Corresponding diagnose:',diagnoseName)
        #Get the file
        sourcepath = os.path.join(imageFolderName,filename)
        targetpath = os.path.join(diagnoseName,filename)
        #Copy the images over to respective class folders
        shutil.copy(sourcepath,targetpath)

classFolders = ['noDR','Mild','Moderate','Severe','Proliferative']
#classFolders = ['noDR']
def splitTrainVal():
    for _class in classFolders:
        folderPath = os.path.join('./',_class)
        #list all files in folder:
        classFiles = os.listdir(folderPath)
        #For now: only select 100 from each class!
        classFiles = classFiles[:100]
        #Select 80% for training and 20% for validation
        num_train = int(len(classFiles)*0.8)
        print('Looking at DR class:',_class)
        print('Number of images in class:',len(classFiles))
        print('Number of images for training:',num_train)
        #Pick 80% randomly from the list
        #Convert to set for easier substraction
        trainFiles = set(random.sample(classFiles,num_train))
        validFiles = set(classFiles) - trainFiles
        #Convert back to list again:
        trainFiles = list(trainFiles)
        validFiles = list(validFiles)
        print(len(trainFiles))
        print(len(validFiles))
        #Check that no duplicates in the train and validation files:
        for tFile in trainFiles:
            if tFile in validFiles:
                print('Same image in valid and training list! Something went wrong here!')
        #Move training files to Train
        for tFile in trainFiles:
            sourceFile = os.path.join(folderPath,tFile)
            targetPath = os.path.join('Train',_class,tFile)
            shutil.copy(sourceFile,targetPath)

        #Move validation files to Valid
        for vFile in validFiles:
            sourceFile = os.path.join(folderPath,vFile)
            targetPath = os.path.join('Valid',_class,vFile)
            shutil.copy(sourceFile,targetPath)



def splitTrainValTest():
    for _class in classFolders:
        #Convert class to number for pytorch classifier to understand:
        if _class == 'noDR':
            classInt = '0'
        elif _class == 'Mild':
            classInt = '1'
        elif _class == 'Moderate':
            classInt = '2'
        elif _class == 'Severe':
            classInt = '3'
        elif _class == 'Proliferative':
            classInt = '4'
        else:
            print('Something is wrong with the class name!!!')
        #Get the correct folder, matching with the class:
        folderPath = os.path.join('./Data/APTOS2019',_class)
        #list all files in folder:
        classFiles = os.listdir(folderPath)
        #Select 80% for training, 10% for validation, 10% for testing
        num_train = int(len(classFiles)*0.8)
        num_valid = int(len(classFiles)*0.1)
        print('Looking at DR class:',_class)
        print('Corresponding class number:', classInt)
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
            targetPath = os.path.join('Data/TrainAPTOS',classInt,tFile)
            shutil.copy(sourceFile,targetPath)
        #Check that no duplicates in validation and test files:
        for vFile in validFiles:
            if vFile in testFiles:
                print('Same image in valid and testing list! Something went wrong!')
        #Move validation files to Valid
        for vFile in validFiles:
            sourceFile = os.path.join(folderPath,vFile)
            targetPath = os.path.join('Data/ValidAPTOS',classInt,vFile)
            shutil.copy(sourceFile,targetPath)
        #Move test files to Test
        for testFile in testFiles:
            sourceFile = os.path.join(folderPath,testFile)
            targetPath = os.path.join('Data/TestAPTOS',classInt,testFile)
            shutil.copy(sourceFile,targetPath)


#sortRawdata()
#splitTrainVal()

#Split into 80% train, 10% valid, 10% testing:
#splitTrainValTest()


#Check that correct number of images for each class:
#for _class in classFolders:
    #classFiles = os.listdir(os.path.join('Data/APTOS2019',_class))
    #print('Name of class:',_class)
    #print('Number of images:',len(classFiles))
