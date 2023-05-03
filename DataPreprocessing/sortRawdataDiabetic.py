import pandas as pd
import numpy as np
import shutil
import os
import random

from sklearn.model_selection import GroupShuffleSplit

imgFolderpath = 'Data/DiabeticRetinopathyDetection/train'
#Read in the label-file for the training data
labelDf = pd.read_csv('Data/DiabeticRetinopathyDetection/trainLabels.csv')
print(labelDf.head())
print('Number of training instances with labels:',labelDf.shape[0])
print('Distribution of the different levels of DR:', labelDf['level'].value_counts())

trainFiles = os.listdir('Data/DiabeticRetinopathyDetection/train')
print('Number of images in training folder:',len(trainFiles))
dataFolder = 'Data/DiabeticRetinopathyDetection/train'


'''
counter = 0
for _img in trainFiles:
    #Remove the .jpeg-part (since this is not in the labelDf)
    imageName = _img[:-5]
    #print(imageName)
    #Check that the image file is actually in the labelDf:
    if imageName in labelDf.iloc[:,0].tolist():
        counter +=1
print('Number of matching images:',str(counter))
'''
#Want to extract the images and place them in folders for corresponding 
#level of DR:
def sortData():
    #Go through each row in the DF and get the DR grade:
    for i in range(labelDf.shape[0]):
        imageId = labelDf.iloc[i,0]
        classDR = labelDf.iloc[i,-1]
        if classDR == 0:
            classFolder = 'noDR'
        elif classDR == 1:
            classFolder = 'Mild'
        elif classDR == 2:
            classFolder = 'Moderate'
        elif classDR == 3: 
            classFolder = 'Severe'
        elif classDR == 4:
            classFolder='Proliferative'
        else:
            print('This grading of DR is out of bounds!')
        #Find corresponding image
        imageFilename = imageId + '.jpeg'
        imageFilepath = os.path.join(imgFolderpath,imageFilename)
        targetPath = os.path.join('Data/DiabeticRetinopathyDetection',classFolder,imageFilename)
        #Copy the file from train folder to the correct class folder
        shutil.copy(imageFilepath,targetPath)

classList = ['noDR','Mild','Moderate','Severe','Proliferative']
#Inspect the class distribution:
#for _class in classList:
    #classFiles = os.listdir(os.path.join('Data/DiabeticRetinopathyDetection',_class))
    #print('Working with class:',_class)
    #print('Number of images for this class:',len(classFiles))

#Split into train, validation and test:
#NB need to ensure that same ID in train/validation/test!
#Because right/left eye from same patient

#Add a new column with just the patient id, regardless of eye:
labelDf['UniqueId'] = 'Lol'
for i in range(labelDf.shape[0]):
    imageName = labelDf.iloc[i,0]
    idName = imageName.strip('_left')
    idName = idName.strip('_right')
    labelDf.iloc[i,-1]=int(idName)
print(labelDf['UniqueId'].describe())

'''
#Check if same ID always  have same grading of both eyes:
# (It does not!)
differentLevelsCounter = 0
for _id in uniqueIds:
    #Get the two imagenames:
    leftImage = _id + '_left'
    rightImage = _id + '_right'
    #print('Name of image left eye:')
    #print(labelDf[labelDf['image']==leftImage]['image'])
    #print(labelDf[labelDf['image']==leftImage]['level'])
    #print('Name of image right eye:')
    #print(labelDf[labelDf['image']==rightImage]['image'])
    #print(labelDf[labelDf['image']==rightImage]['level'])
    leftLevel = labelDf[labelDf['image']==leftImage]['level'].values
    rightLevel = labelDf[labelDf['image']==rightImage]['level'].values
    if leftLevel!=rightLevel:
        differentLevelsCounter += 1
        print('Patient has different DR levels in each eye')
        print(labelDf[labelDf['image']==leftImage]['level'])
        print(labelDf[labelDf['image']==rightImage]['level'])

print('Total number of patients with different levels in each eye:')
print(str(differentLevelsCounter))
'''
def trainValidTestSplit():
#Find correct number for train, valid and test:
    num_train = int(labelDf.shape[0]*0.8)
    num_valid = int(labelDf.shape[0]*0.1)
    #The rest is used for testing
    num_test = labelDf.shape[0] - (num_train + num_valid)

    print('Number of training samples:',str(num_train))
    print('Number of valid samples:',str(num_valid))
    print('Number of test samples:',str(num_test))
    print('In total:', str(num_train+num_valid+num_test))
    print('Should match with number of samples in dataset:',labelDf.shape[0])

    
    #Use GroupShuffleSplit to split the data
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
    #Define the groups:
    my_groups = labelDf['UniqueId']
    #Divide test number by 2 since 2 observations/eyes per ID:
    groupss =GroupShuffleSplit(n_splits=1, random_state=42, test_size=int(num_test/2), train_size=None)


    for i, (trainValid_index, test_index) in enumerate(groupss.split(labelDf['UniqueId'], groups=my_groups)):
        trainValidDf = labelDf.iloc[trainValid_index,:]
        testDf = labelDf.iloc[test_index,:]

    smaller_groups = trainValidDf['UniqueId']
    #Divide valid number by 2 since 2 observations/eyes per ID:
    smallerGroupss =GroupShuffleSplit(n_splits=1, random_state=42, test_size=int(num_valid/2), train_size=None)

    for i, (train_index, valid_index) in enumerate(smallerGroupss.split(trainValidDf['UniqueId'], groups=smaller_groups)):
        trainDf = trainValidDf.iloc[train_index,:]
        validDf = trainValidDf.iloc[valid_index,:]

    print(trainDf.shape)
    print(validDf.shape)
    print(testDf.shape)

    #Check that same id is not in train, test or validation:
    for i in range(trainDf.shape[0]):
        if trainDf.iloc[i,-1] in validDf['UniqueId'].tolist():
            print('Same patient in train and validation set!!! Something went wrong in the splitting process!')
        elif trainDf.iloc[i,-1] in testDf['UniqueId'].tolist():
            print('Same patient in train and testing set!!!')

    for j in range(validDf.shape[0]):
        if validDf.iloc[j,-1] in testDf['UniqueId'].tolist():
            print('Same patient in validation and test set! Something went wrong!')


    #Get the shortened list of unique IDs
    uniqueTrainIds = set(trainDf['UniqueId'].tolist())
    uniqueTrainIds = list(uniqueTrainIds)
    #Repeat for validation and test:
    uniqueValidIds = set(validDf['UniqueId'].tolist())
    uniqueValidIds = list(uniqueValidIds)
    uniqueTestIds = set(testDf['UniqueId'].tolist())
    uniqueTestIds = list(uniqueTestIds)

    #Move train data to correct training folder according to DR level:
    for _id in uniqueTrainIds:
        targetFolder = 'Data/TrainDRDetection'
        #Find the left eye data:
        patient_idLeft = str(_id)+'_left'
        #Create corresponding filename:
        image_nameLeft = patient_idLeft+'.jpeg'
        classLeft = trainDf[trainDf['image']==patient_idLeft]['level'].values[0]
        #Get the source path to find the corresponding image:
        sourcePathLeft = os.path.join(dataFolder,image_nameLeft)
        #Target directory depends on which DR level the patient has:
        if classLeft == 0:
            targetPathLeft = os.path.join(targetFolder,'0')
        elif classLeft == 1:
            targetPathLeft = os.path.join(targetFolder,'1')
        elif classLeft == 2:
            targetPathLeft = os.path.join(targetFolder,'2')
        elif classLeft == 3:
            targetPathLeft = os.path.join(targetFolder,'3')
        elif classLeft == 4:
            targetPathLeft = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathLeft, targetPathLeft)
    
        #Repeat for the right eye for same subject:
        patient_idRight = str(_id)+'_right'
        image_nameRight = patient_idRight+'.jpeg'
        classRight = trainDf[trainDf['image']==patient_idRight]['level'].values[0]
        #print('Class for right eye:',classRight)
        #print('The patient ID:',patient_idRight)
        #print('The entire row:')
        #print(trainDf[trainDf['image']==patient_idRight])
        #Get the source path to find the corresponding image:
        sourcePathRight = os.path.join(dataFolder,image_nameRight)
        #Target directory depends on which DR level the patient has:
        if classRight == 0:
            targetPathRight = os.path.join(targetFolder,'0')
        elif classRight == 1:
            targetPathRight = os.path.join(targetFolder,'1')
        elif classRight == 2:
            targetPathRight = os.path.join(targetFolder,'2')
        elif classRight == 3:
            targetPathRight = os.path.join(targetFolder,'3')
        elif classRight == 4:
            targetPathRight = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathRight, targetPathRight)
    
    #Repeat for validation and test:
    for _id in uniqueValidIds:
        targetFolder = 'Data/ValidDRDetection'
        #Find the left eye data:
        patient_idLeft = str(_id)+'_left'
        #Create corresponding filename:
        image_nameLeft = patient_idLeft+'.jpeg'
        classLeft = validDf[validDf['image']==patient_idLeft]['level'].values[0]
        #Get the source path to find the corresponding image:
        sourcePathLeft = os.path.join(dataFolder,image_nameLeft)
        #Target directory depends on which DR level the patient has:
        if classLeft == 0:
            targetPathLeft = os.path.join(targetFolder,'0')
        elif classLeft == 1:
            targetPathLeft = os.path.join(targetFolder,'1')
        elif classLeft == 2:
            targetPathLeft = os.path.join(targetFolder,'2')
        elif classLeft == 3:
            targetPathLeft = os.path.join(targetFolder,'3')
        elif classLeft == 4:
            targetPathLeft = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathLeft, targetPathLeft)
    
        #Repeat for the right eye for same subject:
        patient_idRight = str(_id)+'_right'
        image_nameRight = patient_idRight+'.jpeg'
        classRight = validDf[validDf['image']==patient_idRight]['level'].values[0]
        #Get the source path to find the corresponding image:
        sourcePathRight = os.path.join(dataFolder,image_nameRight)
        #Target directory depends on which DR level the patient has:
        if classRight == 0:
            targetPathRight = os.path.join(targetFolder,'0')
        elif classRight == 1:
            targetPathRight = os.path.join(targetFolder,'1')
        elif classRight == 2:
            targetPathRight = os.path.join(targetFolder,'2')
        elif classRight == 3:
            targetPathRight = os.path.join(targetFolder,'3')
        elif classRight == 4:
            targetPathRight = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathRight, targetPathRight)

    #...And for the test set:    
    for _id in uniqueTestIds:
        targetFolder = 'Data/TestDRDetection'
        #Find the left eye data:
        patient_idLeft = str(_id)+'_left'
        #Create corresponding filename:
        image_nameLeft = patient_idLeft+'.jpeg'
        classLeft = testDf[testDf['image']==patient_idLeft]['level'].values[0]
        #Get the source path to find the corresponding image:
        sourcePathLeft = os.path.join(dataFolder,image_nameLeft)
        #Target directory depends on which DR level the patient has:
        if classLeft == 0:
            targetPathLeft = os.path.join(targetFolder,'0')
        elif classLeft == 1:
            targetPathLeft = os.path.join(targetFolder,'1')
        elif classLeft == 2:
            targetPathLeft = os.path.join(targetFolder,'2')
        elif classLeft == 3:
            targetPathLeft = os.path.join(targetFolder,'3')
        elif classLeft == 4:
            targetPathLeft = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathLeft, targetPathLeft)
    
        #Repeat for the right eye for same subject:
        patient_idRight = str(_id)+'_right'
        image_nameRight = patient_idRight+'.jpeg'
        classRight = testDf[testDf['image']==patient_idRight]['level'].values[0]
        #Get the source path to find the corresponding image:
        sourcePathRight = os.path.join(dataFolder,image_nameRight)
        #Target directory depends on which DR level the patient has:
        if classRight == 0:
            targetPathRight = os.path.join(targetFolder,'0')
        elif classRight == 1:
            targetPathRight = os.path.join(targetFolder,'1')
        elif classRight == 2:
            targetPathRight = os.path.join(targetFolder,'2')
        elif classRight == 3:
            targetPathRight = os.path.join(targetFolder,'3')
        elif classRight == 4:
            targetPathRight = os.path.join(targetFolder,'4')
        else:
            print('Class level out of bounds!')
        #Copy the image to correct target folder:
        shutil.copy(sourcePathRight, targetPathRight)


trainValidTestSplit()
#Check how many images in each folder:
#Start with training folder:
print('Class distribution in training set:')
for i in range(5):
    classFolder = os.path.join('Data/TrainDRDetection',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))

#Repeat for validation and test folders:
print('Class distribution in validation set:')
for i in range(5):
    classFolder = os.path.join('Data/ValidDRDetection',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))

#Start with training folder:
print('Class distribution in testing set:')
for i in range(5):
    classFolder = os.path.join('Data/TestDRDetection',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))