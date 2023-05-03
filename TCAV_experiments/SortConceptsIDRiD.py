import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd

#Go through the tif masks and ensure that none of them are completely black
#Can then sort the corresponding images depending on which concepts that are present

#For no abnormalitites, can pick the images from the DR grading image folder

#NB! Must always repeat for both train and test folders!!
idrid_segmentationPath = '../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/2. All Segmentation Groundtruths'
seg_train = os.path.join(idrid_segmentationPath,'a. Training Set')
seg_test = os.path.join(idrid_segmentationPath,'b. Testing Set')

def check_emptyMasks():
    abnormalities_list = ['1. Microaneurysms','2. Haemorrhages','3. Hard Exudates','4. Soft Exudates']
    for _anorm in abnormalities_list:
        print('Looking at abnormality:',_anorm)
        print('Train set:')
        print('Number of masks:',len(os.listdir(os.path.join(seg_train,_anorm))))
        mask_list = os.listdir(os.path.join(seg_train,_anorm))
        for _mask in mask_list:
            mask_path = os.path.join(seg_train,_anorm,_mask)
            my_mask = cv.imread(mask_path)
            #If segmentation mask is completely black, the abnormality is not present
            if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                print('Training mask')
                print('No abnormality in segmentation mask')
        print('Test set:')
        print('Number of masks:',len(os.listdir(os.path.join(seg_test,_anorm))))
        mask_listTest = os.listdir(os.path.join(seg_test,_anorm))
        for _mask in mask_listTest:
            mask_path = os.path.join(seg_test,_anorm,_mask)
            my_mask = cv.imread(mask_path)
            #If segmentation mask is completely black, the abnormality is not present
            if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                print('Testing mask')
                print('No abnormality in segmentation mask')

#Because all masks in the folders actually contain segmentations,
#we can just filter for the findings based on whether the mask is present
#in the corresponding segmentation folder or not
#check_emptyMasks()

#Get a list of all images:
train_images = os.listdir('../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/1. Original Images/a. Training Set')
test_images = os.listdir('../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/1. Original Images/b. Testing Set')
img_train = '../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/1. Original Images/a. Training Set'
img_test = '../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/1. Original Images/b. Testing Set'
conceptFolder = 'ConceptFoldersIDRiD/SortedByCombinations'
abnormalities_list = ['1. Microaneurysms','2. Haemorrhages','3. Hard Exudates','4. Soft Exudates']
#Create list of files for each abnormality for training images
MA_trainFiles = os.listdir(os.path.join(seg_train,'1. Microaneurysms'))
HE_trainFiles = os.listdir(os.path.join(seg_train,'2. Haemorrhages'))
HardEx_trainFiles = os.listdir(os.path.join(seg_train,'3. Hard Exudates'))
SoftEx_trainFiles = os.listdir(os.path.join(seg_train,'4. Soft Exudates'))
#Then I repeat for the test images:
MA_testFiles = os.listdir(os.path.join(seg_test,'1. Microaneurysms'))
HE_testFiles = os.listdir(os.path.join(seg_test,'2. Haemorrhages'))
HardEx_testFiles = os.listdir(os.path.join(seg_test,'3. Hard Exudates'))
SoftEx_testFiles = os.listdir(os.path.join(seg_test,'4. Soft Exudates'))



def sortByCombinations():
    MA_HardEx_counter = 0
    MA_HE_HardEx_counter = 0
    MA_HE_SoftEx_HardEx_counter = 0
    print('Number of training images:',len(train_images))
    #Start with looping through the train images and check if they are in the abnorm files:
    for _img in train_images:
        source_path = os.path.join(img_train,_img)
        _imgMA = _img[:-4] + '_MA.tif'
        _imgHE = _img[:-4] + '_HE.tif'
        _imgHardEx = _img[:-4] + '_EX.tif'
        _imgSoftEx = _img[:-4] + '_SE.tif'
        #print(_imgMA)
        #If only MA:
        if (_imgMA in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA',_img)
            #shutil.copy(source_path,target_path)
            print('Only MA in the image')
        #If only hemorrhages:
        elif (_img not in MA_trainFiles) and (_img in HE_trainFiles) and (_img not in HardEx_trainFiles) and (_img not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'HE',_img)
            #shutil.copy(source_path,target_path)
            print('Only hemo in the image')
        #If only HardEx:
        elif (_imgMA not in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('Only HardEx in the image')
        #Only SoftEx
        elif (_imgMA not in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'SoftEx',_img)
            #shutil.copy(source_path,target_path)
        #If MA + HE
        elif (_imgMA in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_HE',_img)
            #shutil.copy(source_path,target_path)
            print('MA + HE in the image')
        #If MA + SoftEx
        elif (_imgMA in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + SoftEx in the image')
        #If MA + HardEx:
        elif (_imgMA in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HardEx_counter += 1
        #If HE + SoftEx
        elif (_imgMA not in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + SoftEx in the image')
        #If HE + HardEx:
        elif (_imgMA not in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + HardEx in the image')
        #If SoftEx + HardEx:
        elif (_imgMA not in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('SoftEx + HardEx in the image')
        #If MA + HE + SoftEx
        elif (_imgMA in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx not in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + HE + SoftEx in the image')
        #If MA + HE + HardEx:
        elif (_imgMA in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx not in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_HardEx_counter +=1
        #If MA + SoftEx + HardEx:
        elif (_imgMA in MA_trainFiles) and (_imgHE not in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + SoftEx + HardEx in the image')
        #If HE + SoftEx + HardEx
        elif (_imgMA not in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + SoftEx + HardEx in the image')
        #If all 4 are present:
        elif (_imgMA in MA_trainFiles) and (_imgHE in HE_trainFiles) and (_imgHardEx in HardEx_trainFiles) and (_imgSoftEx in SoftEx_trainFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_counter +=1
    #Repeat for the test files:
    print('Number of testing images:',len(test_images))
    for _img in test_images:
        source_path = os.path.join(img_test,_img)
        _imgMA = _img[:-4] + '_MA.tif'
        _imgHE = _img[:-4] + '_HE.tif'
        _imgHardEx = _img[:-4] + '_EX.tif'
        _imgSoftEx = _img[:-4] + '_SE.tif'
        #If only MA:
        if (_imgMA in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA',_img)
            #shutil.copy(source_path,target_path)
            print('Only MA in the image')
        #If only hemorrhages:
        elif (_img not in MA_testFiles) and (_img in HE_testFiles) and (_img not in HardEx_testFiles) and (_img not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'HE',_img)
            #shutil.copy(source_path,target_path)
            print('Only hemo in the image')
        #If only HardEx:
        elif (_imgMA not in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('Only HardEx in the image')
        #Only SoftEx
        elif (_imgMA not in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('Only SoftEx in the image')
        #If MA + HE
        elif (_imgMA in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_HE',_img)
            #shutil.copy(source_path,target_path)
            print('MA + HE in the image')
        #If MA + SoftEx
        elif (_imgMA in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + SoftEx in the image')
        #If MA + HardEx:
        elif (_imgMA in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HardEx_counter += 1
        #If HE + SoftEx
        elif (_imgMA not in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + SoftEx in the image')
        #If HE + HardEx:
        elif (_imgMA not in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + HardEx in the image')
        #If SoftEx + HardEx:
        elif (_imgMA not in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('SoftEx + HardEx in the image')
        #If MA + HE + SoftEx
        elif (_imgMA in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx not in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + HE + SoftEx in the image')
        #If MA + HE + HardEx:
        elif (_imgMA in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx not in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_HardEx_counter += 1
        #If MA + SoftEx + HardEx:
        elif (_imgMA in MA_testFiles) and (_imgHE not in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('MA + SoftEx + HardEx in the image')
        #If HE + SoftEx + HardEx
        elif (_imgMA not in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            print('HE + SoftEx + HardEx in the image')
        #If all 4 are present:
        elif (_imgMA in MA_testFiles) and (_imgHE in HE_testFiles) and (_imgHardEx in HardEx_testFiles) and (_imgSoftEx in SoftEx_testFiles):
            target_path = os.path.join(conceptFolder,'MA_HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_counter+=1

    print('Number of images with MA + HardEx:',MA_HardEx_counter)
    print('Number of images with MA + HE + HardEx:',MA_HE_HardEx_counter)
    print('Number of images with MA + HE + SoftEx + HardEx:',MA_HE_SoftEx_HardEx_counter)
        

#sortByCombinations()
#print('Number of files in MA + HardEx sorted folder:',len(os.listdir(os.path.join(conceptFolder,'MA_HardEx'))))
#print('Number of files in MA + HE + HardEx sorted folder:',len(os.listdir(os.path.join(conceptFolder,'MA_HE_HardEx'))))
#print('Number of files in MA + HE + SoftEx + HardEx sorted folder:',len(os.listdir(os.path.join(conceptFolder,'MA_HE_SoftEx_HardEx'))))       
'''
#Also move all healthy images from train and test folders to the NoAbnormalities concept folder
gradingDf_train = pd.read_csv('../Data/IDRiD_dataFebruary/B.DiseaseGrading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv')
gradingDf_test = pd.read_csv('../Data/IDRiD_dataFebruary/B.DiseaseGrading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv')
#Remove empty/not relevant columns
gradingDf_train = gradingDf_train.iloc[:,:2]
gradingDf_test = gradingDf_test.iloc[:,:2]
print('First part of GT training df:')
print(gradingDf_train.head())
print('The columns:',gradingDf_train.columns)
print('Shape Training:',gradingDf_train.shape)
print('Shape Testing:',gradingDf_test.shape)

origImages_train = '../Data/IDRiD_dataFebruary/B.DiseaseGrading/B. Disease Grading/1. Original Images/a. Training Set'
origImages_test = '../Data/IDRiD_dataFebruary/B.DiseaseGrading/B. Disease Grading/1. Original Images/b. Testing Set'
'''
noDR_conceptFolder = os.path.join(conceptFolder,'NoAbnormalities')
def moveNoDRImages():
    noDR_counter = 0
    #Start with the training dataset:
    print('Looking at training data:')
    for i in range(gradingDf_train.shape[0]):
        if gradingDf_train.iloc[i,1]==0:
            _img = gradingDf_train.iloc[i,0] + '.jpg'
            source_path = os.path.join(origImages_train,_img)
            target_path = os.path.join(noDR_conceptFolder,_img)
            shutil.copy(source_path,target_path)
            noDR_counter += 1
    #Repeat with the testing dataset:
    print('Looking at testing data:')
    for i in range(gradingDf_test.shape[0]):
        if gradingDf_test.iloc[i,1]==0:
            _img = gradingDf_test.iloc[i,0] + '.jpg'
            source_path = os.path.join(origImages_test,_img)
            target_path = os.path.join(noDR_conceptFolder,_img)
            shutil.copy(source_path,target_path)
            noDR_counter += 1
    #This number is in line with what was reported about IDRiD in the DDR related datasets section:
    print('Number of noDR images:',noDR_counter)
    print('Number of images in concept folder:',len(os.listdir(noDR_conceptFolder)))

#moveNoDRImages()
print('NUmber of healthy images:',len(os.listdir(noDR_conceptFolder)))