import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import random

random.seed(42)
#Move all the sorted concepts from FGADR, DDR, IDRiD and DiaretDB1 to a common folder
#The common folder contains all sorted folders from FGADR (copied from this since it has most folders)
common_folder = 'ConceptFoldersAll/SortedByCombinations'
DDR_folder = 'ConceptFoldersDDR/SortedByCombinations'
IDRiD_folder = 'ConceptFoldersIDRiD/SortedByCombinations'
DiaretDB1_folder = 'ConceptFoldersDiaretDB/SortedByCombinationsDB1'
#List all folders in the directory
common_combos = os.listdir(common_folder)
'''
print('All folders from FGADR:')
print(common_combos)
#Then list the corresponding folders for DDR, IDRiD and DiaretDB1:
print('Folders from DDR:')
print(os.listdir(DDR_folder))
print('Folders from IDRiD:')
print(os.listdir(IDRiD_folder))
print('Folders from DiaretDB1:')
print(os.listdir(DiaretDB1_folder))
'''
#Check that all folders for DDR, IDRiD and DiaretDB1 are in the common folders:
for _folder in os.listdir(DDR_folder):
    if _folder not in common_combos:
        print('DDR folder not in common folders:',_folder)
for _folder in os.listdir(IDRiD_folder):
    if _folder not in common_combos:
        print('IDRiD folder not in common folders:',_folder)
for _folder in os.listdir(DiaretDB1_folder):
    if _folder not in common_combos:
        print('DiaretDB1 folder not in common folders:',_folder)
#Move sorted combinations folders for each dataset to a large combined folder for all datasets together
def moveFolderImages():
    #Look at all folders for DDR:
    for _folder in os.listdir(DDR_folder):
        folder_path = os.path.join(DDR_folder,_folder)
        #print('Folder:',folder_path)
        image_list = os.listdir(folder_path)
        #For each image in the folder:
        for _image in image_list:
            source_path = os.path.join(folder_path, _image)
            target_path = os.path.join(common_folder, _folder, _image)
            #Copy to the common folder:
            #shutil.copy(source_path, target_path)
    #Look at all folders for IDRiD:
    for _folder in os.listdir(IDRiD_folder):
        folder_path = os.path.join(IDRiD_folder,_folder)
        #print('Folder:',folder_path)
        image_list = os.listdir(folder_path)
        #For each image in the folder:
        for _image in image_list:
            source_path = os.path.join(folder_path, _image)
            target_path = os.path.join(common_folder, _folder, _image)
            #Copy to the common folder:
            #shutil.copy(source_path, target_path)
    #And for the DiaretDB1:
    #Look at all folders for DDR:
    for _folder in os.listdir(DiaretDB1_folder):
        folder_path = os.path.join(DiaretDB1_folder,_folder)
        #print('Folder:',folder_path)
        image_list = os.listdir(folder_path)
        #For each image in the folder:
        for _image in image_list:
            source_path = os.path.join(folder_path, _image)
            target_path = os.path.join(common_folder, _folder, _image)
            #Copy to the common folder:
            #shutil.copy(source_path, target_path)

#moveFolderImages()

#Check that the number of images in original folders and new folder sums up (in case images with same name etc)
#for _folder in common_combos:
    #if 'irma' not in _folder:
        #print('IRMA not in folder:',_folder)
        #print('Number of images:', len(os.listdir(os.path.join(common_folder, _folder))))
    #if 'irma' in str(_folder)[-4:]:
    #    print('MA in folder:',_folder)
    #print('Looking at folder:',_folder)
    #print('Number of images in "all" folder:',len(os.listdir(os.path.join(common_folder,_folder))))
    #print('Number of images in FGADR:',len(os.listdir(os.path.join('ConceptFoldersFGADR/SortedByCombinations',_folder))))
    #if _folder in os.listdir(DDR_folder):
    #    print('Number of images in DDR:',len(os.listdir(os.path.join(DDR_folder,_folder))))
    #if _folder in os.listdir(IDRiD_folder):
    #    print('Number of images in IDRiD:',len(os.listdir(os.path.join(IDRiD_folder,_folder))))
    #if _folder in os.listdir(DiaretDB1_folder):
    #    print('Number of images in DiaretDB1:',len(os.listdir(os.path.join(DiaretDB1_folder,_folder))))
    #print('**************')

#Move NV concept images (positive and negative)
#Positive NV folders select 45 in total: 
#HE_NV: 1
#HE_SoftEx_HardEx_NV: 1
#MA_HE_HardEx_NV: 7
#MA_HE_SoftEx_HardEx_NV: 3
#MA_HE_SoftEx_NV: 1
#NV: 2
#MA_HE_HardEx_NV_irma: 11
#MA_HE_SoftEx_HardEx_NV_irma: 15 (pick 14)

def moveNV_examples(num_negative = 5):
    #NB! The order of these folders should be the same (corresponding positives and negatives have same index)
    NV_positive_folders = ['HE_NV','HE_SoftEx_HardEx_NV','MA_HE_HardEx_NV','MA_HE_SoftEx_HardEx_NV',
        'MA_HE_SoftEx_NV','NV','MA_HE_HardEx_NV_irma','MA_HE_SoftEx_HardEx_NV_irma']
    NV_negative_folders = ['HE','HE_SoftEx_HardEx','MA_HE_HardEx','MA_HE_SoftEx_HardEx',
        'MA_HE_SoftEx','NoAbnormalities','MA_HE_HardEx_irma','MA_HE_SoftEx_HardEx_irma']
    #Start with creating one positive examples folder
    for pos_folder in NV_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/NV/PositiveExamples'
        if pos_folder != 'MA_HE_SoftEx_HardEx_NV_irma':
            for _image in os.listdir(folder_path):
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
        else:
            image_list = os.listdir(folder_path)
            #Select 14 images randomly
            selected_images = set(random.sample(image_list,14))
            selected_images = list(selected_images)
            print('14 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            print(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
    #Continue creating 5 (num_negative) different negative example folders
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(NV_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',NV_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/NV/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = NV_positive_folders[neg_folder]
            if corresp_pos_folder != 'MA_HE_SoftEx_HardEx_NV_irma':
                num_samples = len(os.listdir(os.path.join('ConceptFoldersAll/SortedByCombinations',NV_positive_folders[neg_folder])))
            else:
                num_samples = 14
            print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select 14 images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)

def moveIRMA_examples(num_negative = 5):
    #NB! The order of these folders should be the same (corresponding positives and negatives have same index)
    IRMA_positive_folders = ['HardEx_irma','HE_irma','MA_irma','MA_HE_SoftEx_irma','MA_HE_SoftEx_HardEx_irma',
    'MA_HE_HardEx_irma','MA_HE_HardEx_NV_irma','MA_HE_SoftEx_HardEx_NV_irma'] #31 in total
    IRMA_negative_folders = ['HardEx','HE','MA','MA_HE_SoftEx','MA_HE_SoftEx_HardEx','MA_HE_HardEx',
    'MA_HE_HardEx_NV','MA_HE_SoftEx_HardEx_NV']
    #Start with creating one positive examples folder
    for pos_folder in IRMA_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/irma/PositiveExamples'
        
        #Select 15 random images from this folder:
        if pos_folder == 'MA_HE_HardEx_irma':
            image_list = os.listdir(folder_path)
            #Select 15 images randomly
            selected_images = set(random.sample(image_list,15))
            selected_images = list(selected_images)
            print('15 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
        #Pick 3 images from this folder:
        elif pos_folder == 'MA_HE_HardEx_NV_irma':
            image_list = os.listdir(folder_path)
            #Select 3 images randomly
            selected_images = set(random.sample(image_list,3))
            selected_images = list(selected_images)
            print('3 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
        #Pick 9 images from this folder:
        elif pos_folder == 'MA_HE_SoftEx_HardEx_irma':
            image_list = os.listdir(folder_path)
            print('Checking:',len(os.listdir('ConceptFoldersAll/SortedByCombinations/MA_HE_SoftEx_HardEx_irma')))
            print('Number of images in folder:',len(image_list))
            print(folder_path)
            #Select 9 images randomly
            selected_images = set(random.sample(image_list,9))
            selected_images = list(selected_images)
            print('9 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
        #Pick 1 image from this folder:
        elif pos_folder == 'MA_HE_SoftEx_HardEx_NV_irma':
            image_list = os.listdir(folder_path)
            #Select 1 image randomly
            selected_images = set(random.sample(image_list,1))
            selected_images = list(selected_images)
            print('1 random selected image from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
        #For the other folders, all images are used:
        else:
            print('Not any of the above folders:',pos_folder)
            for _image in os.listdir(folder_path):
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                #shutil.copy(source_path,target_path)
    #Repeat for the negative folders:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(IRMA_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',IRMA_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/irma/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = IRMA_positive_folders[neg_folder]
            if corresp_pos_folder == 'MA_HE_HardEx_irma':
                num_samples = 15
            elif corresp_pos_folder == 'MA_HE_HardEx_NV_irma':
                num_samples = 3
            elif corresp_pos_folder == 'MA_HE_SoftEx_HardEx_irma':
                num_samples = 9
            elif corresp_pos_folder == 'MA_HE_SoftEx_HardEx_NV_irma':
                num_samples = 1    
            else:
                num_samples = len(os.listdir(os.path.join('ConceptFoldersAll/SortedByCombinations',corresp_pos_folder)))
            print('Negative folder:',IRMA_negative_folders[neg_folder])
            print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select 14 images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)

#Add 5 more images:
#HE_HardEx: 3
#MA_SoftEx_HardEx: 2                
def moveHardEx_examples(num_negative = 5):
    #NB! The order with positive and negative corresp. folders must be the same!
    HardEx_positive_folders = ['HardEx','HardEx_NV','MA_HE_HardEx','MA_HE_SoftEx_HardEx',
    'MA_HardEx','MA_HE_SoftEx_HardEx_irma',
    'HE_HardEx','MA_SoftEx_HardEx']
    HardEx_negative_folders = ['NoAbnormalities','NV','MA_HE','MA_HE_SoftEx','MA','MA_HE_SoftEx_irma',
    'HE','MA_SoftEx']
    #Start with creating one positive examples folder
    for pos_folder in HardEx_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/New_HardEx_test20/PositiveExamples'
        #Select 10 random images from this folder:
        if pos_folder == 'HardEx':
            image_list = os.listdir(folder_path)
            #Select 10 images randomly
            selected_images = set(random.sample(image_list,10))
            selected_images = list(selected_images)
            print('10 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        #Pick 10 images from this folder:
        elif pos_folder == 'MA_HE_HardEx':
            image_list = os.listdir(folder_path)
            #Select 10 images randomly
            selected_images = set(random.sample(image_list,10))
            selected_images = list(selected_images)
            print('10 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        #Pick 10 images from this folder:
        elif pos_folder == 'MA_HE_SoftEx_HardEx':
            image_list = os.listdir(folder_path)
            #Select 10 images randomly
            selected_images = set(random.sample(image_list,10))
            selected_images = list(selected_images)
            print('10 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        elif pos_folder == 'MA_HardEx':
            image_list = os.listdir(folder_path)
            #Select 6 images randomly
            selected_images = set(random.sample(image_list,6))
            selected_images = list(selected_images)
            print('6 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        #Pick 3 image from this folder:
        elif (pos_folder == 'MA_HE_SoftEx_HardEx_irma') or (pos_folder == 'HE_HardEx'):
            image_list = os.listdir(folder_path)
            #Select 3 images randomly
            selected_images = set(random.sample(image_list,3))
            selected_images = list(selected_images)
            print('3 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        elif pos_folder == 'MA_SoftEx_HardEx':
            image_list = os.listdir(folder_path)
            #Select 2 images randomly
            selected_images = set(random.sample(image_list,2))
            selected_images = list(selected_images)
            print('2 random selected images from:',pos_folder)
            print('Number of selected images:',len(selected_images))
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
        #For the other folders, all images are used:
        else:
            print('Not any of the above folders:',pos_folder)
            for _image in os.listdir(folder_path):
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
    #Repeat for the negative folders:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(HardEx_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',HardEx_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/New_HardEx_test20/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = HardEx_positive_folders[neg_folder]
            if corresp_pos_folder == 'HardEx':
                num_samples = 10
            elif corresp_pos_folder == 'MA_HE_HardEx':
                num_samples = 10
            elif corresp_pos_folder == 'MA_HE_SoftEx_HardEx':
                num_samples = 10
            elif corresp_pos_folder == 'MA_HardEx':
                num_samples = 6   
            elif (corresp_pos_folder == 'MA_HE_SoftEx_HardEx_irma') or (corresp_pos_folder=='HE_HardEx'):
                num_samples = 3 
            elif corresp_pos_folder == 'MA_SoftEx_HardEx':
                num_samples = 2
            else:
                num_samples = len(os.listdir(os.path.join('ConceptFoldersAll/SortedByCombinations',corresp_pos_folder)))
            print('Negative folder:',HardEx_negative_folders[neg_folder])
            print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select num_samples images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)

#10 SoftEx
#5 MA_HE_SoftEx_HardEx_NV_irma
#10 MA_HE_SoftEx_HardEx
#10 MA_HE_SoftEx_HardEx_irma
#5 HE_SoftEx_HardEx
#Add 5 more images to get 45 in total:
# HE_SoftEx (2)
#MA_SoftEx (2)
#MA_SoftEx_HardEx (1)

def moveSoftEx_examples(num_negative = 5):
    #NB! The order with positive and negative corresp. folders must be the same!
    SoftEx_positive_folders = ['SoftEx','MA_HE_SoftEx_HardEx_NV_irma','MA_HE_SoftEx_HardEx',
    'MA_HE_SoftEx_HardEx_irma','HE_SoftEx_HardEx',
    'HE_SoftEx','MA_SoftEx','MA_SoftEx_HardEx']
    SoftEx_negative_folders = ['NoAbnormalities','MA_HE_HardEx_NV_irma','MA_HE_HardEx',
    'MA_HE_HardEx_irma','HE_HardEx',
    'HE','MA','MA_HardEx']
    #Start with creating one positive examples folder
    for pos_folder in SoftEx_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/New_SoftEx_test20/PositiveExamples'
        #Pick 5 random images if one of these two folders:
        if (pos_folder=='MA_HE_SoftEx_HardEx_NV_irma') or (pos_folder == 'HE_SoftEx_HardEx'):
            num_samples = 5
        elif (pos_folder == 'HE_SoftEx') or (pos_folder=='MA_SoftEx'):
            num_samples = 2
        elif pos_folder == 'MA_SoftEx_HardEx':
            num_samples = 1
        else:
            num_samples = 10
        print('Positive folder:',pos_folder)
        print('Number of samples to pick for folder:',num_samples)
        image_list = os.listdir(folder_path)
        #Select num_samples images randomly
        selected_images = set(random.sample(image_list,num_samples))
        selected_images = list(selected_images)
        for _image in selected_images:
            source_path = os.path.join(folder_path,_image)
            target_path = os.path.join(target_folder,_image)
            shutil.copy(source_path,target_path)
    #Repeat for negative folder:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(SoftEx_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',SoftEx_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/New_SoftEx_test20/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = SoftEx_positive_folders[neg_folder]
            if (corresp_pos_folder == 'MA_HE_SoftEx_HardEx_NV_irma') or (corresp_pos_folder == 'HE_SoftEx_HardEx'):
                num_samples = 5
            elif (corresp_pos_folder == 'HE_SoftEx') or (corresp_pos_folder=='MA_SoftEx'):
                num_samples = 2
            elif corresp_pos_folder == 'MA_SoftEx_HardEx':
                num_samples = 1
            else:
                num_samples = 10
            print('Negative folder:',SoftEx_negative_folders[neg_folder])
            print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select num_samples images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
#HE: 14
#MA_HE_HardEx_irma: 1 
#MA_HE_SoftEx_HardEx_NV: 1
#MA_HE_HardEx: 8
#MA_HE_SoftEx_HardEx: 8   
#HE_HardEx: 8
#Add 5 more images to get 45 in total:
# HE: 2 more
# MA_HE: 3

def moveHE_examples(num_negative = 5):
    #NB! The corresponding negative folders must be in same order as positive folders here!
    HE_positive_folders = ['HE','MA_HE_HardEx_irma','MA_HE_SoftEx_HardEx_NV','MA_HE_HardEx',
    'MA_HE_SoftEx_HardEx','HE_HardEx',
    'MA_HE']
    HE_negative_folders = ['NoAbnormalities','MA_HardEx_irma','MA_SoftEx_HardEx_NV','MA_HardEx',
    'MA_SoftEx_HardEx','HardEx',
    'MA']       
    #Start with creating ONE positive folder:
    for pos_folder in HE_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/New_HE_test20/PositiveExamples'
        #Pick 16 random images if this folder:
        if pos_folder == 'HE':
            num_samples = 16
        #Pick 1 random image if one of these two folders:
        elif (pos_folder=='MA_HE_HardEx_irma') or (pos_folder == 'MA_HE_SoftEx_HardEx_NV'):
            num_samples = 1
        elif pos_folder == 'MA_HE':
            num_samples = 3
        else:
            print('Not among the other folders:',pos_folder)
            num_samples = 8
        print('Looking at folder:',pos_folder)
        print('NUmber of samples to select:',num_samples)
        image_list = os.listdir(folder_path)
        selected_images = set(random.sample(image_list,num_samples))
        selected_images = list(selected_images)
        print('Number of positive selected images:',len(selected_images))
        for _image in selected_images:
            source_path = os.path.join(folder_path,_image)
            target_path = os.path.join(target_folder,_image)
            print('Target path:',target_path)
            shutil.copy(source_path,target_path)
    #Repeat for the negative folders:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(HE_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',HE_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/New_HE_test20/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = HE_positive_folders[neg_folder]
            if corresp_pos_folder == 'HE':
                num_samples = 16
            elif (corresp_pos_folder == 'MA_HE_HardEx_irma') or (corresp_pos_folder == 'MA_HE_SoftEx_HardEx_NV'):
                num_samples = 1
            elif corresp_pos_folder == 'MA_HE':
                num_samples = 3
            else:
                num_samples = 8
            #print('Negative folder:',HE_negative_folders[neg_folder])
            #print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select num_samples images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)

#MA: 10
#MA_HE_SoftEx_HardEx: 10
#MA_HE_HardEx: 10
#MA_HE_SoftEx_HardEx_NV: 1
#MA_HE_HardEx_irma: 3
#MA_HE_SoftEx: 3
#MA_HardEx: 3
#Add 5 more images to get 45 in total:
#MA: 2 more
#MA_HE : 3

def moveMA_examples(num_negative=5):
    #NB! The corresponding negative folders must be in same order as positive folders in the lists here!
    MA_positive_folders = ['MA','MA_HE_SoftEx_HardEx','MA_HE_SoftEx_HardEx_NV','MA_HE_HardEx',
    'MA_HE_HardEx_irma','MA_HE_SoftEx','MA_HardEx',
    'MA_HE']
    MA_negative_folders = ['NoAbnormalities','HE_SoftEx_HardEx','HE_SoftEx_HardEx_NV','HE_HardEx',
    'HE_HardEx_irma','HE_SoftEx','HardEx',
    'HE']
    #Start with positive folders:
    for pos_folder in MA_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/New_MA_test20/PositiveExamples'
        #Pick 14 random images if this folder:
        if (pos_folder == 'MA_HE_SoftEx_HardEx') or (pos_folder=='MA_HE_HardEx'):
            num_samples = 10
        elif (pos_folder == 'MA'):
            num_samples = 12
        elif pos_folder == 'MA_HE_SoftEx_HardEx_NV':
            num_samples = 1
        else:
            num_samples = 3
        print('Looking at folder:',pos_folder)
        print('Number of samples:',num_samples)
        image_list = os.listdir(folder_path)
        selected_images = set(random.sample(image_list,num_samples))
        selected_images = list(selected_images)
        for _image in selected_images:
            source_path = os.path.join(folder_path,_image)
            target_path = os.path.join(target_folder,_image)
            shutil.copy(source_path,target_path)
    #Repeat for negative folders:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(MA_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',MA_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/New_MA_test20/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = MA_positive_folders[neg_folder]
            if (corresp_pos_folder == 'MA_HE_SoftEx_HardEx') or (corresp_pos_folder == 'MA_HE_HardEx'):
                num_samples = 10
            elif (corresp_pos_folder == 'MA'):
                num_samples = 12
            elif corresp_pos_folder == 'MA_HE_SoftEx_HardEx_NV':
                num_samples = 1
            else:
                num_samples = 3
            print('Negative folder:',MA_negative_folders[neg_folder])
            print('Number of samples to pick for folder:',num_samples)
            image_list = os.listdir(folder_path)
            #Select num_samples images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
            
def movePureMA_examples(num_negative=5):
    #NB! The corresponding negative folders must be in same order as positive folders in the lists here!
    MA_positive_folders = ['MA']
    MA_negative_folders = ['NoAbnormalities']
    #Start with positive folders:
    for pos_folder in MA_positive_folders:
        folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',pos_folder)
        target_folder = 'ConceptFoldersAll/PureMA/PositiveExamples'
        #Pick 40 random images if this folder:
        if (pos_folder == 'MA'):
            num_samples = 40
        image_list = os.listdir(folder_path)
        selected_images = set(random.sample(image_list,num_samples))
        selected_images = list(selected_images)
        for _image in selected_images:
            source_path = os.path.join(folder_path,_image)
            target_path = os.path.join(target_folder,_image)
            #shutil.copy(source_path,target_path)
    #Repeat for negative folders:
    for i in range(num_negative):
        random.seed(42+i)
        for neg_folder in range(len(MA_negative_folders)):
            folder_path = os.path.join('ConceptFoldersAll/SortedByCombinations',MA_negative_folders[neg_folder])
            target_folder = 'ConceptFoldersAll/PureMA/NegativeExamples' + str(i+1)
            #Pick same number of samples as number of samples in corresponding POSITIVE folder (for balanced distribution):
            corresp_pos_folder = MA_positive_folders[neg_folder]
            if (corresp_pos_folder == 'MA'):
                num_samples = 40
            image_list = os.listdir(folder_path)
            #Select num_samples images randomly
            selected_images = set(random.sample(image_list,num_samples))
            selected_images = list(selected_images)
            for _image in selected_images:
                source_path = os.path.join(folder_path,_image)
                target_path = os.path.join(target_folder,_image)
                shutil.copy(source_path,target_path)
            
#num_negative is the number of negative example folders we want to create
#I use 5 for now, but it can be increased if necessary

#moveNV_examples(num_negative = 10)
#moveIRMA_examples(num_negative = 10)
#moveHardEx_examples(num_negative = 10)
#moveSoftEx_examples(num_negative = 10)
#moveHE_examples(num_negative=10)
#moveMA_examples(num_negative=10)
#movePureMA_examples(num_negative=10)
#moveMA_examples(num_negative = 20)