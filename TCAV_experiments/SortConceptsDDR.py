import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd

ddr_folder = '../Data/DDR-datasetFebruary/lesion_segmentation'
#NB! Should look at both train, test and validation!
ddr_train = os.path.join(ddr_folder, 'train','label')
ddr_valid = os.path.join(ddr_folder, 'valid','segmentation label')
ddr_test = os.path.join(ddr_folder, 'test','label')

train_masks = os.listdir(os.path.join(ddr_train,'EX'))
valid_masks = os.listdir(os.path.join(ddr_valid,'EX'))
test_masks = os.listdir(os.path.join(ddr_test,'EX'))
#Create a list of all relevant images/masks
#Since all segmented images (757) have corresponding masks for all
#abnormalities, it doesn't matter which abnormality we take from
all_masks_list = os.listdir(os.path.join(ddr_train,'EX'))
all_masks_list += valid_masks
all_masks_list += test_masks
print('Number of segmented images:',len(all_masks_list))

def create_overviewDf():
    #For each image, we want to look if the segmentation masks
    #have segmentations or not
    abnormalities = ['EX','HE','MA','SE']
    #Create a df with overview for each image
    #Start with the training files:
    overview_dfTrain = pd.DataFrame(columns = ['mask_name', 'EX','HE','MA','SE'])
    for i in range(len(train_masks)):
        _file = train_masks[i]
        overview_dfTrain.loc[i] = _file, 'lol','lol','lol','lol'
        for abnorm in abnormalities:
            #print('Inspecting abnormality',abnorm)
            mask_path = os.path.join(ddr_train,abnorm,_file)
            #print(mask_path)
            my_mask = cv.imread(mask_path)
            #print(np.unique(my_mask[:,:,0]))
            #print(np.unique(my_mask[:,:,1]))
            #print(np.unique(my_mask[:,:,2]))
            #If segmentation mask is completely black, the abnormality is not present
            if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                #print('No abnormality in segmentation mask')
                overview_dfTrain.loc[i,abnorm] = 0
            else:
                overview_dfTrain.loc[i,abnorm] = 1
    print(overview_dfTrain.head()) 

    #Repeat with the validation files:
    overview_dfValid = pd.DataFrame(columns = ['mask_name', 'EX','HE','MA','SE'])
    for i in range(len(valid_masks)):
        _file = valid_masks[i]
        overview_dfValid.loc[i] = _file, 'lol','lol','lol','lol'
        for abnorm in abnormalities:
            mask_path = os.path.join(ddr_valid,abnorm,_file)
            my_mask = cv.imread(mask_path)
            #If segmentation mask is completely black, the abnormality is not present
            if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                #print('No abnormality in segmentation mask')
                overview_dfValid.loc[i,abnorm] = 0
            else:
                overview_dfValid.loc[i,abnorm] = 1
    print(overview_dfValid.head()) 

    #And the test files:
    overview_dfTest = pd.DataFrame(columns = ['mask_name', 'EX','HE','MA','SE'])
    for i in range(len(test_masks)):
        _file = test_masks[i]
        overview_dfTest.loc[i] = _file, 'lol','lol','lol','lol'
        for abnorm in abnormalities:
            mask_path = os.path.join(ddr_test,abnorm,_file)
            my_mask = cv.imread(mask_path)
            #If segmentation mask is completely black, the abnormality is not present
            if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                #print('No abnormality in segmentation mask')
                overview_dfTest.loc[i,abnorm] = 0
            else:
                overview_dfTest.loc[i,abnorm] = 1
    print(overview_dfTest.head()) 
    #Concat the three df's into one big df
    overviewDf = pd.concat([overview_dfTrain,overview_dfValid,overview_dfTest],axis = 0,ignore_index = True)
    print('Shape of total overview DF:',overviewDf.shape)
    #Save as csv file in the DDR folder:
    overviewDf.to_csv('ConceptFoldersDDR/file_overviewAbnormalities.csv',index = False)


def sortByCombinations():
    #Read in the overview DF:
    overview_df = pd.read_csv('ConceptFoldersDDR/file_overviewAbnormalities.csv')
    print('Shape of the overview DF:',overview_df.shape)
    print('Beginning of overview df:', overview_df.head())
    sorted_folder = 'ConceptFoldersDDR/SortedByCombinations'
    ex_counter = 0
    he_counter = 0
    ma_counter = 0
    se_counter = 0
    ma_he_counter = 0
    ma_se_counter = 0
    ma_ex_counter = 0
    he_se_counter = 0
    he_ex_counter = 0
    se_ex_counter = 0
    ma_he_se_counter = 0
    ma_he_ex_counter = 0
    ma_se_ex_counter = 0
    he_se_ex_counter = 0
    ma_he_se_ex_counter = 0
    for i in range(overview_df.shape[0]):
        #Convert the mask name (.tif) to image_name (.jpg)
        mask_name = overview_df.iloc[i,0]
        image_name = mask_name[:-3]
        image_name = image_name + 'jpg'
        #Check if the image and mask are in train, test or validation folder
        if mask_name in train_masks:
            subfolder = 'train'
        elif mask_name in valid_masks:
            subfolder = 'valid'
        elif mask_name in test_masks:
            subfolder = 'test'
        source_pathImage = os.path.join(ddr_folder,subfolder,'image',image_name)
        #All segmentations should have abnormalities, since DR level ranges from 1-4 (ref Li et al)
        if (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0):
            print('No abnormalities')
        #If only hard exudates:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0):
            #print(overview_df.iloc[i,:])
            target_path = os.path.join(sorted_folder,'HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ex_counter += 1
        #If only hemorrhages:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'HE',image_name)
            #shutil.copy(source_pathImage,target_path)
            he_counter += 1
        #If only microaneurysms:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'MA',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_counter += 1
        #If only soft exudates:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'SoftEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            se_counter += 1
            #print('Corresponding filename:',image_name)
            #print('Source path:',source_pathImage)
            #print('Saving path:',target_path)
        #If MA + HE:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'MA_HE',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_he_counter +=1
        #If MA + SoftEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'MA_SoftEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_se_counter +=1
        #If MA + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'MA_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_ex_counter += 1
        #If HE + Soft Ex
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'HE_SoftEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            he_se_counter += 1
        #If HE + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'HE_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            he_ex_counter +=1
        #If SoftEx + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'SoftEx_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            se_ex_counter += 1
        #If MA + HE + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0):
            target_path = os.path.join(sorted_folder,'MA_HE_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_he_ex_counter += 1
        #If MA + HE + SoftEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'MA_HE_SoftEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_he_se_counter +=1
        #IF MA + SoftEx + HardEx
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'MA_SoftEx_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_se_ex_counter += 1
        #If HE + SoftEx + HardEx
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'HE_SoftEx_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            he_se_ex_counter +=1
        #If all 4 are present:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1):
            target_path = os.path.join(sorted_folder,'MA_HE_SoftEx_HardEx',image_name)
            #shutil.copy(source_pathImage,target_path)
            ma_he_se_ex_counter+=1
        else:
            print('This is not covered by the filters:')
            print(overview_df.iloc[i,:])

    
    print('Number of images with only hard exudates:',ex_counter)
    print('Number of images with only hemorrhages:',he_counter)
    print('Number of images with only microaneurysms:',ma_counter)
    print('Number of images with only soft exudates:',se_counter)
    print('*********************')
    print('Number of images with MA + HE:',ma_he_counter)
    print('Number of images with MA + softEx:',ma_se_counter)
    print('Number of images with MA + HardEx:',ma_ex_counter)
    print('Number of images with HE + soft exudates:',he_se_counter)
    print('Number of images with HE + hard exudates:',he_ex_counter)
    print('Number of images with soft exudates + hard exudates:',se_ex_counter)
    print('*******************')
    print('Number of images with MA + HE + soft exudates:',ma_he_se_counter)
    print('Number of images with MA + HE + hard exudates:',ma_he_ex_counter)
    print('Number of images with MA + soft exudates + hard exudates:',ma_se_ex_counter)
    print('Number of images with HE + soft exudates + hard exudates:',he_se_ex_counter)
    print('Number of images with MA + HE + soft exudates + hard exudates:',ma_he_se_ex_counter)
    print('*************************')
    print('Sum of obsevations:', str(ex_counter+he_counter+ma_counter+se_counter+ma_he_counter+ma_se_counter+ma_ex_counter+he_se_counter+he_ex_counter+se_ex_counter+ma_he_se_counter+ma_he_ex_counter+ma_se_ex_counter+he_se_ex_counter+ma_he_se_ex_counter))

    #Check that all images were copied to correct folder
    target_foldersList = ['HardEx','HE','MA','SoftEx','MA_HE','MA_SoftEx','MA_HardEx','HE_SoftEx','HE_HardEx',
    'SoftEx_HardEx','MA_HE_SoftEx','MA_HE_HardEx','MA_SoftEx_HardEx','HE_SoftEx_HardEx','MA_HE_SoftEx_HardEx']

    for _folder in target_foldersList:
        print('Number of files in folder',_folder,len(os.listdir(os.path.join('ConceptFoldersDDR/SortedByCombinations',_folder))))
'''
concept_folder = 'ConceptFoldersDiaretDB/SortedByCombinationsDB1'
total_images = 0
combi_list = os.listdir(concept_folder)
for _combi in combi_list:
    print('Looking at concept combination:',_combi)
    if _combi != 'NoAbnormalities':
        print('Number of images:',len(os.listdir(os.path.join(concept_folder,_combi))))
        total_images += len(os.listdir(os.path.join(concept_folder,_combi)))
    else:
        print('Number of noDR images:',len(os.listdir(os.path.join(concept_folder,_combi))))
print('Total number of sorted images:', total_images)
'''
'''
#Move all class 0 images over to the NoAbnormalities folder
ddr_gradingFolder = '../Data/DDR-datasetFebruary/DR_grading'
ddr_gradingTrain = pd.read_table(os.path.join(ddr_gradingFolder,'train.txt'),sep=' ', header = None)
ddr_gradingValid = pd.read_table(os.path.join(ddr_gradingFolder,'valid.txt'),sep=' ',header = None)
ddr_gradingTest = pd.read_table(os.path.join(ddr_gradingFolder,'test.txt'),sep=' ',header = None)

ddr_gradingTrain = ddr_gradingTrain.rename(columns={0:'Image',1:'DR_level'})
ddr_gradingTest = ddr_gradingTest.rename(columns={0:'Image',1:'DR_level'})
ddr_gradingValid = ddr_gradingValid.rename(columns={0:'Image',1:'DR_level'})
print(ddr_gradingTrain.columns)
print(ddr_gradingTrain.head())

#Go through all images and move the ones with grade 0 to the DDR noAbnormality folder
ddr_trainImages = os.path.join(ddr_gradingFolder,'train')
ddr_validImages = os.path.join(ddr_gradingFolder,'valid')
ddr_testImages = os.path.join(ddr_gradingFolder,'test')

noAbnormalityFolder = 'ConceptFoldersDDR/SortedByCombinations/NoAbnormalities'
noAbnorm_counter = 0
#First, look at the train images
for i in range(ddr_gradingTrain.shape[0]):
    image_name = ddr_gradingTrain.iloc[i,0]
    if ddr_gradingTrain.iloc[i,1]==0:
        source_path = os.path.join(ddr_trainImages,image_name)
        target_path = os.path.join(noAbnormalityFolder,image_name)
        #shutil.copy(source_path,target_path)
        noAbnorm_counter+=1
#Continue with the valid and test images:
for i in range(ddr_gradingValid.shape[0]):
    image_name = ddr_gradingValid.iloc[i,0]
    if ddr_gradingValid.iloc[i,1]==0:
        source_path = os.path.join(ddr_validImages,image_name)
        target_path = os.path.join(noAbnormalityFolder,image_name)
        #shutil.copy(source_path,target_path)
        noAbnorm_counter+=1
for i in range(ddr_gradingTest.shape[0]):
    image_name = ddr_gradingTest.iloc[i,0]
    if ddr_gradingTest.iloc[i,1]==0:
        source_path = os.path.join(ddr_testImages,image_name)
        target_path = os.path.join(noAbnormalityFolder,image_name)
        #shutil.copy(source_path,target_path)
        noAbnorm_counter+=1
print('Number of healthy images in the dataset:',noAbnorm_counter)
print('Number of healthy images in NoAbnormality folder:',len(os.listdir(noAbnormalityFolder)))
'''

overview_df = pd.read_csv('ConceptFoldersDDR/file_overviewAbnormalities.csv')
no_abnorm = 0
for i in range(overview_df.shape[0]):
    if (overview_df.iloc[i,1] == 0) and (overview_df.iloc[i,2] == 0) and (overview_df.iloc[i,3] == 0) and (overview_df.iloc[i,4] == 0):
        no_abnorm += 1
#print('Number of healthy images in overview df:',no_abnorm)
#print('Number of sorted healthy images:',len(os.listdir('ConceptFoldersIDRiD/SortedByCombinations/NoAbnormalities')))
#print('Number of concept images:',len(os.listdir('ConceptFoldersIDRiD/SortedByCombinations/MA_HardEx'))+len(os.listdir('ConceptFoldersIDRiD/SortedByCombinations/MA_HE_HardEx'))+len(os.listdir('ConceptFoldersIDRiD/SortedByCombinations/MA_HE_SoftEx_HardEx')))

print('Concept train DDR:',len(os.listdir('../Data/CroppedDataKaggle/CroppedTrainDDR')))
print('Concept valid DDR:',len(os.listdir('../Data/CroppedDataKaggle/CroppedValidDDR')))
print('Concept test DDR:',len(os.listdir('../Data/CroppedDataKaggle/CroppedTestDDR')))
