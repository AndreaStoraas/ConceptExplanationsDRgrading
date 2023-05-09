import os
import shutil
import cv2 as cv
import numpy as np
import pandas as pd

########################
# This code finds the original concept image and matches it with the corresponding masked image
# To ensure that the exact same concept images are included in the TCAV experiments with and without masking
# (i.e., the only difference is the masking itself) 
# The masked images are copied over to their respective concept folders
########################


masked_conceptCombinations = 'LargerMaskedConceptAll'
masked_conceptFolders = os.listdir(masked_conceptCombinations)
#MA:
target_maskedConcepts_MA = 'MaskedConcepts/New_MA_test20'
original_MA_conceptFolder = '../ConceptFoldersAll/New_MA_test20'
original_MA_exampleFolders = os.listdir(original_MA_conceptFolder)
#HE:
target_maskedConcepts_HE = 'MaskedConcepts/New_HE_test20'
original_HE_conceptFolder = '../ConceptFoldersAll/New_HE_test20'
original_HE_exampleFolders = os.listdir(original_HE_conceptFolder)
#HardEx:
target_maskedConcepts_HardEx = 'MaskedConcepts/New_HardEx_test20'
original_HardEx_conceptFolder = '../ConceptFoldersAll/New_HardEx_test20'
original_HardEx_exampleFolders = os.listdir(original_HardEx_conceptFolder)
#SoftEx:
target_maskedConcepts_SoftEx = 'MaskedConcepts/New_SoftEx_test20'
original_SoftEx_conceptFolder = '../ConceptFoldersAll/New_SoftEx_test20'
original_SoftEx_exampleFolders = os.listdir(original_SoftEx_conceptFolder)
#NB! For NV and IRMA will original folders be in the FGADR concept folder!
#IRMA:
target_maskedConcepts_IRMA = 'MaskedConcepts/New_CompromisePureIRMA_test20'
original_IRMA_conceptFolder = '../ConceptFoldersFGADR/NewCompromisePureIRMA_test20'
original_IRMA_exampleFolders = os.listdir(original_IRMA_conceptFolder)
#NV:
target_maskedConcepts_NV = 'MaskedConcepts/New_CompromisePureNV_test20'
original_NV_conceptFolder = '../ConceptFoldersFGADR/NewCompromisePureNV_test20'
original_NV_exampleFolders = os.listdir(original_NV_conceptFolder)

def moveMA_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for MA:
    for MA_folder in original_MA_exampleFolders:
        print('Looking at MA folder:',MA_folder)
        folder_images = os.listdir(os.path.join(original_MA_conceptFolder,MA_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked MA-folder
        for image_name in folder_images:
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked MA-folder image:
                    target_path = os.path.join(target_maskedConcepts_MA,MA_folder,image_name)
                    shutil.copy(source_path,target_path)
        print('Number of images in original MA concept folder:', len(folder_images))
        print('Number of images in corresponding masked MA concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_MA,MA_folder))))

#Repeat for the HE concept folders:
def moveHE_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for HE:
    for HE_folder in original_HE_exampleFolders:
        print('Looking at HE folder:',HE_folder)
        folder_images = os.listdir(os.path.join(original_HE_conceptFolder,HE_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked HE-folder
        for image_name in folder_images:
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked HE-folder image:
                    target_path = os.path.join(target_maskedConcepts_HE,HE_folder,image_name)
                    shutil.copy(source_path,target_path)
        print('Number of images in original HE concept folder:', len(folder_images))
        print('Number of images in corresponding masked HE concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_HE,HE_folder))))

#Sort out the HardExudates concept folders:
def moveHardEx_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for HE:
    for HardEx_folder in original_HardEx_exampleFolders:
        
        print('Looking at HardEx folder:',HardEx_folder)
        folder_images = os.listdir(os.path.join(original_HardEx_conceptFolder,HardEx_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked HE-folder
        for image_name in folder_images:
            copied = 0
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked HE-folder image:
                    target_path = os.path.join(target_maskedConcepts_HardEx,HardEx_folder,image_name)
                    shutil.copy(source_path,target_path)
                    copied=1
            if copied == 0:
                print('Image not copied:',image_name)
        print('Number of images in original HardEx concept folder:', len(folder_images))
        print('Number of images in corresponding masked HardEx concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_HardEx,HardEx_folder))))

#Soft exudates
def moveSoftEx_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for HE:
    for SoftEx_folder in original_SoftEx_exampleFolders:
        print('Looking at SoftEx folder:',SoftEx_folder)
        folder_images = os.listdir(os.path.join(original_SoftEx_conceptFolder,SoftEx_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked HE-folder
        for image_name in folder_images:
            copied = 0
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked HE-folder image:
                    target_path = os.path.join(target_maskedConcepts_SoftEx,SoftEx_folder,image_name)
                    shutil.copy(source_path,target_path)
                    copied=1
            if copied == 0:
                print('Image not copied:',image_name)
        print('Number of images in original SoftEx concept folder:', len(folder_images))
        print('Number of images in corresponding masked SoftEx concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_SoftEx,SoftEx_folder))))

#IRMA from FGADR folder:
def moveIRMA_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for HE:
    for IRMA_folder in original_IRMA_exampleFolders:
        print('Looking at IRMA folder:',IRMA_folder)
        folder_images = os.listdir(os.path.join(original_IRMA_conceptFolder,IRMA_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked IRMA-folder
        for image_name in folder_images:
            copied = 0
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked IRMA-folder image:
                    target_path = os.path.join(target_maskedConcepts_IRMA,IRMA_folder,image_name)
                    shutil.copy(source_path,target_path)
                    copied=1
            if copied == 0:
                print('Image not copied:',image_name)
        print('Number of images in original IRMA concept folder:', len(folder_images))
        print('Number of images in corresponding masked IRMA concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_IRMA,IRMA_folder))))

#NV from FGADR folder:
def moveNV_masks2concepts():
    #Loop through all 20 negative and 1 positive example folders for NV:
    for NV_folder in original_NV_exampleFolders:
        print('Looking at NV folder:',NV_folder)
        folder_images = os.listdir(os.path.join(original_NV_conceptFolder,NV_folder))
        #For each image, want to copy same image name from 
        #corresponding masked conceptFolder to target masked IRMA-folder
        for image_name in folder_images:
            copied = 0
            for masked_folder in masked_conceptFolders:
                masked_images = os.listdir(os.path.join(masked_conceptCombinations,masked_folder))
                if image_name in masked_images:
                    #Masked sorted concept folder image
                    source_path = os.path.join(masked_conceptCombinations,masked_folder,image_name)
                    #Corresponding masked IRMA-folder image:
                    target_path = os.path.join(target_maskedConcepts_NV,NV_folder,image_name)
                    shutil.copy(source_path,target_path)
                    copied=1
            if copied == 0:
                print('Image not copied:',image_name)
        print('Number of images in original NV concept folder:', len(folder_images))
        print('Number of images in corresponding masked NV concept folder:',len(os.listdir(os.path.join(target_maskedConcepts_NV,NV_folder))))


#Uncomment the lines below to move the masked concept images
# over to their resepctive concept folders in the same pattern as for the full concept images
#moveMA_masks2concepts()
#moveHE_masks2concepts()
#moveHardEx_masks2concepts()
#moveSoftEx_masks2concepts()
#moveIRMA_masks2concepts()
#moveNV_masks2concepts()