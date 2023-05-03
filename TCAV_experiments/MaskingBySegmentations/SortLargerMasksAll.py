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
common_folder = 'LargerMaskedConceptAll'
DDR_folder = 'LargerMaskedConceptDDR'
IDRiD_folder = 'LargerMaskedConceptIDRiD'
DiaretDB1_folder = 'LargerMaskedConceptDiaretDB1'
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
            shutil.copy(source_path, target_path)
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
            shutil.copy(source_path, target_path)
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
            shutil.copy(source_path, target_path)

#moveFolderImages()

#Check that the number of images in original folders and new folder sums up (in case images with same name etc)
for _folder in common_combos:
    #if 'irma' not in _folder:
        #print('IRMA not in folder:',_folder)
        #print('Number of images:', len(os.listdir(os.path.join(common_folder, _folder))))
    #if 'irma' in str(_folder)[-4:]:
    #    print('MA in folder:',_folder)
    print('Looking at folder:',_folder)
    print('Number of images in "all" folder:',len(os.listdir(os.path.join(common_folder,_folder))))
    print('Number of images in FGADR:',len(os.listdir(os.path.join('LargerMaskedConceptFGADR',_folder))))
    if _folder in os.listdir(DDR_folder):
        print('Number of images in DDR:',len(os.listdir(os.path.join(DDR_folder,_folder))))
    if _folder in os.listdir(IDRiD_folder):
        print('Number of images in IDRiD:',len(os.listdir(os.path.join(IDRiD_folder,_folder))))
    if _folder in os.listdir(DiaretDB1_folder):
        print('Number of images in DiaretDB1:',len(os.listdir(os.path.join(DiaretDB1_folder,_folder))))
    print('**************')

