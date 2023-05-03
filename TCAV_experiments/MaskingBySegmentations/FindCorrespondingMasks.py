import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import random

#FGADR:
FGADR_conceptsPath = '../ConceptFoldersFGADR/SortedByCombinations'
FGADR_segmentationsPath = '../../Data/FGADR-Seg-set'
FGADR_combinationFolders = os.listdir(FGADR_conceptsPath)
#DDR
#DDR_conceptsPath = '../ConceptFoldersDDR/SortedByCombinations'
#DDR_combinationFolders = os.listdir(DDR_conceptsPath)
#DDR_folder = '../../Data/DDR-datasetFebruary/lesion_segmentation'
#NB! Should look at both train, test and validation!
#ddr_train = os.path.join(DDR_folder, 'train','label')
#ddr_valid = os.path.join(DDR_folder, 'valid','segmentation label')
#ddr_test = os.path.join(DDR_folder, 'test','label')
#IDRiD:
#IDRiD_conceptsPath = '../ConceptFoldersIDRiD/SortedByCombinations'
#IDRiD_combinationFolders = os.listdir(IDRiD_conceptsPath)
#IDRiD_folder = '../../Data/IDRiD_dataFebruary/A.Segmentation/A. Segmentation/2. All Segmentation Groundtruths'
#IDRiD_train = os.path.join(IDRiD_folder,'a. Training Set')
#IDRiD_test = os.path.join(IDRiD_folder,'b. Testing Set')

# DiaretDB1:
#diaretDB_conceptsPath = '../ConceptFoldersDiaretDB/SortedByCombinationsDB1'
#diaretDB_combinationFolders = os.listdir(diaretDB_conceptsPath)
#diaretDB_folder =  '../../Data/diaretdb1_v_1_1/resources/images/ddb1_groundtruth'

#Get largest width and height:
largest_ybDist = 0
largest_xaDist = 0
#Get smallest width and height:
smallest_ybDist = 700
smallest_xaDist = 700


'''
#Check size of the cropped concept images...
All_croppedFolder = 'LargerMaskedConceptAll'
All_croppedCombinations = os.listdir(All_croppedFolder)
#Double check smallest and largest image for each combination of concepts
#For each concept combination:
for combo in All_croppedCombinations:
    print('Looking at combination:',combo)
    #Get the images for that specific concept combination
    combo_images = os.listdir(os.path.join(All_croppedFolder,combo))
    #Get the segmentation mask for each concept
    for _img in combo_images:
        #Get the original image:
        image_path = os.path.join(All_croppedFolder,combo,_img)
        my_image = cv.imread(image_path)
        height, widht, _ = my_image.shape
        if widht > largest_xaDist:
            largest_xaDist = widht
        if widht < smallest_xaDist:
            smallest_xaDist = widht
        if height > largest_ybDist:
            largest_ybDist = height
        if height < smallest_ybDist:
            smallest_ybDist = height
print('Tallest image:',largest_ybDist)
print('Lowest image:',smallest_ybDist)
print('Widest image:',largest_xaDist)
print('Smalest image:',smallest_xaDist)

'''

#For each concept combination:
for combo in FGADR_combinationFolders:
    print('Looking at combination:',combo)
    #Get the images for that specific concept combination
    combo_images = os.listdir(os.path.join(FGADR_conceptsPath,combo))
    #Split into the different individual concepts (since one mask for each concept)
    concepts_list = combo.split('_')
    print('The concepts present in this combo folder:',concepts_list)
    #Get the segmentation mask for each concept
    for _img in combo_images:
        #Get the original image:
        image_path = os.path.join(FGADR_conceptsPath,combo,_img)
        my_image = cv.imread(image_path)
        #my_image = cv.cvtColor(my_image, cv.COLOR_BGR2RGB)
        #mask_name = _img[:-3]+'tif' #Only for DDR dataset
        #Collect the masks for all concepts in a list:
        mask_lists = []
        for _concept in concepts_list:
            if _concept != 'NoAbnormalities':
                print('Name of image:',_img)
                if _concept == 'MA':
                    #segmentation_folder = os.path.join(diaretDB_folder,'redsmalldots')   
                    #mask_name = _img[:-4]+'_MA.tif' #<-Only relevant for IDRiD
                    #print('Name of corresponding mask:',mask_name)
                    #NB! Must check if image in train or test set of IDRiD, and train/test/valid for DDR!
                    #if mask_name in os.listdir(os.path.join(ddr_train,'MA')):
                    #    segmentation_folder = os.path.join(ddr_train,'MA')
                    #elif mask_name in os.listdir(os.path.join(ddr_valid,'MA')):
                    #    segmentation_folder = os.path.join(ddr_valid,'MA')
                    #elif mask_name in os.listdir(os.path.join(ddr_test,'MA')):
                    #    segmentation_folder = os.path.join(ddr_test,'MA')
                    #else:
                    #    print('Could not find segmentation folder!')
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'Microaneurysms_Masks')   
                elif _concept == 'HE':
                    #segmentation_folder = os.path.join(diaretDB_folder,'hemorrhages')   
                    #mask_name = _img[:-4]+'_HE.tif'
                    #if mask_name in os.listdir(os.path.join(ddr_train,'HE')):
                    #    segmentation_folder = os.path.join(ddr_train,'HE')
                    #elif mask_name in os.listdir(os.path.join(ddr_valid,'HE')):
                    #    segmentation_folder = os.path.join(ddr_valid,'HE')
                    #elif mask_name in os.listdir(os.path.join(ddr_test,'HE')):
                    #    segmentation_folder = os.path.join(ddr_test,'HE')
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'Hemohedge_Masks')
                elif _concept == 'HardEx':
                    #segmentation_folder = os.path.join(diaretDB_folder,'hardexudates')   
                    #mask_name = _img[:-4]+'_EX.tif'
                    #if mask_name in os.listdir(os.path.join(ddr_train,'EX')):
                    #    segmentation_folder = os.path.join(ddr_train,'EX')
                    #elif mask_name in os.listdir(os.path.join(ddr_valid,'EX')):
                    #    segmentation_folder = os.path.join(ddr_valid,'EX')
                    #elif mask_name in os.listdir(os.path.join(ddr_test,'EX')):
                    #    segmentation_folder = os.path.join(ddr_test,'EX')
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'HardExudate_Masks')
                elif _concept == 'SoftEx':
                    #segmentation_folder = os.path.join(diaretDB_folder,'softexudates')   
                    #mask_name = _img[:-4]+'_SE.tif'
                    #if mask_name in os.listdir(os.path.join(ddr_train,'SE')):
                    #    segmentation_folder = os.path.join(ddr_train,'SE')
                    #elif mask_name in os.listdir(os.path.join(ddr_valid,'SE')):
                    #    segmentation_folder = os.path.join(ddr_valid,'SE')
                    #elif mask_name in os.listdir(os.path.join(ddr_test,'SE')):
                    #    segmentation_folder = os.path.join(ddr_test,'SE')
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'SoftExudate_Masks')
                #This part is only relevant for FGADR dataset:
                elif _concept == 'NV':
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'Neovascularization_Masks')
                elif _concept == 'irma':
                    segmentation_folder = os.path.join(FGADR_segmentationsPath,'IRMA_Masks')
                else:
                    print('The concept is not recognized!')
                    break
                #Get the segmentation mask for each concept
                mask_path = os.path.join(segmentation_folder,_img)
                #mask_path = os.path.join(segmentation_folder,mask_name)
                my_mask = cv.imread(mask_path)
                #Check that mask and original image has the same size:
                if my_mask.shape != my_image.shape:
                    print('Mask size does not match the image size!')
                    break
                else:
                    gray = cv.cvtColor(my_mask, cv.COLOR_BGR2GRAY)
                    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
                    mask_lists.append(thresh)
        #If NoAbnormalities (-> mask_lists > 0):
        if len(mask_lists)>0:
            #Combine the masks to a large segmentation mask
            #Crop based on that common segmentation mask:
            #Possible way of cropping:
            #https://stackoverflow.com/questions/71734341/how-to-crop-and-save-segmented-objects-from-an-image
            # Find contours, obtain bounding box, extract and save ROI
            ROI_number = 0
            #Set initially low numbers for the rectangles:
            x,y,a,b = 0,0,0,0
            #For each segmentation mask connected to the image:
            for i in range(len(mask_lists)):
                cnts = cv.findContours(mask_lists[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    _x,_y,_w,_h = cv.boundingRect(c)
                    #If we want to draw a rectangle around the finding:
                    #cv.rectangle(my_image, (_x, _y), (_x + _w, _y + _h), (0,0,0), 2)
                    if ROI_number ==0:
                        x,y,a,b = _x,_y,_x + _w, _y + _w
                    #Want to have as large cropped area as possible to include all the findings:
                    else:
                        if _x < x:
                            x = _x
                        if _y < y:
                            y = _y
                        if (_x + _w)> a:
                            a = (_x + _w)
                        if (_y + _h)> b:
                            b = (_y + _h)
                    ROI_number += 1
            #1. Want to ensure square masks, since very rectangular are stretched a lot only one way
            y_bDist = b-y
            x_aDist = a-x
            #If too low image compared to width:
            if y_bDist<x_aDist:
                distance_togo = x_aDist - y_bDist
                #Adjust so that y and b gets same distance as x and a
                new_y = y - int(np.round(distance_togo/2,0))
                new_b = b + int(np.round(distance_togo/2,0))
                if new_y>=0: 
                    y = new_y
                #If new_y is negative, increase it to 0 to avoid error
                else:
                    y = 0
                if new_b<=my_image.shape[0]:
                    b = new_b
                #If new_b is larger than max image size:
                else:
                    b = my_image.shape[0]
            #If too 'smalt' image compared to height:
            elif y_bDist>x_aDist:
                distance_togo = y_bDist - x_aDist
                #Adjust so that x and a gets same distance as y and b
                new_x = x - int(np.round(distance_togo/2,0))
                new_a = a + int(np.round(distance_togo/2,0))
                if new_x>=0: 
                    x = new_x
                #If new_x is negative, increase to 0
                else:
                    x = 0
                if new_a<=my_image.shape[1]:
                    a = new_a
                #If new_a is larger than max image size:
                else:
                    a = my_image.shape[1]
            #Update the distances:
            y_bDist = b-y
            x_aDist = a-x
            #2. Want to prevent the cropped images to be smaller than 310x310
            if y_bDist<520:
                add_dist = 521-y_bDist
                new_y = y-(int(np.round(add_dist/2,0)))
                new_b = b+(int(np.round(add_dist/2,0)))
                if new_y>=0: 
                    y = new_y
                #If new_y is negative, increase it to 0 to avoid error
                else:
                    y = 0
                if new_b<=my_image.shape[0]:
                    b = new_b
                #If new_b is larger than max image size:
                else:
                    b = my_image.shape[0]
                #Again, want to check that the new image is still large enough:
                y_bDist = b-y
                if y_bDist<520:
                    print('New height is too small:', y_bDist)
                    if y==0:
                        b = 520
                    elif b==my_image.shape[0]:
                        y = my_image.shape[0] - 520
                    y_bDist = b-y
                print('New height after correction:',y_bDist)
            if x_aDist<520:
                add_dist = 521-x_aDist
                new_x = x-(int(np.round(add_dist/2,0)))
                new_a = a+(int(np.round(add_dist/2,0)))
                if new_x>=0: 
                    x = new_x
                else:
                    x = 0
                if new_a<=my_image.shape[1]:
                    a = new_a
                else:
                    a = my_image.shape[1]
                #Again, want to check that the new image is still large enough:
                x_aDist = a-x
                if x_aDist<520:
                    print('New width is too small:', x_aDist)
                    if x==0:
                        a = 520
                    elif a==my_image.shape[1]:
                        x = my_image.shape[1] - 520
                    x_aDist = a-x
                print('New width after correction:',x_aDist)
            #Just check largest and smallest image dimensions:
            if y_bDist>largest_ybDist:
                largest_ybDist = y_bDist
            if x_aDist>largest_xaDist:
                largest_xaDist = x_aDist
            if y_bDist<smallest_ybDist:
                smallest_ybDist = y_bDist
            if x_aDist<smallest_xaDist:
                smallest_xaDist = x_aDist
            ROI = my_image[y:b, x:a]
            save_path = os.path.join('LargerMaskedConceptFGADR',combo,_img) 
            cv.imwrite(save_path, ROI)
print('Widest image:',largest_xaDist)
print('Highest image:',largest_ybDist)
print('Smallest (smalest) image:',smallest_xaDist)
print('Lowest image:',smallest_ybDist)

print('Number of MA_HE_SoftEx in original folder:',len(os.listdir('../ConceptFoldersFGADR/SortedByCombinations/MA_HE_SoftEx_HardEx')))
print('Number of MA_HE_SoftEx in cropped folder:',len(os.listdir('LargerMaskedConceptFGADR/MA_HE_SoftEx_HardEx')))

