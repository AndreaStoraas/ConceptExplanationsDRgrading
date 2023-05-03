import numpy as np
import pandas as pd
import os
import shutil
import cv2

#Want to find images that contains one/more abnormalities to
#find positive/negative examples for training the
#concept activation vectors (CAVs)

#Start with the DIARETDB1 dataset, as we have segmentation masks
#for 4 potentially relevant concepts (but not NV)
diaret_images = os.listdir('../Data/diaretdb1_v_1_1/resources/images/ddb1_fundusimages')
#print('Some example images:',diaret_images[:5])
print('Total number of images:',len(diaret_images))

#Want to find images that do not have completely black segmentation masks
# for a given abnormality
mask_path = '../Data/diaretdb1_v_1_1/resources/images/ddb1_groundtruth'

'''
smalldots_path = os.path.join(mask_path,'redsmalldots')
smalldots_images = os.listdir(smalldots_path)
print('Number of images with redsmalldots masks:',len(smalldots_images))

images_with_smalldots = []
for _img in smalldots_images:
    img_mask = os.path.join(smalldots_path,_img)
    #Read in the segmentation mask:
    mask = cv2.imread(img_mask,0)
    #Look at the different values in the mask
    #print(np.unique(mask))
    #If only 0's, this means the abnormality is not present in the image
    if (len(np.unique(mask))==1) and (mask[0][0]==0):
        print('Entire mask is black!')
    else:
        #This image includes the abnormality
        images_with_smalldots.append(_img)
print('Number of images which actually contain redsmalldots:', len(images_with_smalldots))
#print(images_with_smalldots)

#Repeat for the other abnormalities as well:
#Hemorrhages:
hemorrhage_path = os.path.join(mask_path,'hemorrhages')
hemorrhage_images = os.listdir(hemorrhage_path)
print('Number of images with hemorrhage masks:',len(hemorrhage_images))

images_with_hemorrhage = []
for _img in hemorrhage_images:
    img_mask = os.path.join(hemorrhage_path,_img)
    #Read in the segmentation mask:
    mask = cv2.imread(img_mask,0)
    #Look at the different values in the mask
    #print(np.unique(mask))
    #If only 0's, this means the abnormality is not present in the image
    if (len(np.unique(mask))==1) and (mask[0][0]==0):
        print('Entire mask is black!')
    else:
        #This image includes the abnormality
        images_with_hemorrhage.append(_img)
print('Number of images which actually contain hemorrhages:', len(images_with_hemorrhage))
#print(images_with_hemorrhage)

#Hard exudates:
hardEx_path = os.path.join(mask_path,'hardexudates')
hardEx_images = os.listdir(hardEx_path)
print('Number of images with hard exudates-masks:',len(hardEx_images))

images_with_hardEx = []
for _img in hardEx_images:
    img_mask = os.path.join(hardEx_path,_img)
    #Read in the segmentation mask:
    mask = cv2.imread(img_mask,0)
    #Look at the different values in the mask
    #print(np.unique(mask))
    #If only 0's, this means the abnormality is not present in the image
    if (len(np.unique(mask))==1) and (mask[0][0]==0):
        print('Entire mask is black!')
    else:
        #This image includes the abnormality
        images_with_hardEx.append(_img)
print('Number of images which actually contain hard exudates:', len(images_with_hardEx))


#Soft exudates:
softEx_path = os.path.join(mask_path,'softexudates')
softEx_images = os.listdir(softEx_path)
print('Number of images with soft exudates-masks:',len(softEx_images))

images_with_softEx = []
for _img in softEx_images:
    img_mask = os.path.join(softEx_path,_img)
    #Read in the segmentation mask:
    mask = cv2.imread(img_mask,0)
    #Look at the different values in the mask
    #print(np.unique(mask))
    #If only 0's, this means the abnormality is not present in the image
    if (len(np.unique(mask))==1) and (mask[0][0]==0):
        print('Entire mask is black!')
    else:
        #This image includes the abnormality
        images_with_softEx.append(_img)
print('Number of images which actually contain soft exudates:', len(images_with_softEx))
'''

#Next, we can pick representative example images for the different concepts:
def sortByAbnormalities():
    sortedAbnormalities_path = 'DiaretDB1_sortedAbnormalities'
    for _img in images_with_smalldots:
        #If small red dots in image, we copy it to the redsmalldots
        source_path = os.path.join(smalldots_path,_img)
        target_path = os.path.join(sortedAbnormalities_path,'redsmalldots',_img)
        shutil.copy(source_path, target_path)
    for _img in images_with_hemorrhage:
        source_path = os.path.join(hemorrhage_path,_img)
        target_path = os.path.join(sortedAbnormalities_path,'hemorrhages',_img)
        shutil.copy(source_path,target_path)
    for _img in images_with_hardEx:
        source_path = os.path.join(hardEx_path,_img)
        target_path = os.path.join(sortedAbnormalities_path,'hardexudates',_img)
        shutil.copy(source_path,target_path)
    for _img in images_with_softEx:
        source_path = os.path.join(softEx_path,_img)
        target_path = os.path.join(sortedAbnormalities_path,'softexudates',_img)
        shutil.copy(source_path,target_path)

#sortByAbnormalities()
#Check that all images were correctly moved:
#print('Number of images in MA folder:', len(os.listdir('DiaretDB1_sortedAbnormalities/redsmalldots')))
#print('Number of images in hemorrhage folder:', len(os.listdir('DiaretDB1_sortedAbnormalities/hemorrhages')))
#print('Number of images in hard exudates folder:', len(os.listdir('DiaretDB1_sortedAbnormalities/hardexudates')))
#print('Number of images in soft exudates folder:', len(os.listdir('DiaretDB1_sortedAbnormalities/softexudates')))

def createOverviewDf():
    #Also want to create an overview df over which abnormalities that are present in which image
    #Similar to what was done for the DiaretDB0 dataset
    overviewDf = pd.DataFrame(columns = ['ImageName','redsmalldots','Hemorrhages','HardEx','SoftEx'])
    #Loop through all images and say whether each of the four abnormalities are present or not
    for i in range(len(diaret_images)):
        _img = diaret_images[i]
        if _img in images_with_smalldots:
            has_smalldots = 1
        else:
            has_smalldots = 0
        if _img in images_with_hemorrhage:
            has_hemorrhage = 1
        else:   
            has_hemorrhage = 0
        if _img in images_with_hardEx:
            has_hardEx = 1
        else:
            has_hardEx = 0
        if _img in images_with_softEx:
            has_softEx = 1
        else:
            has_softEx = 0
        #Append the values to the df:
        overviewDf.loc[i] = _img, has_smalldots, has_hemorrhage, has_hardEx, has_softEx
    print(overviewDf.tail(10))
    print('Number of rows in the df:',overviewDf.shape[0])
    #Write to csv for later reference:
    overviewDf.to_csv('DiaretDB1_overviewAbnormalities.csv',header=True)

#createOverviewDf()

#Read in the overview file:
overviewDf = pd.read_csv('ConceptFoldersDiaretDB/DiaretDB1_overviewAbnormalities.csv', index_col = 'Unnamed: 0')
#Loop through and see which images have MA + all other abnormalities
#And which have not MA, but the rest of the abonrmalities
# -> Several images with all abnormalities present
# BUT only ONE image without MA, where all other abnormalities are present...


#Also inspect hemorrhages, soft and hard exudates:
print(overviewDf.head())
print('Total number of images:',overviewDf.shape[0])
def SortByCombinations():
    target_folder = 'ConceptFoldersDiaretDB/SortedByCombinations'
    source_folder = '../Data/diaretdb1_v_1_1/resources/images/ddb1_fundusimages'
    noAbnorm_counter = 0
    MA_counter = 0
    HE_counter = 0
    SoftEx_counter = 0
    HardEx_counter = 0
    MA_HE_counter = 0
    MA_SoftEx_counter = 0
    MA_HardEx_counter = 0
    HE_SoftEx_counter = 0
    HE_HardEx_counter = 0
    SoftEx_HardEx_counter = 0
    MA_HE_SoftEx_counter = 0
    MA_HE_HardEx_counter = 0
    MA_SoftEx_HardEx_counter = 0
    HE_SoftEx_HardEx_counter = 0
    MA_HE_SoftEx_HardEx_counter = 0
    for i in range(overviewDf.shape[0]):
        _img = overviewDf.iloc[i,0]
        source_path = os.path.join(source_folder,_img)
        #If no abnormalities:
        if (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'NoAbnormalities',_img)
            #shutil.copy(source_path,target_path)
            noAbnorm_counter += 1
        #If only MA:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'MA',_img)
            #shutil.copy(source_path,target_path)
            MA_counter += 1
        #If only Hemo:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'HE',_img)
            #shutil.copy(source_path,target_path)
            HE_counter += 1
        #If only SoftEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'SoftEx',_img)
            #shutil.copy(source_path,target_path)
            SoftEx_counter += 1
        #If only HardEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'HardEx',_img)
            #shutil.copy(source_path,target_path)
            HardEx_counter += 1
        #If MA + HE:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'MA_HE',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_counter += 1
        #If MA + SoftEx:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            MA_SoftEx_counter += 1
        #If MA + HardEx:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'MA_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HardEx_counter += 1
        #If HE + SoftEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            HE_SoftEx_counter += 1
        #If HE + HardEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            HE_HardEx_counter += 1
        #If SoftEx + HardEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            SoftEx_HardEx_counter += 1
        #If MA_HE_SoftEx:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==0) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_SoftEx_counter += 1
        #If MA + HE + HardEx:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==0):
            target_path = os.path.join(target_folder,'MA_HE_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_HardEx_counter += 1
        #If MA + SoftEx + HardEx:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==0) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_SoftEx_HardEx_counter += 1
        #If HE + SoftEx + HardEx:
        elif (overviewDf.iloc[i,1]==0) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            HE_SoftEx_HardEx_counter += 1
        #If all abnormalities:
        elif (overviewDf.iloc[i,1]==1) and (overviewDf.iloc[i,2]==1) and (overviewDf.iloc[i,3]==1) and (overviewDf.iloc[i,4]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_HardEx',_img)
            #shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_counter += 1
    print('Number of no DR images:',noAbnorm_counter)
    print('Number of MA images:',MA_counter)
    print('Number of HE images:',HE_counter)
    print('Number of SoftEx images:',SoftEx_counter)
    print('Number of HardEx images:',HardEx_counter)
    print('Number of MA + HE images:',MA_HE_counter)
    print('Number of MA +SoftEx images:',MA_SoftEx_counter)
    print('Number of MA + HardEx images:',MA_HardEx_counter)
    print('Number of HE + SoftEx images:',HE_SoftEx_counter)
    print('Number of HE + HardEx images:',HE_HardEx_counter)
    print('Number of SoftEx + HardEx images:',SoftEx_HardEx_counter)
    print('Number of MA + HE + SoftEx images:',MA_HE_SoftEx_counter)
    print('Number of MA + HE + HardEx images:',MA_HE_HardEx_counter)
    print('Number of MA + SoftEx + HardEx images:',MA_SoftEx_HardEx_counter)
    print('Number of HE + SoftEx + HardEx images:', HE_SoftEx_HardEx_counter)
    print('Number of ALL abnorm images:',MA_HE_SoftEx_HardEx_counter)
    print('*************************')
    print('Number of sorted images with no DR:',len(os.listdir(os.path.join(target_folder, 'NoAbnormalities'))))
    print('Number of sorted images with MA:',len(os.listdir(os.path.join(target_folder, 'MA'))))
    print('Number of sorted images with HE:',len(os.listdir(os.path.join(target_folder, 'HE'))))
    print('Number of sorted images with MA + HE:',len(os.listdir(os.path.join(target_folder, 'MA_HE'))))
    print('Number of sorted images with MA + SoftEx:',len(os.listdir(os.path.join(target_folder, 'MA_SoftEx'))))
    print('Number of sorted images with MA + HardEx:',len(os.listdir(os.path.join(target_folder, 'MA_HardEx'))))
    print('Number of sorted images with HE + SoftEx:',len(os.listdir(os.path.join(target_folder, 'HE_SoftEx'))))
    print('Number of sorted images with SoftEx + HardEx:',len(os.listdir(os.path.join(target_folder, 'SoftEx_HardEx'))))
    print('Number of sorted images with MA + HE + SoftEx:',len(os.listdir(os.path.join(target_folder, 'MA_HE_SoftEx'))))
    print('Number of sorted images with MA + HE + HardEx:',len(os.listdir(os.path.join(target_folder, 'MA_HE_HardEx'))))
    print('Number of sorted images with  HE + SoftEx + HardEx:',len(os.listdir(os.path.join(target_folder, 'HE_SoftEx_HardEx'))))
    print('Number of sorted images with ALL abnormalities:',len(os.listdir(os.path.join(target_folder, 'MA_HE_SoftEx_HardEx'))))

#SortByCombinations()
