import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd

#Want to have 6 concepts here:
#MA, HE, HardEx, SoftEx, IRMA and NV
fgadr_folder = '../Data/FGADR-Seg-set'
abnormality_list = ['HardExudate_Masks','Hemohedge_Masks','IRMA_Masks','Microaneurysms_Masks',
    'Neovascularization_Masks','SoftExudate_Masks']

def createOverviewDf():
    overview_df = pd.DataFrame(columns = ['image_name', 'Microaneurysms','Hemohedge','SoftExudate','HardExudate','Neovascularization','IRMA'])
    #Should create an overview DF to see which abnormalities that are present for each image
    all_images = os.listdir(os.path.join(fgadr_folder,'Original_Images'))
    for i in range(len(all_images)):
        _img = all_images[i]
        overview_df.loc[i] = _img, 'lol','lol','lol','lol','lol','lol'
        for anorm in abnormality_list:
            anorm_name = anorm[:-6]
            #First, check if the image is present in the abnormality folder:
            anorm_images = os.listdir(os.path.join(fgadr_folder,anorm))
            #If the image is present, we check if the corresponding mask has segmentations:
            if _img in anorm_images:
                #Read in the corresponding mask
                mask_path = os.path.join(fgadr_folder,anorm,_img)
                my_mask = cv.imread(mask_path)
                #If the mask is completely black, the abnormality is not present for that image:
                if (len(np.unique(my_mask))==1) and (np.unique(my_mask)[0]==0):
                    overview_df.loc[i,anorm_name] = 0
                #Else, the abnormality is present and is flagged to 1
                else:
                    overview_df.loc[i,anorm_name] = 1
            #If the image is not present in the abnormality folder, then the abnormality is not present:
            else:
                overview_df.loc[i,anorm_name] = 0

    #Save the df as csv:
    overview_df.to_csv('ConceptFoldersFGADR/file_overviewAbnormalities.csv',index = False)

overview_df = pd.read_csv('ConceptFoldersFGADR/file_overviewAbnormalities.csv')
print(overview_df.head())
#Want to create subfolders based on which combinations of anormalities that are present
#Start with the "single" concept combinations
def pickSingleAbnormalities():
    noAnorm_count = 0
    MA_count = 0
    Hemo_count = 0
    SoftEx_count = 0
    HardEx_count = 0
    NV_count = 0
    IRMA_count = 0
    target_folder = 'ConceptFoldersFGADR/SortedByCombinations'
    for i in range(overview_df.shape[0]):
        _img = overview_df.iloc[i,0]
        source_path = os.path.join(fgadr_folder,'Original_Images',_img)
        #If no abnormalities:
        if (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'NoAbnormalities')
            shutil.copy(source_path,target_path)
            noAnorm_count +=1
        #If only MA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA')
            shutil.copy(source_path,target_path)
            MA_count += 1
        #If only Hemo
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE')
            shutil.copy(source_path,target_path)
            Hemo_count += 1
        #If only SoftEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'SoftEx')
            shutil.copy(source_path,target_path)
            SoftEx_count += 1
        #If only HardEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HardEx')
            shutil.copy(source_path,target_path)
            HardEx_count +=1
        #If only NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'NV')
            shutil.copy(source_path,target_path)
            NV_count += 1
        #If only IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            #No IRMA, so I don't need to move the images
            target_path = os.path.join(target_folder,'IRMA')
            shutil.copy(source_path,target_path)
            IRMA_count += 1
    print('No abnormalities:',noAnorm_count)
    print('MA images:',MA_count)
    print('Hemo images:', Hemo_count)
    print('SoftEx images:',SoftEx_count)
    print('HardEx images:',HardEx_count)
    print('NV images:', NV_count)
    print('IRMA images:',IRMA_count)
    print('******************')
    print('Number of images in no abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'NoAbnormalities'))))
    print('Number of images in MA concept folder:',len(os.listdir(os.path.join(target_folder,'MA'))))
    print('Number of images in no HEMO folder:',len(os.listdir(os.path.join(target_folder,'HE'))))
    print('Number of images in SoftEx concept folder:',len(os.listdir(os.path.join(target_folder,'SoftEx'))))
    print('Number of images in HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'HardEx'))))
    print('Number of images in NV concept folder:',len(os.listdir(os.path.join(target_folder,'NV'))))

#Sort the all the combinations of two concepts:
def pickDoubleAbnormalities():
    MA_HE_count = 0
    MA_SoftEx_count = 0
    MA_HardEx_count = 0
    MA_NV_count = 0
    MA_IRMA_count = 0
    HE_SoftEx_count = 0
    HE_HardEx_count = 0
    HE_NV_count = 0
    HE_IRMA_count = 0
    SoftEx_HardEx_count = 0
    SoftEx_NV_count = 0
    SoftEx_IRMA_count = 0
    HardEx_NV_count = 0
    HardEx_IRMA_count = 0
    NV_IRMA_count = 0
    target_folder = 'ConceptFoldersFGADR/SortedByCombinations'
    for i in range(overview_df.shape[0]):
        _img = overview_df.iloc[i,0]
        source_path = os.path.join(fgadr_folder,'Original_Images',_img)
        #If MA + HE:
        if (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE')
            shutil.copy(source_path,target_path)
            MA_HE_count +=1
        #If MA + SoftEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_SoftEx')
            shutil.copy(source_path,target_path)
            MA_SoftEx_count += 1
        #If MA + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HardEx')
            shutil.copy(source_path,target_path)
            MA_HardEx_count += 1
        #If MA + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            #No observations here....
            target_path = os.path.join(target_folder,'MA_NV')
            shutil.copy(source_path,target_path)
            MA_NV_count += 1
        #If MA + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_IRMA')
            shutil.copy(source_path,target_path)
            MA_IRMA_count +=1
        #If HE + SoftEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_SoftEx')
            shutil.copy(source_path,target_path)
            HE_SoftEx_count += 1
        #If HE + HardEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_HardEx')
            shutil.copy(source_path,target_path)
            HE_HardEx_count += 1
        #If HE + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_NV')
            shutil.copy(source_path,target_path)
            HE_NV_count +=1
        #If HE + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_IRMA')
            shutil.copy(source_path,target_path)
            HE_IRMA_count +=1
        #If SoftEx + HardEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'SoftEx_HardEx')
            shutil.copy(source_path,target_path)
            SoftEx_HardEx_count +=1
        #If SoftEx + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            #No observations here....
            target_path = os.path.join(target_folder,'SoftEx_NV')
            shutil.copy(source_path,target_path)
            SoftEx_NV_count +=1
        #If SoftEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            #No observations here....
            target_path = os.path.join(target_folder,'SoftEx_IRMA')
            shutil.copy(source_path,target_path)
            SoftEx_IRMA_count +=1
        #If HardEx + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HardEx_NV')
            shutil.copy(source_path,target_path)
            HardEx_NV_count +=1
        #If HardEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HardEx_IRMA')
            shutil.copy(source_path,target_path)
            HardEx_IRMA_count +=1
        #If NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            #No observations here....
            target_path = os.path.join(target_folder,'NV_IRMA')
            shutil.copy(source_path,target_path)
            NV_IRMA_count +=1
    
    print('MA + HE:',MA_HE_count)
    print('MA + SoftEx images:',MA_SoftEx_count)
    print('MA + HardEx images:', MA_HardEx_count)
    print('MA + NV images:',MA_NV_count)
    print('MA + IRMA images:',MA_IRMA_count)
    print('HE + SoftEx images:', HE_SoftEx_count)
    print('HE + HardEx images:',HE_HardEx_count)
    print('HE + NV images:',HE_NV_count)
    print('HE + IRMA images:',HE_IRMA_count)
    print('SoftEx + HardEx images:',SoftEx_HardEx_count)
    print('SoftEx + NV images:',SoftEx_NV_count)
    print('SoftEx + IRMA images:',SoftEx_IRMA_count)
    print('HardEx + NV images:',HardEx_NV_count)
    print('HardEx + IRMA images:',HardEx_IRMA_count)
    print('NV + IRMA images:',NV_IRMA_count)
    print('******************')
    print('Number of images in MA + HE concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE'))))
    print('Number of images in MA + SoftEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_SoftEx'))))
    print('Number of images in MA + HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HardEx'))))
    print('Number of images in MA + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_IRMA'))))
    print('Number of images in HE + SoftEx concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx'))))
    print('Number of images in no HE + HardEx folder:',len(os.listdir(os.path.join(target_folder,'HE_HardEx'))))
    print('Number of images in no HE + NV folder:',len(os.listdir(os.path.join(target_folder,'HE_NV'))))
    print('Number of images in no HE + IRMA folder:',len(os.listdir(os.path.join(target_folder,'HE_IRMA'))))
    print('Number of images in SoftEx + HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'SoftEx_HardEx'))))
    print('Number of images in HardEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'HardEx_NV'))))
    print('Number of images in HardEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'HardEx_IRMA'))))

#Sort the all the unique combinations of three concepts:
def pickTripleAbnormalities():
    MA_HE_SoftEx_count = 0
    MA_HE_HardEx_count = 0
    MA_HE_NV_count = 0
    MA_HE_IRMA_count = 0
    MA_SoftEx_HardEx_count = 0
    MA_SoftEx_NV_count = 0
    MA_SoftEx_IRMA_count = 0
    MA_HardEx_NV_count = 0
    MA_HardEx_IRMA_count = 0
    MA_NV_IRMA_count = 0
    HE_SoftEx_HardEx_count = 0
    HE_SoftEx_NV_count = 0
    HE_SoftEx_IRMA_count = 0
    HE_HardEx_NV_count = 0
    HE_HardEx_IRMA_count = 0
    HE_NV_IRMA_count = 0
    SoftEx_HardEx_NV_count = 0
    SoftEx_HardEx_IRMA_count = 0
    SoftEx_NV_IRMA_count = 0
    HardEx_NV_IRMA_count = 0
    target_folder = 'ConceptFoldersFGADR/SortedByCombinations'
    for i in range(overview_df.shape[0]):
        _img = overview_df.iloc[i,0]
        source_path = os.path.join(fgadr_folder,'Original_Images',_img)
        #If MA + HE + SoftEx:
        if (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_count +=1
        #If MA + HE + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_HardEx')
            shutil.copy(source_path,target_path)
            MA_HE_HardEx_count += 1
        #If MA + HE + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_NV')
            shutil.copy(source_path,target_path)
            MA_HE_NV_count += 1
        #If MA + HE + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_IRMA_count += 1
        #If MA + SoftEx + HardEx:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_SoftEx_HardEx')
            shutil.copy(source_path,target_path)
            MA_SoftEx_HardEx_count +=1
        #If MA + SoftEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_SoftEx_NV')
            shutil.copy(source_path,target_path)
            MA_SoftEx_NV_count += 1
        #If MA + SoftEx + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_SoftEx_IRMA_count += 1
        #If MA + HardEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HardEx_NV')
            shutil.copy(source_path,target_path)
            MA_HardEx_NV_count +=1
        #If MA + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_HardEx_IRMA_count +=1
        #If MA + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_NV_IRMA_count +=1
        #If HE + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_NV_IRMA')
            shutil.copy(source_path,target_path)
            HE_NV_IRMA_count +=1
        #If HE + SoftEx + HardEx:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_SoftEx_HardEx')
            shutil.copy(source_path,target_path)
            HE_SoftEx_HardEx_count +=1
        #If HE + SoftEx + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_SoftEx_NV')
            shutil.copy(source_path,target_path)
            HE_SoftEx_NV_count +=1
        #If HE + SoftEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx_IRMA')
            shutil.copy(source_path,target_path)
            HE_SoftEx_IRMA_count +=1
        #If HE + HardEx + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_HardEx_NV')
            shutil.copy(source_path,target_path)
            HE_HardEx_NV_count +=1
        #If HE + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            HE_HardEx_IRMA_count +=1
        #If SoftEx + HardEx + NV
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'SoftEx_HardEx_NV')
            shutil.copy(source_path,target_path)
            SoftEx_HardEx_NV_count +=1
        #If SoftEx + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'SoftEx_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            SoftEx_HardEx_IRMA_count +=1
        #If HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            HardEx_NV_IRMA_count +=1
        #If SoftEx + NV + IRMA
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'SoftEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            SoftEx_NV_IRMA_count +=1

    print('MA + HE + SoftEx:',MA_HE_SoftEx_count)
    print('MA + HE + HardEx:',MA_HE_HardEx_count)
    print('MA + HE + NV:',MA_HE_NV_count)
    print('MA + HE + IRMA:',MA_HE_IRMA_count)
    print('MA + SoftEx + HardEx images:',MA_SoftEx_HardEx_count)
    print('MA + SoftEx + NV images:',MA_SoftEx_NV_count)
    print('MA + SoftEx + IRMA images:',MA_SoftEx_IRMA_count)
    print('MA + HardEx + NV images:', MA_HardEx_NV_count)
    print('MA + HardEx + IRMA images:', MA_HardEx_IRMA_count)
    print('MA + NV + IRMA images:',MA_NV_IRMA_count)
    print('HE + NV + IRMA images:', HE_NV_IRMA_count)
    print('HE + SoftEx + HardEx images:', HE_SoftEx_HardEx_count)
    print('HE + SoftEx + NV images:', HE_SoftEx_NV_count)
    print('HE + SoftEx + IRMA images:', HE_SoftEx_IRMA_count)
    print('HE + HardEx + NV images:',HE_HardEx_NV_count)
    print('HE + HardEx + IRMA images:',HE_HardEx_IRMA_count)
    print('SoftEx + HardEx + NV images:',SoftEx_HardEx_NV_count)
    print('SoftEx + HardEx + IRMA images:',SoftEx_HardEx_IRMA_count)
    print('SoftEx + NV + IRMA images:',SoftEx_NV_IRMA_count)
    print('HardEx + NV + IRMA images:',HardEx_NV_IRMA_count)
    print('******************')
    print('Number of images in MA + HE + SoftEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx'))))
    print('Number of images in MA + HE + HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_HardEx'))))
    print('Number of images in MA + HE + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_IRMA'))))
    print('Number of images in MA + SoftEx+ HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_SoftEx_HardEx'))))
    print('Number of images in MA + HardEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HardEx_IRMA'))))
    print('Number of images in HE + SoftEx + HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx_HardEx'))))
    print('Number of images in HE + SoftEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx_NV'))))
    print('Number of images in HE + SoftEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx_IRMA'))))
    print('Number of images in no HE + HardEx + IRMA folder:',len(os.listdir(os.path.join(target_folder,'HE_HardEx_IRMA'))))

#Sort the all the unique combinations of four concepts:    
def pickFourAbnormalities():
    MA_HE_SoftEx_HardEx_count = 0
    MA_HE_SoftEx_NV_count = 0
    MA_HE_SoftEx_IRMA_count = 0
    MA_HE_HardEx_NV_count = 0
    MA_HE_HardEx_IRMA_count = 0
    MA_HE_NV_IRMA_count = 0
    MA_SoftEx_HardEx_NV_count = 0
    MA_SoftEx_HardEx_IRMA_count = 0
    MA_SoftEx_NV_IRMA_count = 0
    MA_HardEx_NV_IRMA_count = 0
    HE_SoftEx_HardEx_NV_count = 0
    HE_SoftEx_HardEx_IRMA_count = 0
    HE_SoftEx_NV_IRMA_count = 0
    HE_HardEx_NV_IRMA_count = 0
    SoftEx_HardEx_NV_IRMA_count = 0
    target_folder = 'ConceptFoldersFGADR/SortedByCombinations'
    for i in range(overview_df.shape[0]):
        _img = overview_df.iloc[i,0]
        source_path = os.path.join(fgadr_folder,'Original_Images',_img)
        #If MA + HE + SoftEx + HardEx:
        if (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_HardEx')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_count +=1
        #If MA + HE + SoftEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_NV')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_NV_count += 1
        #If MA + HE + SoftEx + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_IRMA_count += 1
        #If MA + HE + HardEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_HardEx_NV')
            shutil.copy(source_path,target_path)
            MA_HE_HardEx_NV_count += 1
        #If MA + HE + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_HardEx_IRMA_count +=1
        #If MA + HE + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_NV_IRMA_count += 1
        #If MA + SoftEx + HardEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_SoftEx_HardEx_NV')
            shutil.copy(source_path,target_path)
            MA_SoftEx_HardEx_NV_count += 1
        #If MA + SoftEx + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_SoftEx_HardEx_IRMA_count +=1
        #If MA + SoftEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_SoftEx_NV_IRMA_count +=1
        #If MA + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_HardEx_NV_IRMA_count +=1
        #If HE + SoftEx + HardEx + NV:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'HE_SoftEx_HardEx_NV')
            shutil.copy(source_path,target_path)
            HE_SoftEx_HardEx_NV_count +=1
        #If HE + SoftEx + HardEx + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            HE_SoftEx_HardEx_IRMA_count +=1
        #If HE + SoftEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            HE_SoftEx_NV_IRMA_count +=1
        #If HE + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            HE_HardEx_NV_IRMA_count +=1
        #If SoftEx + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'SoftEx_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            SoftEx_HardEx_NV_IRMA_count +=1
        
    print('MA + HE + SoftEx + HardEx:',MA_HE_SoftEx_HardEx_count)
    print('MA + HE + SoftEx + NV:',MA_HE_SoftEx_NV_count)
    print('MA + HE + SoftEx + IRMA:',MA_HE_SoftEx_IRMA_count)
    print('MA + HE + HardEx + NV:',MA_HE_HardEx_NV_count)
    print('MA + HE + HardEx + IRMA:',MA_HE_HardEx_IRMA_count)
    print('MA + HE + NV + IRMA:',MA_HE_NV_IRMA_count)
    print('MA + SoftEx + HardEx + NV images:',MA_SoftEx_HardEx_NV_count)
    print('MA + SoftEx + HardEx + IRMA images:',MA_SoftEx_HardEx_IRMA_count)
    print('MA + SoftEx + NV + IRMA images:',MA_SoftEx_NV_IRMA_count)
    print('MA + HardEx + NV + IRMA images:', MA_HardEx_NV_IRMA_count)
    print('HE + SoftEx + HardEx + NV images:', HE_SoftEx_HardEx_NV_count)
    print('HE + SoftEx + HardEx + IRMA images:', HE_SoftEx_HardEx_IRMA_count)
    print('HE + SoftEx + NV + IRMA images:', HE_SoftEx_NV_IRMA_count)
    print('HE + HardEx + NV + IRMA images:',HE_HardEx_NV_IRMA_count)
    print('SoftEx + HardEx + NV + IRMA images:',SoftEx_HardEx_NV_IRMA_count)

    print('******************')
    print('Number of images in MA + HE + SoftEx + HardEx concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_HardEx'))))
    print('Number of images in MA + HE + SoftEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_NV'))))
    print('Number of images in MA + HE + SoftEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_IRMA'))))
    print('Number of images in MA + HE + HardEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_HardEx_NV'))))
    print('Number of images in MA + HE + HardEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_HardEx_IRMA'))))
    print('Number of images in MA + SoftEx+ HardEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'MA_SoftEx_HardEx_NV'))))
    print('Number of images in MA + SoftEx+ NV + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_SoftEx_NV_IRMA'))))
    print('Number of images in MA + HardEx+ NV + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HardEx_NV_IRMA'))))
    print('Number of images in HE + SoftEx + HardEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx_HardEx_NV'))))
    print('Number of images in HE + SoftEx + HardEx + IRMA concept folder:',len(os.listdir(os.path.join(target_folder,'HE_SoftEx_HardEx_IRMA'))))

#Sort the all the combinations of five concepts + all six concepts together:
def pickFiveAllAbnormalities():
    all_count = 0
    MA_HE_SoftEx_HardEx_NV_count = 0
    MA_HE_SoftEx_HardEx_IRMA_count = 0
    MA_HE_SoftEx_NV_IRMA_count = 0
    MA_SoftEx_HardEx_NV_IRMA_count = 0
    MA_HE_HardEx_NV_IRMA_count = 0
    HE_SoftEx_HardEx_NV_IRMA_count = 0
    target_folder = 'ConceptFoldersFGADR/SortedByCombinations'
    for i in range(overview_df.shape[0]):
        _img = overview_df.iloc[i,0]
        source_path = os.path.join(fgadr_folder,'Original_Images',_img)
        #If all abnormalities:
        if (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            all_count +=1
        #If MA + HE + SoftEx + HardEx + NV:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==0):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_HardEx_NV')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_NV_count += 1
        #If MA + HE + SoftEx + HardEx + IRMA
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==0) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_HardEx_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_HardEx_IRMA_count += 1
        #If MA + HE + SoftEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==0) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_SoftEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_SoftEx_NV_IRMA_count += 1
        #If MA + SoftEx + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==0) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_SoftEx_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_SoftEx_HardEx_NV_IRMA_count +=1
        #If MA + HE + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==1) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==0) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'MA_HE_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            MA_HE_HardEx_NV_IRMA_count += 1
        #If HE + SoftEx + HardEx + NV + IRMA:
        elif (overview_df.iloc[i,1]==0) and (overview_df.iloc[i,2]==1) and (overview_df.iloc[i,3]==1) and (overview_df.iloc[i,4]==1) and (overview_df.iloc[i,5]==1) and (overview_df.iloc[i,6]==1):
            target_path = os.path.join(target_folder,'HE_SoftEx_HardEx_NV_IRMA')
            shutil.copy(source_path,target_path)
            HE_SoftEx_HardEx_NV_IRMA_count += 1
    print('All abnormalities:',all_count)
    print('MA + HE + SoftEx + HardEx + NV images:',MA_HE_SoftEx_HardEx_NV_count)
    print('MA + HE + SoftEx + HardEx + IRMA images:',MA_HE_SoftEx_HardEx_IRMA_count)
    print('MA + HE + SoftEx + NV + IRMA images:',MA_HE_SoftEx_NV_IRMA_count)
    print('MA + SoftEx + HardEx + NV + IRMA images:',MA_SoftEx_HardEx_NV_IRMA_count)
    print('MA + HE + HardEx + NV + IRMA images:',MA_HE_HardEx_NV_IRMA_count)
    print('HE + SoftEx + HardEx + NV + IRMA images:',HE_SoftEx_HardEx_NV_IRMA_count)

    print('******************')
    print('Number of images in ALL abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_HardEx_NV_IRMA'))))
    print('Number of images in MA + HE + SoftEx + HardEx + NV concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_HardEx_NV'))))
    print('Number of images in MA + HE + SoftEx + HardEx + IRMA abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_HardEx_IRMA'))))
    print('Number of images in MA + HE + SoftEx + NV + IRMA abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_SoftEx_NV_IRMA'))))
    print('Number of images in MA + SoftEx + HardEx +NV + IRMA abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'MA_SoftEx_HardEx_NV_IRMA'))))
    print('Number of images in MA + HE +  HardEx + NV + IRMA abnorm concept folder:',len(os.listdir(os.path.join(target_folder,'MA_HE_HardEx_NV_IRMA'))))
    

#Uncomment code below to
# 1. Create the overview DF of concept presence
#createOverviewDf()

# 2. Sort the images based on which combinations of concepts that are present:
# Split the functions into combinations consisting of one, two, three, four and five+all concepts:
#pickSingleAbnormalities()
#pickDoubleAbnormalities()
#pickTripleAbnormalities()
#pickFourAbnormalities()
#pickFiveAllAbnormalities()