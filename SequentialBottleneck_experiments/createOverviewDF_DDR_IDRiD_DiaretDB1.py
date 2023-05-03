import os
import shutil
import numpy as np
import pandas as pd

#Create Overview DFs for DDR, IDRiD and DiaretDB1 in same manner as for FGADR
#Looking at the FGADR overview DF for inspiration:
FGADR_df = pd.read_csv('FGADR_training/FGADR_Concept_DR_annotations.csv',index_col = 'Unnamed: 0')
print('FGADR overview for inspiration:')
print(FGADR_df.head())
#DDR: 
def create_conceptOverview_DDR():
    prelim_overview = pd.read_csv('../TCAV_work/ConceptFoldersDDR/file_overviewAbnormalities.csv')
    print('Number of images in preliminary overview (only with the concepts present)',prelim_overview.shape[0])
    print(prelim_overview.head())
    #Create df for the normal images:
    noAbnormalities_images = os.listdir('../Data/CroppedDataKaggle/CroppedDDR_noAbnormalities')
    overview_noAbnormalities = pd.DataFrame(columns = ['mask_name', 'EX','HE','MA','SE'])
    for i in range(len(noAbnormalities_images)):
        _file = noAbnormalities_images[i]
        #Add only 0s since no abnormalities are present:
        overview_noAbnormalities.loc[i] = _file, 0,0,0,0
    print('Size of normal overview df:',overview_noAbnormalities.shape)
    #Change from mask name to image name (from tif to jpg)
    for j in range(prelim_overview.shape[0]):
        _image_name = prelim_overview.iloc[j,0][:-3]+'jpg'
        #print('Image name:',_image_name)
        prelim_overview.iloc[j,0] = _image_name
    print('Corrected the image names:')
    print(prelim_overview.head())
    #Combine the two DFs:
    ddr_conceptOverview = pd.concat([prelim_overview,overview_noAbnormalities],axis=0,ignore_index=True)
    ddr_conceptOverview=ddr_conceptOverview.rename(columns = {'mask_name':'image_name','EX':'HardExudate','HE':'Hemohedge','MA':'Microaneurysms','SE':'SoftExudate'})
    new_column_order = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate']
    ddr_conceptOverview = ddr_conceptOverview[new_column_order]
    print('Shape of new df:',ddr_conceptOverview.shape)
    print('Saving DDR overview DF to csv...')
    ddr_conceptOverview.to_csv('DDR_conceptOverview.csv')

def create_conceptOverview_IDRiD():
    idrid_conceptOverview = pd.DataFrame(columns = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate'])
    folders_path = '../TCAV_work/ConceptFoldersIDRiD/SortedByCombinations'
    concept_folders = os.listdir(folders_path)
    for _folder in concept_folders:
        folder_files = os.listdir(os.path.join(folders_path,_folder))
        for _file in folder_files:
            #Create a mini-df to be concatenated with the large overview df:
            miniDF = pd.DataFrame(columns = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate'])
            if _folder == 'MA_HardEx':
                miniDF.loc[0]=_file, 1, 0, 0, 1
            elif _folder =='MA_HE_HardEx':
                miniDF.loc[0]=_file, 1, 1, 0, 1
            elif _folder =='MA_HE_SoftEx_HardEx':
                miniDF.loc[0]=_file, 1, 1, 1, 1
            elif _folder == 'NoAbnormalities':
                miniDF.loc[0]=_file, 0, 0, 0, 0
            else:
                print('Folder is not recognized:',_folder)
            idrid_conceptOverview = pd.concat([idrid_conceptOverview,miniDF],axis = 0, ignore_index=True)
    print('Overview df for IDRiD dataset:')
    print(idrid_conceptOverview.shape)
    print(idrid_conceptOverview.head())
    print('Saving IDRiD overview DF to csv...')
    idrid_conceptOverview.to_csv('IDRiD_conceptOverview.csv')

def create_conceptOverview_DiaretDB1():
    diaretDB1_overview = pd.read_csv('../TCAV_work/ConceptFoldersDiaretDB/DiaretDB1_overviewAbnormalities.csv',index_col = 'Unnamed: 0')
    print('Shape of the df:',diaretDB1_overview.shape)
    print(diaretDB1_overview.head())
    #Need to change the column names:
    diaretDB1_overview = diaretDB1_overview.rename(columns = {'ImageName':'image_name','redsmalldots':'Microaneurysms','Hemorrhages':'Hemohedge','HardEx':'HardExudate','SoftEx':'SoftExudate'})
    #Change the order:
    new_column_order = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate']
    diaretDB1_overview = diaretDB1_overview[new_column_order]
    print('DiaretDB1 overview DF after correcting columns:')
    print(diaretDB1_overview.head())
    print('Saving DiaretDB1 overview df as csv...')
    diaretDB1_overview.to_csv('DiaretDB1_conceptOverview.csv')

def combine_overviewDFs():
    FGADR_overview = pd.read_csv('FGADR_training/FGADR_Concept_DR_annotations.csv',index_col = 'Unnamed: 0')
    #Drop the last 3 columnd (NV, IRMA and DR level)
    FGADR_overview = FGADR_overview.iloc[:,:-3]
    print('FGADR overview:')
    print(FGADR_overview.head())
    DDR_overview = pd.read_csv('DDR_conceptOverview.csv',index_col='Unnamed: 0')
    IDRiD_overview = pd.read_csv('IDRiD_conceptOverview.csv',index_col='Unnamed: 0')
    DiaretDB1_overview = pd.read_csv('DiaretDB1_conceptOverview.csv',index_col='Unnamed: 0')
    total_overview = pd.concat([FGADR_overview,DDR_overview,IDRiD_overview,DiaretDB1_overview],axis = 0,ignore_index=True)
    print('Beginning of total overview:')
    print(total_overview.head())
    print('Shape of total overview file:',total_overview.shape)
    print('Shape FGADR:',FGADR_overview.shape[0])
    print('Size of DDR:',DDR_overview.shape[0])
    print('Size of IDRiD:',IDRiD_overview.shape[0])
    print('Size of DiaretDB1:',DiaretDB1_overview.shape[0])
    print('Saving total overview DF as csv:')
    total_overview.to_csv('FGADR_DDR_IDRiD_DiaretDB1_conceptOverview.csv')

def create_overviewDF_CroppedTestCombinedXL():
    #Creating an overview of image name and DR level for compatibility with extracting predicted concepts 
    #(and further test the sequential models for DR level predictions on the full combined testset)
    combinedTestXL_Overview = pd.DataFrame(columns = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate','Neovascularization','IRMA','DR_level'])
    folders_path = '../Data/CroppedDataKaggle/CroppedTestCombinedXL'
    #Extract image name and corresponding class for each image in each DR level folder:
    class_folders = os.listdir(folders_path)
    for _class in range(len(class_folders)):
        class_path = os.path.join(folders_path,str(_class))
        for _image in os.listdir(class_path):
            #Create a mini-df to be concatenated with the large overview df:
            miniDF = pd.DataFrame(columns = ['image_name','Microaneurysms','Hemohedge','SoftExudate','HardExudate','Neovascularization','IRMA','DR_level'])
            #Add the image name and class name (the concepts in between are just lol since not known)
            miniDF.loc[0] = _image, 'lol','lol','lol','lol','lol','lol',_class
            #Merge the df's:
            combinedTestXL_Overview = pd.concat([combinedTestXL_Overview,miniDF],axis = 0,ignore_index=True)
    print('Size of overview DF:',combinedTestXL_Overview.shape)
    print('Beginning of the DF:',combinedTestXL_Overview.head())
    print('Saving combined XL overview DF as csv file...')
    combinedTestXL_Overview.to_csv('CroppedTestCombinedXL_overview.csv')

#create_conceptOverview_DDR()
#create_conceptOverview_IDRiD()
#create_conceptOverview_DiaretDB1()
#combine_overviewDFs()
#create_overviewDF_CroppedTestCombinedXL()
