import pandas as pd
import numpy as np

#Counting how many images where the different concepts are present (for numbers reported in Table 1)
#For the concept datasets: FGADR, DiaretDB1, DDR and IDRiD
#The overview DFs were created in BottleneckModels/createOverviewDF.py
#Except for FGADR, which were created in BottleneckModels/FGADR_training/createConceptDf.py


#Start with FGDAR:
fgadr_conceptOverview = pd.read_csv('../SequentialBottleneck_experiments/FGADR_Concept_DR_annotations.csv', index_col = 'Unnamed: 0')
print('Head of FGADR df:',fgadr_conceptOverview.head())
print(fgadr_conceptOverview.describe())
print(fgadr_conceptOverview.shape)

print('Counting number of MA images:')
print(fgadr_conceptOverview['Microaneurysms'].sum())
print('Counting number of hemorrhages images:')
print(fgadr_conceptOverview['Hemohedge'].sum())
print('Counting number of Soft exudates images:')
print(fgadr_conceptOverview['SoftExudate'].sum())
print('Counting number of Hard exudates images:')
print(fgadr_conceptOverview['HardExudate'].sum())
print('Counting number of IRMA images:')
print(fgadr_conceptOverview['IRMA'].sum())
print('Counting number of NV images:')
print(fgadr_conceptOverview['Neovascularization'].sum())

#Next, inspecting DDR:
ddr_conceptOverview = pd.read_csv('../SequentialBottleneck_experiments/DDR_conceptOverview.csv', index_col='Unnamed: 0')
print('Head of DDR:',ddr_conceptOverview.head())
print(ddr_conceptOverview.shape)
print(ddr_conceptOverview.describe())
normal_counter = 0
for i in range(ddr_conceptOverview.shape[0]):
    #Check if all concepts are missing:
    if (ddr_conceptOverview.iloc[i,1]==0) and (ddr_conceptOverview.iloc[i,2]==0) and (ddr_conceptOverview.iloc[i,3]==0) and (ddr_conceptOverview.iloc[i,4]==0):
        normal_counter += 1
print('Number of normal images in DDR:', normal_counter)
print('Number of images with concepts:',ddr_conceptOverview.shape[0]-normal_counter)
print('Counting MA:', ddr_conceptOverview['Microaneurysms'].sum())
print('Counting hemorrhages:', ddr_conceptOverview['Hemohedge'].sum())
print('Counting soft exudates:', ddr_conceptOverview['SoftExudate'].sum())
print('Counting hard exudates:', ddr_conceptOverview['HardExudate'].sum())

#DiaretDB
diaretDB1_overview = pd.read_csv('../SequentialBottleneck_experiments/DiaretDB1_conceptOverview.csv',index_col = 'Unnamed: 0')
print('Head of DiaretDB1:',diaretDB1_overview.head())
print(diaretDB1_overview.shape)
print(diaretDB1_overview.describe())
print('Number of MA:',diaretDB1_overview['Microaneurysms'].sum())
print('Number of hemorrhages:',diaretDB1_overview['Hemohedge'].sum())
print('Number of soft exudates:',diaretDB1_overview['SoftExudate'].sum())
print('Number of hard exudates:',diaretDB1_overview['HardExudate'].sum())

#IDRiD:
IDRiD_conceptOverview = pd.read_csv('../SequentialBottleneck_experiments/IDRiD_conceptOverview.csv',index_col='Unnamed: 0')
print('Head of IDRiD df:', IDRiD_conceptOverview.head())
print(IDRiD_conceptOverview.shape)
print(IDRiD_conceptOverview.describe())

normal_counter = 0
for i in range(IDRiD_conceptOverview.shape[0]):
    if (IDRiD_conceptOverview.iloc[i,1]==0) and (IDRiD_conceptOverview.iloc[i,2]==0) and (IDRiD_conceptOverview.iloc[i,3]==0) and (IDRiD_conceptOverview.iloc[i,4]==0):
        normal_counter+= 1
print('Number of images without concepts:',normal_counter)
print('Number of images with concepts:',IDRiD_conceptOverview.shape[0]-normal_counter)

print('Counting images with MA:',IDRiD_conceptOverview['Microaneurysms'].sum())
print('Counting images with hemorrhages:',IDRiD_conceptOverview['Hemohedge'].sum())
print('Counting images with soft exudates:',IDRiD_conceptOverview['SoftExudate'].sum())
print('Counting images with hard exudates:',IDRiD_conceptOverview['HardExudate'].sum())