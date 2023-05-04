import pandas as pd
import numpy as np
import shutil
import os
import random

from sklearn.model_selection import GroupShuffleSplit

#############Code for sorting the MESSIDOR-2 dataset################

image_folder = '../Data/MESSIDOR2/IMAGES'
paired_images = pd.read_csv('../Data/MESSIDOR2/messidor-2.csv', sep=';')
#The annotations are from Google Brain's Kaggle page:
#https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades?select=messidorData.csv
image_annotations = pd.read_csv('../Data/MESSIDOR2/messidor_data.csv')

print('Number of images in dataset:',len(os.listdir(image_folder)))
print('Number of image pairs, i.e. patients',paired_images.shape)

#First:
#Create a df with image_name and patient ID
#Each patient have two images (one from left eye and one from right eye)
#First go through the left-eye column and create patient ID = row number
overviewDf = pd.DataFrame(columns = ['Id','ImageName','Eye'])
for i in range(paired_images.shape[0]):
    image_nameLeft = paired_images.iloc[i,0]
    overviewDf.loc[i] = i, image_nameLeft, 'left'

#Also go through the right-eye column and use the same patient ID as row-number for the paired Df
#-> Same patient ID for left and right eye
for i in range(paired_images.shape[0]):
    patient_id = i
    row_number = i + paired_images.shape[0]
    image_nameRight = paired_images.iloc[i,-1]
    overviewDf.loc[row_number] = i, image_nameRight, 'right'
print('Number of observations in overviewDF:',overviewDf.shape[0])

#Next, we want to find the images that are annotated on Kaggle:
annotated_images = []
for j in range(image_annotations.shape[0]):
    if image_annotations.iloc[j,-1]==1:
        if '.jpg' in image_annotations.iloc[j,0]:
            new_name = image_annotations.iloc[j,0]
            new_name = new_name.replace('jpg','JPG')
            annotated_images.append(new_name)
        else:
            annotated_images.append(image_annotations.iloc[j,0])
        
print('Number of annotated images:',len(annotated_images))
not_annotated = []
for i in range(overviewDf.shape[0]):
    if overviewDf.iloc[i,1] not in annotated_images:
        drop_image = overviewDf.iloc[i,1]
        not_annotated.append(drop_image)
        
print('Number of not annotated images:',len(not_annotated))
#Remove the not annotated images from the overview DF:
for _img in not_annotated:
    overviewDf = overviewDf[overviewDf['ImageName']!=_img]
print('New number of observations in our overviewDf:',overviewDf.shape[0])

#Join the overview Df and the DR annotations from the annotated
#images:
overviewDf['DR-grade']='Lol'
for j in range(overviewDf.shape[0]):
    img_name = overviewDf.iloc[j,1]
    #Our image names have capital letters for .JPG, while 
    #the annotation df has .jpg:
    if '.JPG' in img_name:
        img_name = img_name.replace('JPG','jpg')
    #Find corresponding row in the annotation DF:
    corresponding_row = image_annotations[image_annotations['image_id']==img_name]
    grading = corresponding_row['adjudicated_dr_grade'].values[0]
    #Set the grading as the new value for the DR-grade column:
    overviewDf.iloc[j,-1]=grading

print('Distribution of the DR grades:')
print(overviewDf['DR-grade'].value_counts())

#Finally, we can split into train, validation and test sets
#Grouped by patient ID
print('Number of unique patient IDs:',overviewDf['Id'].nunique())

def sort_images():
    #Take the DF and copy images depending on the DR class
    #Also want to indicate whether it is from left or right eye:
    for i in range(overviewDf.shape[0]):
        DR_class = int(overviewDf.iloc[i,-1])
        eye_side = overviewDf.iloc[i,-2]
        image_name = overviewDf.iloc[i,1]
        #print(eye_side)
        #print(image_name)
        new_name, ending = image_name.split('.')
        new_name = new_name + '_' + eye_side
        new_name = new_name + '.' + ending
        #print(new_name)
        source_path = os.path.join(image_folder,image_name)
        target_path = os.path.join('../Data/MESSIDOR2/SortedMESSIDOR2',str(DR_class),new_name)
        shutil.copy(source_path, target_path)

    #Check the distributions of the sorted folders
    for j in range(5):
        class_folder = os.path.join('../Data/MESSIDOR2/SortedMESSIDOR2',str(j))
        class_list = os.listdir(class_folder)
        print('Looking at class',j)
        print('Number of observations:',len(class_list))

def split_trainValTest():
    #Find correct number for train, valid and test:
    num_train = int(overviewDf.shape[0]*0.8)
    num_valid = int(overviewDf.shape[0]*0.1)
    #The rest is used for testing
    num_test = overviewDf.shape[0] - (num_train + num_valid)

    print('Number of training samples:',str(num_train))
    print('Number of valid samples:',str(num_valid))
    print('Number of test samples:',str(num_test))
    print('In total:', str(num_train+num_valid+num_test))
    print('Should match with number of samples in dataset:',overviewDf.shape[0])

    #Use GroupShuffleSplit to split the data
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
    #Define the groups:
    my_groups = overviewDf['Id']
    #Divide test number by 2 since 2 observations/eyes per ID:
    groupss =GroupShuffleSplit(n_splits=1, random_state=42, test_size=int(num_test/2), train_size=None)
    #First, get the test and (train+valid) sets:
    for i, (trainValid_index, test_index) in enumerate(groupss.split(overviewDf['Id'], groups=my_groups)):
        trainValidDf = overviewDf.iloc[trainValid_index,:]
        testDf = overviewDf.iloc[test_index,:]

    smaller_groups = trainValidDf['Id']
    #Divide valid number by 2 since 2 observations/eyes per ID:
    smallerGroupss =GroupShuffleSplit(n_splits=1, random_state=42, test_size=int(num_valid/2), train_size=None)

    for i, (train_index, valid_index) in enumerate(smallerGroupss.split(trainValidDf['Id'], groups=smaller_groups)):
        trainDf = trainValidDf.iloc[train_index,:]
        validDf = trainValidDf.iloc[valid_index,:]

    print(trainDf.shape)
    print(validDf.shape)
    print(testDf.shape)
    print('Total samples in the three datasets:',
    trainDf.shape[0]+validDf.shape[0]+testDf.shape[0])
    
    #Check that same id is not in train, test or validation:
    for i in range(trainDf.shape[0]):
        if trainDf.iloc[i,0] in validDf['Id'].tolist():
            print('Same patient in train and validation set!!! Something went wrong in the splitting process!')
        elif trainDf.iloc[i,0] in testDf['Id'].tolist():
            print('Same patient in train and testing set!!!')

    for j in range(validDf.shape[0]):
        if validDf.iloc[j,0] in testDf['Id'].tolist():
            print('Same patient in validation and test set! Something went wrong!')

    #Next, loop through the df's and copy over to train, valid and test folders:
    for i in range(trainDf.shape[0]):
        #Get the image name:
        DR_class = int(trainDf.iloc[i,-1])
        eye_side = trainDf.iloc[i,-2]
        image_name = trainDf.iloc[i,1]
        #print(eye_side)
        #print(image_name)
        new_name, ending = image_name.split('.')
        new_name = new_name + '_' + eye_side
        new_name = new_name + '.' + ending
        #print(new_name)
        source_path = os.path.join(image_folder,image_name)
        target_path = os.path.join('../Data/TrainMESSIDOR',str(DR_class),new_name)
        shutil.copy(source_path, target_path)

    #Repeat for the validation and test sets:
    for i in range(validDf.shape[0]):
        #Get the image name:
        DR_class = int(validDf.iloc[i,-1])
        eye_side = validDf.iloc[i,-2]
        image_name = validDf.iloc[i,1]
        new_name, ending = image_name.split('.')
        new_name = new_name + '_' + eye_side
        new_name = new_name + '.' + ending
        source_path = os.path.join(image_folder,image_name)
        target_path = os.path.join('../Data/ValidMESSIDOR',str(DR_class),new_name)
        shutil.copy(source_path, target_path)

    for i in range(testDf.shape[0]):
        #Get the image name:
        DR_class = int(testDf.iloc[i,-1])
        eye_side = testDf.iloc[i,-2]
        image_name = testDf.iloc[i,1]
        new_name, ending = image_name.split('.')
        new_name = new_name + '_' + eye_side
        new_name = new_name + '.' + ending
        source_path = os.path.join(image_folder,image_name)
        target_path = os.path.join('../Data/TestMESSIDOR',str(DR_class),new_name)
        shutil.copy(source_path, target_path)

#Uncomment below to
# 1. Sort the raw dataset:
#sort_images()

# 2. Split into training, validation and test sets (80%, 10%, 10%):
#split_trainValTest()

#Check how many images in each folder:
#Start with training folder:
print('Class distribution in training set:')
for i in range(5):
    classFolder = os.path.join('../Data/TrainMESSIDOR',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))
#Then look at validation and test sets...
print('Class distribution in validation set:')
for i in range(5):
    classFolder = os.path.join('../Data/ValidMESSIDOR',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))

print('Class distribution in test set:')
for i in range(5):
    classFolder = os.path.join('../Data/TestMESSIDOR',str(i))
    classFiles = os.listdir(classFolder)
    print('Looking at class:',str(i))
    print('Number of images:',len(classFiles))

