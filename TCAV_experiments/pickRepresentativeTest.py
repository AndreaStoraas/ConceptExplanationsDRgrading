import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import random

#This code picks 50 representative test images for each of the DR classes 0 to 4
# Because of memory limitations when calculating the TCAV scores
# The representative test set should include Kaggle cropped images from all four sub-parts of the test set:
#DR Detection, APTOS, Messidor and FGADR
random.seed(42)
representative_TestFolderPath = './RepresentativeTestFolderKaggleCropped'
#Starts with class 4 (smallest class):
target_folder_class4 = os.path.join(representative_TestFolderPath,'4')
class_4_allPath = '../Data/CroppedDataKaggle/CroppedTestCombinedXL/4'
class_4_DRDetection = '../Data/CroppedDataKaggle/CroppedTestDRDetection/4'
class_4_APTOS = '../Data/CroppedDataKaggle/CroppedTestAPTOS/4'
class_4_Messidor = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR/4'
class_4_FGADR = '../Data/CroppedDataKaggle/CroppedTestFGADR/4'
print('Total number of class 4 images:',len(os.listdir(class_4_allPath)))
print('Relative amount class for images DRDetection:',(len(os.listdir(class_4_DRDetection))/(len(os.listdir(class_4_allPath)))))
print('Relative amount class for images APTOS:',(len(os.listdir(class_4_APTOS))/(len(os.listdir(class_4_allPath)))))
print('Relative amount class for images Messidor:',(len(os.listdir(class_4_Messidor))/(len(os.listdir(class_4_allPath)))))
print('Relative amount class for images FGADR:',(len(os.listdir(class_4_FGADR))/(len(os.listdir(class_4_allPath)))))
fraction_DRDetection4 = len(os.listdir(class_4_DRDetection))/(len(os.listdir(class_4_allPath)))
fraction_APTOS4 = len(os.listdir(class_4_APTOS))/(len(os.listdir(class_4_allPath)))
fraction_Messidor4 = len(os.listdir(class_4_Messidor))/(len(os.listdir(class_4_allPath)))
fraction_FGADR4 = len(os.listdir(class_4_FGADR))/(len(os.listdir(class_4_allPath)))
num_images_DRDetection_class4 = int(np.round(50*fraction_DRDetection4,0))
num_images_APTOS_class4 = int(np.round(50*fraction_APTOS4,0))
num_images_Messidor_class4 = int(np.round(50*fraction_Messidor4,0))
num_images_FGADR_class4 = int(np.round(50*fraction_FGADR4,0))

deviation = (num_images_DRDetection_class4+num_images_APTOS_class4+num_images_FGADR_class4+num_images_Messidor_class4)-50
#If we have picked too many images (deviation >0), remove images from largest dataset:
if deviation>0:
    num_images_DRDetection_class4= num_images_DRDetection_class4 - deviation
#If we have picked too few images (deviation<0), add images to smallest dataset:
elif deviation<0:
    num_images_Messidor_class4 = num_images_Messidor_class4 - deviation
print('There are this number of excessive images:',deviation)
print('Num images DRDetection:',num_images_DRDetection_class4)
print('Num images APTOS:',num_images_APTOS_class4)
print('Num images Messidor:',num_images_Messidor_class4)
print('Num images FGADR:',num_images_FGADR_class4)
#Copy images from the class 4 folder of the test sets to the representative test folder
#Select images from each of the smaller datasets:
image_list_DRDetection4 = os.listdir(class_4_DRDetection)
selected_images_DRDetection4 = random.sample(image_list_DRDetection4,num_images_DRDetection_class4)
print('Number of selected images class 4 DRDetection:',len(selected_images_DRDetection4))
image_list_APTOS4 = os.listdir(class_4_APTOS)
selected_images_APTOS4 = random.sample(image_list_APTOS4,num_images_APTOS_class4)
image_list_Messidor4 = os.listdir(class_4_Messidor)
selected_images_Messidor4 = random.sample(image_list_Messidor4,num_images_Messidor_class4)
image_list_FGADR4 = os.listdir(class_4_FGADR)
selected_images_FGADR4 = random.sample(image_list_FGADR4,num_images_FGADR_class4)
#Copy images over to representative test set class 4:
for _image in selected_images_DRDetection4:
    source_path = os.path.join(class_4_DRDetection,_image)
    target_path = os.path.join(target_folder_class4,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_APTOS4:
    source_path = os.path.join(class_4_APTOS,_image)
    target_path = os.path.join(target_folder_class4,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_Messidor4:
    source_path = os.path.join(class_4_Messidor,_image)
    target_path = os.path.join(target_folder_class4,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_FGADR4:
    source_path = os.path.join(class_4_FGADR,_image)
    target_path = os.path.join(target_folder_class4,_image)
    shutil.copy(source_path,target_path)
print('Number of images in class for representative test folder:',len(os.listdir(target_folder_class4)))
#Repeat for class 3, 2, 1 and 0
#Class 3
target_folder_class3 = os.path.join(representative_TestFolderPath,'3')
class_3_allPath = '../Data/CroppedDataKaggle/CroppedTestCombinedXL/3'
class_3_DRDetection = '../Data/CroppedDataKaggle/CroppedTestDRDetection/3'
class_3_APTOS = '../Data/CroppedDataKaggle/CroppedTestAPTOS/3'
class_3_Messidor = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR/3'
class_3_FGADR = '../Data/CroppedDataKaggle/CroppedTestFGADR/3'

fraction_DRDetection3 = len(os.listdir(class_3_DRDetection))/(len(os.listdir(class_3_allPath)))
fraction_APTOS3 = len(os.listdir(class_3_APTOS))/(len(os.listdir(class_3_allPath)))
fraction_Messidor3 = len(os.listdir(class_3_Messidor))/(len(os.listdir(class_3_allPath)))
fraction_FGADR3 = len(os.listdir(class_3_FGADR))/(len(os.listdir(class_3_allPath)))
print('Looking at class 3')
num_images_DRDetection_class3 = int(np.round(50*fraction_DRDetection3,0))
num_images_APTOS_class3 = int(np.round(50*fraction_APTOS3,0))
num_images_Messidor_class3 = int(np.round(50*fraction_Messidor3,0))
num_images_FGADR_class3 = int(np.round(50*fraction_FGADR3,0))
deviation = num_images_DRDetection_class3+num_images_APTOS_class3+num_images_Messidor_class3+num_images_FGADR_class3 - 50
print('For class 3, we have this number of excessive images:',deviation)
#If deviation is positive, we have picked too many umages and will remove these from the largest datset:
if deviation>0:
    num_images_DRDetection_class3 = num_images_DRDetection_class3-deviation
#If negative, we have picked too few images and will add one image to the smallest dataset:
if deviation<0:
    num_images_Messidor_class3 = num_images_Messidor_class3-deviation
print('Number of images for DRDetection:',num_images_DRDetection_class3)
print('Number of images for APTOS:',num_images_APTOS_class3)
print('Number of images for Messidor:',num_images_Messidor_class3)
print('Number of images for FGADR:',num_images_FGADR_class3)

#Copy images from the class 3 folder of the test sets to the representative test folder
#Select images from each of the smaller datasets:
image_list_DRDetection3 = os.listdir(class_3_DRDetection)
selected_images_DRDetection3 = random.sample(image_list_DRDetection3,num_images_DRDetection_class3)
print('Number of selected images class 3 DRDetection:',len(selected_images_DRDetection3))
image_list_APTOS3 = os.listdir(class_3_APTOS)
selected_images_APTOS3 = random.sample(image_list_APTOS3,num_images_APTOS_class3)
image_list_Messidor3 = os.listdir(class_3_Messidor)
selected_images_Messidor3 = random.sample(image_list_Messidor3,num_images_Messidor_class3)
image_list_FGADR3 = os.listdir(class_3_FGADR)
selected_images_FGADR3 = random.sample(image_list_FGADR3,num_images_FGADR_class3)
#Copy images over to representative test set class 3:
for _image in selected_images_DRDetection3:
    source_path = os.path.join(class_3_DRDetection,_image)
    target_path = os.path.join(target_folder_class3,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_APTOS3:
    source_path = os.path.join(class_3_APTOS,_image)
    target_path = os.path.join(target_folder_class3,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_Messidor3:
    source_path = os.path.join(class_3_Messidor,_image)
    target_path = os.path.join(target_folder_class3,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_FGADR3:
    source_path = os.path.join(class_3_FGADR,_image)
    target_path = os.path.join(target_folder_class3,_image)
    shutil.copy(source_path,target_path)
print('Number of class 3 images in representative folder:',len(os.listdir(target_folder_class3)))
#Class 2:
target_folder_class2 = os.path.join(representative_TestFolderPath,'2')
class_2_allPath = '../Data/CroppedDataKaggle/CroppedTestCombinedXL/2'
class_2_DRDetection = '../Data/CroppedDataKaggle/CroppedTestDRDetection/2'
class_2_APTOS = '../Data/CroppedDataKaggle/CroppedTestAPTOS/2'
class_2_Messidor = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR/2'
class_2_FGADR = '../Data/CroppedDataKaggle/CroppedTestFGADR/2'

fraction_DRDetection2 = len(os.listdir(class_2_DRDetection))/(len(os.listdir(class_2_allPath)))
fraction_APTOS2 = len(os.listdir(class_2_APTOS))/(len(os.listdir(class_2_allPath)))
fraction_Messidor2 = len(os.listdir(class_2_Messidor))/(len(os.listdir(class_2_allPath)))
fraction_FGADR2 = len(os.listdir(class_2_FGADR))/(len(os.listdir(class_2_allPath)))
print('Looking at class 2')
num_images_DRDetection_class2 = int(np.round(50*fraction_DRDetection2,0))
num_images_APTOS_class2 = int(np.round(50*fraction_APTOS2,0))
num_images_Messidor_class2 = int(np.round(50*fraction_Messidor2,0))
num_images_FGADR_class2 = int(np.round(50*fraction_FGADR2,0))
deviation = num_images_DRDetection_class2+num_images_APTOS_class2+num_images_Messidor_class2+num_images_FGADR_class2 - 50
print('For class 2, we have this number of excessive images:',deviation)
#Since 0 deviation, we don't adjust anything for now...
print('Number of images for DRDetection:',num_images_DRDetection_class2)
print('Number of images for APTOS:',num_images_APTOS_class2)
print('Number of images for Messidor:',num_images_Messidor_class2)
print('Number of images for FGADR:',num_images_FGADR_class2)

#Copy images from the class 2 folder of the test sets to the representative test folder
image_list_DRDetection2 = os.listdir(class_2_DRDetection)
selected_images_DRDetection2 = random.sample(image_list_DRDetection2,num_images_DRDetection_class2)
print('Number of selected images class 2 DRDetection:',len(selected_images_DRDetection2))
image_list_APTOS2 = os.listdir(class_2_APTOS)
selected_images_APTOS2 = random.sample(image_list_APTOS2,num_images_APTOS_class2)
image_list_Messidor2 = os.listdir(class_2_Messidor)
selected_images_Messidor2 = random.sample(image_list_Messidor2,num_images_Messidor_class2)
image_list_FGADR2 = os.listdir(class_2_FGADR)
selected_images_FGADR2 = random.sample(image_list_FGADR2,num_images_FGADR_class2)
#Copy images over to representative test set class 2:
for _image in selected_images_DRDetection2:
    source_path = os.path.join(class_2_DRDetection,_image)
    target_path = os.path.join(target_folder_class2,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_APTOS2:
    source_path = os.path.join(class_2_APTOS,_image)
    target_path = os.path.join(target_folder_class2,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_Messidor2:
    source_path = os.path.join(class_2_Messidor,_image)
    target_path = os.path.join(target_folder_class2,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_FGADR2:
    source_path = os.path.join(class_2_FGADR,_image)
    target_path = os.path.join(target_folder_class2,_image)
    shutil.copy(source_path,target_path)
print('Number of class 2 images in representative test folder:',len(os.listdir(target_folder_class2)))
#Class 1:
target_folder_class1 = os.path.join(representative_TestFolderPath,'1')
class_1_allPath = '../Data/CroppedDataKaggle/CroppedTestCombinedXL/1'
class_1_DRDetection = '../Data/CroppedDataKaggle/CroppedTestDRDetection/1'
class_1_APTOS = '../Data/CroppedDataKaggle/CroppedTestAPTOS/1'
class_1_Messidor = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR/1'
class_1_FGADR = '../Data/CroppedDataKaggle/CroppedTestFGADR/1'

fraction_DRDetection1 = len(os.listdir(class_1_DRDetection))/(len(os.listdir(class_1_allPath)))
fraction_APTOS1 = len(os.listdir(class_1_APTOS))/(len(os.listdir(class_1_allPath)))
fraction_Messidor1 = len(os.listdir(class_1_Messidor))/(len(os.listdir(class_1_allPath)))
fraction_FGADR1 = len(os.listdir(class_1_FGADR))/(len(os.listdir(class_1_allPath)))
print('Looking at class 1')
num_images_DRDetection_class1 = int(np.round(50*fraction_DRDetection1,0))
num_images_APTOS_class1 = int(np.round(50*fraction_APTOS1,0))
num_images_Messidor_class1 = int(np.round(50*fraction_Messidor1,0))
num_images_FGADR_class1 = int(np.round(50*fraction_FGADR1,0))
deviation = num_images_DRDetection_class1+num_images_APTOS_class1+num_images_Messidor_class1+num_images_FGADR_class1 - 50
print('For class 1, we have this number of excessive images:',deviation)
#Since no deviations, we keep the numbers as they are
print('Number of images for DRDetection:',num_images_DRDetection_class1)
print('Number of images for APTOS:',num_images_APTOS_class1)
print('Number of images for Messidor:',num_images_Messidor_class1)
print('Number of images for FGADR:',num_images_FGADR_class1)

#Copy images from the class 1 folder of the test sets to the representative test folder
image_list_DRDetection1 = os.listdir(class_1_DRDetection)
selected_images_DRDetection1 = random.sample(image_list_DRDetection1,num_images_DRDetection_class1)
print('Number of selected images class 1 DRDetection:',len(selected_images_DRDetection1))
image_list_APTOS1 = os.listdir(class_1_APTOS)
selected_images_APTOS1 = random.sample(image_list_APTOS1,num_images_APTOS_class1)
image_list_Messidor1 = os.listdir(class_1_Messidor)
selected_images_Messidor1 = random.sample(image_list_Messidor1,num_images_Messidor_class1)
image_list_FGADR1 = os.listdir(class_1_FGADR)
selected_images_FGADR1 = random.sample(image_list_FGADR1,num_images_FGADR_class1)
#Copy images over to representative test set class 1:
for _image in selected_images_DRDetection1:
    source_path = os.path.join(class_1_DRDetection,_image)
    target_path = os.path.join(target_folder_class1,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_APTOS1:
    source_path = os.path.join(class_1_APTOS,_image)
    target_path = os.path.join(target_folder_class1,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_Messidor1:
    source_path = os.path.join(class_1_Messidor,_image)
    target_path = os.path.join(target_folder_class1,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_FGADR1:
    source_path = os.path.join(class_1_FGADR,_image)
    target_path = os.path.join(target_folder_class1,_image)
    shutil.copy(source_path,target_path)
print('Number of class 1 images in representative test folder:',len(os.listdir(target_folder_class1)))
#For class 0:
target_folder_class0 = os.path.join(representative_TestFolderPath,'0')
class_0_allPath = '../Data/CroppedDataKaggle/CroppedTestCombinedXL/0'
class_0_DRDetection = '../Data/CroppedDataKaggle/CroppedTestDRDetection/0'
class_0_APTOS = '../Data/CroppedDataKaggle/CroppedTestAPTOS/0'
class_0_Messidor = '../Data/CroppedDataKaggle/CroppedTestMESSIDOR/0'
class_0_FGADR = '../Data/CroppedDataKaggle/CroppedTestFGADR/0'

fraction_DRDetection0 = len(os.listdir(class_0_DRDetection))/(len(os.listdir(class_0_allPath)))
fraction_APTOS0 = len(os.listdir(class_0_APTOS))/(len(os.listdir(class_0_allPath)))
fraction_Messidor0 = len(os.listdir(class_0_Messidor))/(len(os.listdir(class_0_allPath)))
fraction_FGADR0 = len(os.listdir(class_0_FGADR))/(len(os.listdir(class_0_allPath)))
print('Looking at class 0')
num_images_DRDetection_class0 = int(np.round(50*fraction_DRDetection0,0))
num_images_APTOS_class0 = int(np.round(50*fraction_APTOS0,0))
num_images_Messidor_class0 = int(np.round(50*fraction_Messidor0,0))
num_images_FGADR_class0 = int(np.round(50*fraction_FGADR0,0))
deviation = num_images_DRDetection_class0+num_images_APTOS_class0+num_images_Messidor_class0+num_images_FGADR_class0 - 50
print('For class 0, we have this number of excessive images:',deviation)
#Since deviation = 0, we keep the numbers as they are for now...
#Since no images from FGADR, we take remove one image from DRDetection and give it to the FGADR dataset
num_images_DRDetection_class0 -= 1
num_images_FGADR_class0 += 1
print('Number of images for DRDetection:',num_images_DRDetection_class0)
print('Number of images for APTOS:',num_images_APTOS_class0)
print('Number of images for Messidor:',num_images_Messidor_class0)
print('Number of images for FGADR:',num_images_FGADR_class0)
#Copy images from the class 0 folder of the test sets to the representative test folder
image_list_DRDetection0 = os.listdir(class_0_DRDetection)
selected_images_DRDetection0 = random.sample(image_list_DRDetection0,num_images_DRDetection_class0)
print('Number of selected images class 0 DRDetection:',len(selected_images_DRDetection0))
image_list_APTOS0 = os.listdir(class_0_APTOS)
selected_images_APTOS0 = random.sample(image_list_APTOS0,num_images_APTOS_class0)
image_list_Messidor0 = os.listdir(class_0_Messidor)
selected_images_Messidor0 = random.sample(image_list_Messidor0,num_images_Messidor_class0)
image_list_FGADR0 = os.listdir(class_0_FGADR)
selected_images_FGADR0 = random.sample(image_list_FGADR0,num_images_FGADR_class0)
#Copy images over to representative test set class 0:
for _image in selected_images_DRDetection0:
    source_path = os.path.join(class_0_DRDetection,_image)
    target_path = os.path.join(target_folder_class0,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_APTOS0:
    source_path = os.path.join(class_0_APTOS,_image)
    target_path = os.path.join(target_folder_class0,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_Messidor0:
    source_path = os.path.join(class_0_Messidor,_image)
    target_path = os.path.join(target_folder_class0,_image)
    shutil.copy(source_path,target_path)
for _image in selected_images_FGADR0:
    source_path = os.path.join(class_0_FGADR,_image)
    target_path = os.path.join(target_folder_class0,_image)
    shutil.copy(source_path,target_path)
