import os
import random
import argparse
import torch
import copy

import numpy as np
from torch import functional
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

#Inspect the dimensions of the images:
training_path = 'TCAV_work/ConceptFoldersDDR/SortedByCombinations/NoAbnormalities'
class0 = os.path.join(training_path,'0')
class1 = os.path.join(training_path,'1')
class2 = os.path.join(training_path,'2')
class3 = os.path.join(training_path,'3')
class4 = os.path.join(training_path,'4')

#1 get the height and width of image: https://note.nkmk.me/en/python-pillow-image-resize/
#2 crop the image to equal width as height
#3 Place into new training folder (to check what the new images actually look like) or keep as a part of the training script...
#4 pass the new images into the neural network. NB! Check what the resize preprocessing is doing!


#Detect the circle in the image and crop around this: 
#From Zoi's code:
def crop_fundus(image_path, save_path):
    img = cv.imread(image_path,1)
    #print('Width of image:',img.shape)
    cimg = cv.medianBlur(img,5)
    cimg = cv.cvtColor(cimg,cv.COLOR_BGR2GRAY)
    #Tips to setting params:
    # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    #circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1.5,300,
                        #param1=50,param2=30,minRadius=400,maxRadius=800)
    circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1.5,100,
                        param1=200,param2=0.9,minRadius=100,maxRadius=2600)
    #Since not all images have circles:
    # https://stackoverflow.com/questions/65043866/error-when-performing-hough-circle-transformation-on-opencv-python
    if circles is not None:
        #print('Found a circle!')
        circles = np.uint16(np.around(circles))
        top = circles[0,0,1] - circles[0,0,2]
        left = circles[0,0,0] - circles[0,0,2]
        img = Image.open(image_path)
        #print('Width of image:',img.width)
        #print('Height of image:',img.height)
        cropped_img = transforms.functional.crop(img, top, left, 2*circles[0,0,2], 2*circles[0,0,2])
        #Check that the cropped image is NOT completely black!
        pixels = cropped_img.getdata()
        #If the cropped image is completely black, we keep the original image
        if (len(np.unique(pixels))==1) and (np.unique(pixels)[0]==0):
            print('Not saved since black')
            img.save(save_path)
        else:
            #print('Cropped image saved')
            print(save_path)
            cropped_img.save(save_path)
    else:
        img = Image.open(image_path)
        img.save(save_path)

#Cropping code from this kaggle page for the APTOS 2019 dataset:
# https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy?scriptVersionId=20340219
def crop_fundus_kaggle(img, tol=7):
    #Write in code from Kaggle here:
    if img.ndim ==2:
        print('Only 2 dimensions in image...')
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            print('Image too dark, which gives black cropping')
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

class_list = [class0, class1, class2, class3, class4]
class_list = [class0, class1, class2]

target_path = 'Data/CroppedDataKaggle/CroppedDDR_noAbnormalities'


for _file in os.listdir(training_path):
    #print('Image file:', _file)
    target_filepath = os.path.join(target_path, _file)
    #print(target_filepath)
    image_path = os.path.join(training_path,_file)
    img = cv.imread(image_path)
    croppedImage = crop_fundus_kaggle(img,tol=7)
    #print(croppedImage)
    #cv.imwrite(target_filepath,croppedImage)
print('Number of images in original train path:',len(os.listdir(training_path)))
print('Number of cropped images:',len(os.listdir(target_path)))
'''
for i in range(len(class_list)):
    print('Working on class',i)
    for _file in os.listdir(class_list[i]):
        #print('Image file:', _file)
        target_filepath = os.path.join(target_path,str(i), _file)
        #print(target_filepath)
        image_path = os.path.join(class_list[i],_file)
        img = cv.imread(image_path)
        croppedImage = crop_fundus_kaggle(img,tol=7)
        #print(croppedImage)
        #cv.imwrite(target_filepath,croppedImage)
'''
'''
for i in range(len(class_list)):
    print('Working on class',i)
    for _file in os.listdir(class_list[i]):
        #print('Image file:', _file)
        target_filepath = os.path.join(target_path,str(i), _file)
        image_path = os.path.join(class_list[i],_file)
        crop_fundus(image_path,target_filepath)


for _class in class_list:
    black_counter = 0
    crop_counter = 0
    print('Working on class 0')
    for _file in os.listdir(_class)[:20]:
        #print('Image file:', _file)
        target_filepath = os.path.join(target_path, _file)
        image_path = os.path.join(_class,_file)
        crop_fundus(image_path,target_filepath, black_counter,crop_counter)
'''
#print('Number of images in original class 4 DRDetection test:',len(os.listdir(class4)))
#print('Number of images in cropped class 4 DRDetection test:',len(os.listdir(os.path.join(target_path,'4'))))

#for i in range(len(class_list)):
#    print('Number of files in class',i)
#    print(len(os.listdir(class_list[i])))