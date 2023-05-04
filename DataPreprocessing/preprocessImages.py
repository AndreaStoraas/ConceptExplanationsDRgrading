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

#Selecting the dataset part to crop:
#Should repeat for training, validation and test sets of 
# APTOS 2019, Diabetic Retinopathy Detection, MESSIDOR2 and FGADR
training_path = '../Data/TrainDRDetection'
class0 = os.path.join(training_path,'0')
class1 = os.path.join(training_path,'1')
class2 = os.path.join(training_path,'2')
class3 = os.path.join(training_path,'3')
class4 = os.path.join(training_path,'4')

#Cropping code from this kaggle page for the APTOS 2019 dataset:
# https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy?scriptVersionId=20340219
def crop_fundus_kaggle(img, tol=7):
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
            img = np.stack([img1,img2,img3],axis=-1)
        return img

class_list = [class0, class1, class2, class3, class4]
target_path = '../Data/CroppedDataKaggle/CroppedTrainDRDetection'

print('Number of images in original train path:',len(os.listdir(training_path)))
print('Number of cropped images:',len(os.listdir(target_path)))

#For each class folder, loop through the images, crop out black areas and
# save the new images in a corresponding CroppedDataKaggle-folder:
for i in range(len(class_list)):
    print('Working on class',i)
    for _file in os.listdir(class_list[i]):
        target_filepath = os.path.join(target_path,str(i), _file)
        image_path = os.path.join(class_list[i],_file)
        img = cv.imread(image_path)
        croppedImage = crop_fundus_kaggle(img,tol=7)
        cv.imwrite(target_filepath,croppedImage)
