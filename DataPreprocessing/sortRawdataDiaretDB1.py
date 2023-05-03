import pandas as pd
import numpy as np
import shutil
import os
import random

data_path = 'Data/diaretdb1_v_1_1/resources/images'

#First, I want to get all images that are regarded as normal
# i.e., they do not have any of the abnormalities
image_list = os.listdir(os.path.join(data_path,'ddb1_fundusimages'))
print('Number of images in dataset:',len(image_list))

hardEx_path = os.path.join(data_path,'ddb1_groundtruth/hardexudates')
softEx_path = os.path.join(data_path,'ddb1_groundtruth/softexudates')
redDots_path = os.path.join(data_path,'ddb1_groundtruth/redsmalldots')
hemorrhage_path = os.path.join(data_path,'ddb1_groundtruth/hemorrhages')

#Since all images have segmentation masks, it is not possible to just filter based on
#whether the image is in the corresponding folder or not...
#Need to check that all pixels in the segmentation masks are black...
print('Number of images with hard exudates:', len(os.listdir(hardEx_path)))
print('Number of images with red dots:', len(os.listdir(redDots_path)))
print('Number of images with soft exudates:', len(os.listdir(softEx_path)))
print('Number of images with hemorrhages:', len(os.listdir(hemorrhage_path)))

normal_images = []
for _img in image_list:
    #If the image is not in any of the abnormal folders,
    #Then it can be classified as 'Normal'
    if _img not in os.listdir(hardEx_path):
        print('No hard exudates')
        if _img not in os.listdir(softEx_path):
            print('Also no soft exudates')
            if _img not in os.listdir(redDots_path):
                if _img not in os.listdir(hemorrhage_path):
                    normal_images.append(_img)

print('Number of normal images:',len(normal_images))
print(normal_images)
