import numpy as np
from random import randint

import cv2,csv
import scipy.io as scio
def data_augmentation(images,rate):
    #over sample one to five
    #crop 256*256 to 224*224 four from corners and one from center

    IMAGE_SIZE = 256
    CROP_SIZE = 224
    images = np.reshape(images,[-1,IMAGE_SIZE,IMAGE_SIZE,3])
    MARGIN_SIZE = (IMAGE_SIZE-CROP_SIZE)//2
    permutation_list=np.random.permutation(images.shape[0])
    images_ag = np.zeros((images.shape[0],CROP_SIZE,CROP_SIZE,3))
    augmentation_num = round(images.shape[0]*rate)

    for i in range(0,images.shape[0]):
        if i < augmentation_num:
            ag_class = randint(0,5)
            if ag_class == 0 :
                #center crop
                images_ag[permutation_list[i]] = images[permutation_list[i], MARGIN_SIZE:MARGIN_SIZE+CROP_SIZE , MARGIN_SIZE:MARGIN_SIZE+CROP_SIZE]
            elif ag_class == 1 :
                #top left crop
                images_ag[permutation_list[i]] = images[permutation_list[i], 0:CROP_SIZE , 0:CROP_SIZE]
            elif ag_class == 2 :
                #top right crop
                images_ag[permutation_list[i]] = images[permutation_list[i], 0:CROP_SIZE , 2*MARGIN_SIZE:]
            elif ag_class == 3 :
                #bot left crop
                images_ag[permutation_list[i]] = images[permutation_list[i], 2*MARGIN_SIZE: , 0:CROP_SIZE]
            elif ag_class == 4 :
                #bot right crop
                images_ag[permutation_list[i]] = images[permutation_list[i], 2*MARGIN_SIZE: , 2*MARGIN_SIZE:]
            elif ag_class == 5 :
                image_temp =cv2.resize(images[permutation_list[i]],(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_CUBIC )
                images_ag[permutation_list[i]] = np.flip(image_temp,2)
            else :
                images_ag[permutation_list[i]] = cv2.resize(images[permutation_list[i]],(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_CUBIC )

        else :        
            images_ag[permutation_list[i]] = cv2.resize(images[permutation_list[i]],(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_CUBIC )


    return images_ag