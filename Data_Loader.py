import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from os import walk
from os.path import join
from PIL import Image

import cv2,csv
import scipy.io as scio
#import Image
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def load_data(path):
    #train type :
    image_path = path+"/image"
    label_path = path+"/label"
    """Load data from `path`"""
    for root, dirs, files in walk(image_path):
        FILE_SIZE = len(files)

    images = np.zeros((FILE_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,3))
    # label_gender  : 0->Male
    #                 1->Female 
    #
    # label age     : 0->Child
    #                 1->Teen
    #                 2->Adult
    #                 3->Middle
    #                 4->Senior
    label_gender = np.zeros((FILE_SIZE),dtype=int)
    label_age = np.zeros((FILE_SIZE),dtype=int)
    file_count = 0
    for root, dirs, files in walk(image_path):
        for f in files:

            fullpath_image = root+'/'+ f
            fullpath_label = (root[0:-5]+'label/'+f[0:-4]+'.txt')
            #print(fullpath_image)
            #print(fullpath_label)
            image_temp = cv2.imread(fullpath_image)
            #cv2.imshow('image'+str(file_count), image)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            images[file_count]= cv2.resize(image_temp,(IMAGE_HEIGHT,IMAGE_WIDTH),interpolation=cv2.INTER_CUBIC)
            #print(files_name[file_count])

            f=open(fullpath_label,'r')
            lines=f.readline()
            
            gender = lines.split()[2]
            #print(gender)
            if gender == 'GenderMale':
                label_gender[file_count] = 0
            elif gender == 'GenderFemale':
                label_gender[file_count] = 1

            lines=f.readline()
            
            age = lines.split()[2]
            #print(age)
            if age == 'AgeChild':
                label_age[file_count] = 0
            #elif age == 'AgeTeen':
            #    label_age[file_count] = 1
            elif age == 'AgeAdult':
                label_age[file_count] = 1
            #elif age == 'AgeMiddle':
            #    label_age[file_count] = 3
            elif age == 'AgeSenior':
                label_age[file_count] = 2
            
            #print(label_gender[file_count])
            #print(label_age[file_count])
            file_count +=1

    return images,label_gender, label_age