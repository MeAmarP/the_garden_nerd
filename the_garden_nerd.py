'''
Filename: f:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\the_garden_nerd.py
Path: f:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data
Created Date: Friday, August 30th 2019, 6:34:17 pm
Author: apotdar
'''

"""[Flower Recognition - Deep Learning]
The application of deep learning is rapidly growing in the field of computer vision
and is helping in building powerful classification and identification models.
We can leverage this power of deep learning to build models that can classify
and differentiate between different species of flower as well.

Total 102 categories of flowers:
    train - Contains 18540 images
    test  - Contains 2009 images

TODO:
# * Add Column in dframe for img_path
# * Check no of images available per class
# * Decide upon model architecture to be applied.
# * Refer <https://www.kaggle.com/carlolepelaars/efficientnetb5-with-keras-aptos-2019>
"""

import pandas as pd
import numpy as np
import cv2
import os
from datetime import date
import matplotlib.pyplot as plt

#For CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, ZeroPadding3D
from tensorflow.keras.layers import (Conv2D, MaxPooling3D,MaxPooling2D)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 456 # Cause EfficientNetB5 - (456, 456, 3)
def displayImageData(in_df,no_of_sample=9):
    """Disaplay Sample Image data
    
    Arguments:
        in_df {pandas dataframe} -- Input train Dataframe with Img_Path Column
    
    Keyword Arguments:
        no_of_sample {int} -- Number of Image samples to display (default: {9})
    """
    fig = plt.figure("EDA Image Data")
    fig.subplots_adjust(hspace=0.7, wspace=0.1)
    fig.suptitle('Understanding Image Data', fontsize=16)

    for n, path in enumerate(in_df.img_path[:no_of_sample]):
        label = str(in_df.category[n])
        ImgTitle = in_df.image_id[n] +'_'+label
        Img = cv2.imread(path)
        #Note: Adjust the subplot grid so nb_of_images fits pefect ina grid.
        ax = fig.add_subplot(3,3,(n+1))
        plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        ax.set_title(ImgTitle)
        plt.axis('off')
    plt.show()
    return

def get_preds_and_labels(model, generator):
    """Get predictions and labels from the generator
    
    Arguments:
        model {[type]} -- [description]
        generator {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    BATCH_SIZE = 4
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()

def preprocess_img(src_img_path,img_size):
    """[summary]
    
    Arguments:
        img_path {string} -- path to input image
        img_size {[type]} -- image resize value
    """
    MainImg = cv2.imread(src_img_path)
    MainImg_out = cv2.resize(MainImg,(img_size,img_size), interpolation=cv2.INTER_LANCZOS4)
    MainImg_out = cv2.cvtColor(MainImg_out,cv2.COLOR_BGR2RGB)
    return MainImg_out

path_to_train_csv = r'the_garden_nerd\train.csv'
path_to_test_csv = r'the_garden_nerd\test.csv'

train_dir_path = r'F:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\data\train'
test_dir_path = r'F:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\data\test'

train_df = pd.read_csv(path_to_train_csv)
test_df = pd.read_csv(path_to_test_csv)


print(len(os.listdir(train_dir_path)))
# print(train_df.head())
# print(test_df.head())

# get no of images per category
flowerClasses = train_df.category.unique()
print("perClass_nb_of_imgs:", train_df.category.value_counts())

# EDA: Lets find out the class distribtion 
train_df.category.value_counts().sort_index().plot(kind="bar",figsize=(20,10),rot=0)
plt.title("Flower Class Distribution")
plt.xticks(rotation='vertical')
plt.xlabel("Flower Class")
plt.ylabel("Freq")
plt.savefig('EDA_class_distribution.png')


train_df.image_id = train_df.image_id.astype(str)

#Add Img_path Column to access images in Dataframe
train_df['img_path'] = os.getcwd()+  '\\data\\train\\' + train_df.image_id[:] + '.jpg'

# Add Routie To display sample data
displayImageData(train_df)



