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
"""


import pandas as pd
import os
from datetime import date
import matplotlib.pyplot as plt

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