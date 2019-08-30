'''
Filename: f:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\the_garden_nerd.py
Path: f:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data
Created Date: Friday, August 30th 2019, 6:34:17 pm
Author: apotdar
'''

"""[Flower Recognition - Deep Learning]
The application of deep learning is rapidly growing in the field of computer vision and is helping in building powerful classification and identification models. 
We can leverage this power of deep learning to build models that can classify and differentiate between different species of flower as well.

Total 102 categories of flowers:
    train - Contains 18540 images
    test  - Contains 2009 images

TODO:
# * Check no of images available per class
# * Decide upon model architecture to be applied.
# * 
"""

import pandas as pd

path_to_train_csv = r'F:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\data\train.csv'
path_to_test_csv = r'F:\Dataset\forFatNinja\hEarth_dQuest_flowerRecg\HE_Challenge_data\data\test.csv'

train_df = pd.read_csv(path_to_train_csv)
test_df = pd.read_csv(path_to_test_csv)

