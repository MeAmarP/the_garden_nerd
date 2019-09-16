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

import cv2
import os
from datetime import date
import matplotlib.pyplot as plt


def displayImageData(in_df,no_of_sample=9):
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

train_df['img_path'] = os.getcwd()+  '\\data\\train\\' + train_df.image_id[:] + '.jpg'

# Add Routie To display sample data
displayImageData(train_df)




