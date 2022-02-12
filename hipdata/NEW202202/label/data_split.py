import json
import os
import re
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import math

import torch
import torchvision
import torchvision.transforms as transforms





df_labels = pd.read_csv('stage_2_labels_processed.csv')

df_splits = pd.read_csv('splits_with_migration_index.csv')

image_name = df_labels[['filename']]
image_name_split = df_splits[['filename']]
image_split = df_splits[['split']]



loader_train, loader_val, loader_test = pd.DataFrame({}),pd.DataFrame({}),pd.DataFrame({})
for index in range(len(image_name)):
    img_name = image_name.iloc[index].iat[0]
    for index2 in range(len(image_name_split)):
        img_name_split = image_name_split.iloc[index2].iat[0]
        if img_name == img_name_split:
            if image_split['split'].loc[index2] == 'train':
            #print(df_labels.loc[idx:idx])
                loader_train = loader_train.append(df_labels.loc[index:index])
            if image_split['split'].loc[index2] == 'validation':
            #print(df_labels.loc[idx:idx])
                loader_val = loader_val.append(df_labels.loc[index:index])
            if image_split['split'].loc[index2] == 'test':
            #print(df_labels.loc[idx:idx])
                loader_test = loader_test.append(df_labels.loc[index:index])

loader_train.to_csv('stage_2_labels_processed_train.csv')
loader_val.to_csv('stage_2_labels_processed_val.csv')
loader_test.to_csv('stage_2_labels_processed_test.csv')


