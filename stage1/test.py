# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, level_stage, rotation_pairs, xypoint_norm, cal_ang

###hipdata
import json
import os
import re

import pandas as pd
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import csv

from sklearn.metrics import mean_absolute_error

 

def dataset_generate_hip(_IMAGE_DIR, label_file, level_labels, rotation_base=20):
    dataset = []

    image_name = label_file[['filename']]
    #input level label and output its corresponding left and right point
    level_left, level_right = level_stage(level_labels, label_file)


    for i in range(len(image_name)):

        all_rotations = rotation_pairs(i, level_left, level_right)
        avg_rot = np.mean(all_rotations)

        component = []

        image_path = os.path.join(_IMAGE_DIR, image_name.iloc[i].iat[0])
        filenames = image_name.iloc[i].filename

        print('loading:{}'.format(image_path))
        #get the image
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        img_center_x, img_center_y = float(w/2), float(h/2)

        line = 100
        shift_y = line * math.tan(avg_rot*math.pi/180)
        
        right_x_p,right_y_p  = img_center_x - line,img_center_y - shift_y
        left_x_p,left_y_p = img_center_x + line,img_center_y + shift_y

        scale = 1
        rc_size = int(min(image.shape[0:2])*scale)
        train_transform = A.Compose([
            A.CenterCrop(width=rc_size, height=rc_size),
            A.Resize(224, 224),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy'))

        keypoints = [
            (right_x_p, right_y_p),
            (left_x_p, left_y_p),
        ]
        transformed = train_transform(image=image, keypoints=keypoints)
        img_trans = transformed['image']
        keyps_trans = transformed['keypoints']

        right_x_nor, right_y_nor = xypoint_norm(keyps_trans[0][0],keyps_trans[0][1],224)
        left_x_nor, left_y_nor = xypoint_norm(keyps_trans[1][0],keyps_trans[1][1],224)

        center_x, center_y = (left_x_nor+right_x_nor)/2, (left_y_nor+right_y_nor)/2
        #calculate the normalized rotation angle
        center_right_x, center_right_y = center_x, center_y+0.1
        rot_normal = cal_ang((right_x_nor,right_y_nor),(center_x,center_y), \
            (center_right_x,center_right_y),rotation_base)


        component.append(img_trans)# image
        component.append(center_x)
        component.append(center_y)
        component.append(rot_normal)
        component.append(filenames)
        component = tuple(component)

        dataset.append(component)
    

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-BASE_DIR', type=str, default='/home/ucbdandan38/work/hipscreen_repo/hipdata/NEW202202', help='access data/label')
    parser.add_argument('-LABEL_DIR', type=str, default='stage_1_labels_processed', help='access label file')
    parser.add_argument('-OUTPUT_PRED', type=str, default='avgR_MP_update_data', help='path to output prediction file')
    parser.add_argument('-MODE', type=str, default='test', choices=['val', 'test'], help='evaluate on validation or test set')
    parser.add_argument('-level_labels', nargs="+", default=['triradiate','tuberosity','crest','obturator'], help='type of level label for avg rot')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-batch', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    ###hipdata function
    _IMAGE_DIR = os.path.join(args.BASE_DIR, 'stage_1')
    label_file = pd.read_csv(os.path.join(args.BASE_DIR, 'label', args.LABEL_DIR+'_{}.csv'.format(args.MODE)))#the baseline

    testset_hip = dataset_generate_hip(_IMAGE_DIR, label_file, args.level_labels)
    testloader_hip = torch.utils.data.DataLoader(testset_hip, batch_size=args.batch, shuffle=False,drop_last=True, num_workers=16) 
    ndata = testset_hip.__len__()
    print('number of {} hip data: {}'.format(args.MODE,ndata)) 


    net = get_network(args)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    total_rot_mae = 0
    csv_data = []
    if not os.path.exists(os.path.join(args.BASE_DIR,'label')):
        os.mkdir(os.path.join(args.BASE_DIR,'label'))
        df = pd.DataFrame(list())
        df.to_csv(os.path.join(args.BASE_DIR,'label',args.OUTPUT_PRED'_{}.csv'.format(args.MODE)))

    with torch.no_grad():
        for n_iter, (images,label_x,label_y,label_rot, path) in enumerate(testloader_hip):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader_hip)))

            if args.gpu:
                images = images.cuda()
                label_rot = label_rot.cuda()

            output = net(images)

            rotation_pred = output[0][2].cpu().detach().numpy()
            rotation_label = label_rot[0].cpu().detach().numpy()
            print('rotation pred:{} | label:{}'.format(rotation_pred,rotation_label))

            single_rot_mae = mean_absolute_error([rotation_pred], [rotation_label])

            total_rot_mae = total_rot_mae + single_rot_mae
            csv_data.append([path[0],rotation_pred*20])  

        print('avg rotation MAE: {}'.format(total_rot_mae/len(testloader_hip)))


        with open(os.path.join(args.BASE_DIR,'label',args.OUTPUT_PRED+'_{}.csv'.format(args.MODE)), 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(['file_name', 'pred_rotation'])
            csvwriter.writerows(csv_data)  


