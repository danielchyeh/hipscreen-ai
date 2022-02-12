# train.py
#!/usr/bin/env	python3

""" stage 2 CPN key point detection eval """

import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, calc_mi, visualization_highq, \
     gen_outputcsv_file, calc_rotation_diff

from CPN import CoordRegressionNetwork
from dsnt import dsntnn


import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from sklearn.metrics import mean_absolute_error



def dataset_generate_hip(_IMAGE_DIR, label_file, mode, model_img, rot_diff, visual_mode, visual_path, Resize):
    image_name = label_file[['filename']]
    MP_right, MP_left = label_file[['mi_right']].fillna(np.nan), label_file[['mi_left']].fillna(np.nan)

    keypoint_right = [label_file[['med_head_right']], label_file[['lat_head_right']], label_file[['lat_acetabulum_right']]]
    keypoint_left = [label_file[['med_head_left']], label_file[['lat_head_left']], label_file[['lat_acetabulum_left']]]
    dataset = []
    model_mp_right, model_mp_left = [], []

    if visual_mode:
        v_path = os.path.join(visual_path,'base_img_{}'.format(mode))
        if not os.path.exists(v_path):
            os.makedirs(v_path)


    for i in range(len(model_img)):
        component = []

        for idx in range(len(image_name)):
            q = model_img[i]
            k = image_name.iloc[idx].iat[0]
            if q == k:
                key_right_list = []
                for key in keypoint_right:
                    key_right_list.append(float(key.iloc[idx].iat[0].split(', ')[0].split('(')[1]))
                    key_right_list.append(float(key.iloc[idx].iat[0].split(', ')[1].split(')')[0]))
                
                key_left_list = []
                for key in keypoint_left:
                    key_left_list.append(float(key.iloc[idx].iat[0].split(', ')[0].split('(')[1]))
                    key_left_list.append(float(key.iloc[idx].iat[0].split(', ')[1].split(')')[0]))


                gt_mp_right = float(MP_right.iloc[idx].iat[0])
                gt_mp_left = float(MP_left.iloc[idx].iat[0])

        image_path = os.path.join(_IMAGE_DIR, model_img[i])
        print('loading: {}'.format(image_path))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        #Transform the GT upright image in stage2 to model upright image (stage1)
        train_transform = A.Compose([
            A.Affine(rotate=rot_diff[i],p=1.0)
            ], keypoint_params=A.KeypointParams(format='xy'))

        keypoints = [
                (key_right_list[0], key_right_list[1]),(key_right_list[2], key_right_list[3]),
                (key_right_list[4], key_right_list[5]),(key_left_list[0], key_left_list[1]),
                (key_left_list[2], key_left_list[3]),(key_left_list[4], key_left_list[5]),
            ]

        transformed = train_transform(image=image, keypoints=keypoints)
        img_trans = transformed['image']
        keyps_trans = transformed['keypoints']
        
        #Data augmentation for the model upright image (stage1)
        scale = 1
        rc_size = int(min(image.shape[0:2])*scale)

        train_transform_2 = A.Compose([
            A.CenterCrop(width=rc_size, height=rc_size),
            A.Resize(Resize, Resize),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy'))
        
        keypoints_2 = [
            (keyps_trans[0][0], keyps_trans[0][1]),(keyps_trans[1][0], keyps_trans[1][1]),
            (keyps_trans[2][0], keyps_trans[2][1]),(keyps_trans[3][0], keyps_trans[3][1]),
            (keyps_trans[4][0], keyps_trans[4][1]),(keyps_trans[5][0], keyps_trans[5][1]),
        ]
        transformed = train_transform_2(image=img_trans, keypoints=keypoints_2)
        img_trans_2 = transformed['image']
        keyps_trans_2 = transformed['keypoints']

        keys = []
        for k in range(6):
            keys.append([(keyps_trans_2[k][0]*2+1)/Resize-1,(keyps_trans_2[k][1]*2+1)/Resize-1])
        keys = torch.Tensor(keys)

        mp_right = calc_mi(keyps_trans_2[0][0],keyps_trans_2[1][0],keyps_trans_2[2][0],'right')
        mp_left = calc_mi(keyps_trans_2[3][0],keyps_trans_2[4][0],keyps_trans_2[5][0],'left')
        model_mp_right.append(mp_right)
        model_mp_left.append(mp_left)

        print(mp_right, mp_left)

        # #save the model upright image (stage1) for visualization
        # scale = 1
        # rc_size = int(min(image.shape[0:2])*scale)
        # train_transform_3 = A.Compose([
        #     A.CenterCrop(width=rc_size, height=rc_size),
        #     A.Resize(Resize, Resize),
        #     ])
        # transformed = train_transform_3(image=img_trans)
        # img_trans_3 = transformed['image']
        # cv2.imwrite(os.path.join(v_path, str(model_img[i])), img_trans_3)

        # #save the model upright image (stage1) for visualization
        if visual_mode:
            cv2.imwrite(os.path.join(v_path, str(model_img[i])), img_trans)


        component.append(img_trans_2)# image
        component.append(keys)
        component.append(gt_mp_right)
        component.append(gt_mp_left)
        component.append(model_img[i])
        component = tuple(component)

        dataset.append(component)

    return dataset




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='CPN', help='net type')
    parser.add_argument('-BASE_DIR', type=str, default='../data', help='access data/label')
    parser.add_argument('-LABEL_DIR', type=str, default='stage_2_labels_processed', help='access label file')
    parser.add_argument('-GT_STAGE1_LABEL', type=str, default='stage_1_labels_processed', help='access label file')
    parser.add_argument('-MODEL_STAGE1_PRED', type=str, default='avgR_MP_update_data', help='access label file')
    parser.add_argument('-MODE', type=str, default='test', choices=['val', 'test'], help='evaluate on validation or test set')
    parser.add_argument('-VISUAL_MODE', action='store_true', help='ON / OFF visualization')
    parser.add_argument('-VISUAL_PATH', type=str, default='visualization', help='create visualization results on images')
    parser.add_argument('-CSVFILE_MODE', action='store_true', help='ON / OFF generate csv file')

    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    ###hipdata function
    _IMAGE_DIR = os.path.join(args.BASE_DIR, 'stage_2')
    label_file = pd.read_csv(os.path.join(args.BASE_DIR, 'label', args.LABEL_DIR+'_{}.csv'.format(args.MODE)))#the baseline

    gt_stage1_label = os.path.join(args.BASE_DIR, 'label', args.GT_STAGE1_LABEL+'_{}.csv'.format(args.MODE))
    model_stage1_pred = os.path.join(args.BASE_DIR, 'label', args.MODEL_STAGE1_PRED+'_{}.csv'.format(args.MODE))

    model_image, rotation_diff = calc_rotation_diff(gt_stage1_label, model_stage1_pred)


    testset_hip = dataset_generate_hip(_IMAGE_DIR, label_file, args.MODE, model_image, \
         rotation_diff, args.VISUAL_MODE, args.VISUAL_PATH, Resize=224)
    testloader_hip = torch.utils.data.DataLoader(testset_hip, batch_size=1, shuffle=False, drop_last=True, num_workers=16) 
    ndata = testset_hip.__len__()
    print('number of {} hip data for evaluation: {}'.format(args.MODE, ndata)) 

    net = CoordRegressionNetwork().cuda()

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    csv_data = [] 
    csv_file = []
    total_mp_right_mae = 0
    total_mp_left_mae = 0
    resize = 224

    with torch.no_grad():
        for n_iter, (images, keyps, gt_mp_right, gt_mp_left, img_name) in enumerate(testloader_hip):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader_hip)))

            coords, heatmaps = net(images.cuda())

            mp_right_label = calc_mi(((keyps[0][0][0].item()+1)*resize-1)/2,\
                ((keyps[0][1][0].item()+1)*resize-1)/2,((keyps[0][2][0].item()+1)*resize-1)/2,'right')
            mp_left_label = calc_mi(((keyps[0][3][0].item()+1)*resize-1)/2,\
                ((keyps[0][4][0].item()+1)*resize-1)/2,((keyps[0][5][0].item()+1)*resize-1)/2,'left')

            mp_right_pred = calc_mi(((coords[0][0][0].item()+1)*resize-1)/2, \
                ((coords[0][1][0].item()+1)*resize-1)/2,((coords[0][2][0].item()+1)*resize-1)/2,'right')
            mp_left_pred = calc_mi(((coords[0][3][0].item()+1)*resize-1)/2, \
                ((coords[0][4][0].item()+1)*resize-1)/2,((coords[0][5][0].item()+1)*resize-1)/2,'left')


            mp_right_mae = mean_absolute_error([mp_right_pred], [mp_right_label])
            mp_left_mae = mean_absolute_error([mp_left_pred], [mp_left_label])

            print('Image:{} | MP RIGHT MAE:{} | MP LEFT MAE:{}'.format(img_name[0],mp_right_mae,mp_left_mae))

            #Visualization of upright image with key points
            #visualization(args.VISUAL_PATH, args.MODE, img_name, keyps, coords, resize=224, num_key=6)
            
            if args.VISUAL_MODE:
                mp_legend = [round(mp_right_label,2)*100,round(mp_left_label,2)*100, \
                    round(mp_right_pred,2)*100,round(mp_left_pred,2)*100]

                visualization_highq(args.VISUAL_PATH, args.MODE, img_name, \
                    keyps, coords, mp_legend, resize=224, num_key=6)


            total_mp_right_mae = total_mp_right_mae + mp_right_mae
            total_mp_left_mae = total_mp_left_mae + mp_left_mae

            csv_file.append([img_name[0],round(gt_mp_right[0].item(),2),round(gt_mp_left[0].item(),2), \
                round(mp_right_pred,2),round(mp_left_pred,2)])


        print('AVG MP RIGHT MAE: {}'.format(total_mp_right_mae/len(testloader_hip)))
        print('AVG MP LEFT MAE: {}'.format(total_mp_left_mae/len(testloader_hip)))

        if args.CSVFILE_MODE:
            gen_outputcsv_file(args.VISUAL_PATH, args.MODE, csv_file)



