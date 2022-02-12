# train.py
#!/usr/bin/env	python3

""" train key point detection network using pytorch"""
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

from conf import settings
from utils import get_network, WarmUpLR, most_recent_folder, \
     most_recent_weights, last_epoch, best_acc_weights, calc_mi
from CPN import CoordRegressionNetwork
from dsnt import dsntnn

###hipdata
import re
import pandas as pd
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def dataset_generate_hip(_IMAGE_DIR, label_file, Resize, num_keyps=6):
    image_name = label_file[['filename']]
    MP_right, MP_left = label_file[['mi_right']].fillna(np.nan), label_file[['mi_left']].fillna(np.nan)

    keypoint_right = [label_file[['med_head_right']], label_file[['lat_head_right']], label_file[['lat_acetabulum_right']]]
    keypoint_left = [label_file[['med_head_left']], label_file[['lat_head_left']], label_file[['lat_acetabulum_left']]]

    dataset = []

    for i in range(len(image_name)):
        if str(MP_right.iloc[i].iat[0]) != 'nan' and str(MP_left.iloc[i].iat[0]) != 'nan': 
            component = []

            image_path = os.path.join(_IMAGE_DIR, image_name.iloc[i].iat[0])
            print('loading: {}'.format(image_path))
            image = cv2.imread(image_path)
            h, w, _ = image.shape

            key_right_list = []
            for key in keypoint_right:
                key_right_list.append(float(key.iloc[i].iat[0].split(', ')[0].split('(')[1]))
                key_right_list.append(float(key.iloc[i].iat[0].split(', ')[1].split(')')[0]))
            
            key_left_list = []
            for key in keypoint_left:
                key_left_list.append(float(key.iloc[i].iat[0].split(', ')[0].split('(')[1]))
                key_left_list.append(float(key.iloc[i].iat[0].split(', ')[1].split(')')[0]))

            scale = 1
            rc_size = int(min(image.shape[0:2])*scale)
            train_transform = A.Compose([
                A.RandomCrop(width=rc_size, height=rc_size),
                A.Resize(Resize, Resize),
                A.Affine(translate_percent=(-0.01,0.01),p=1.0),
                A.Rotate(limit=5),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
                ], keypoint_params=A.KeypointParams(format='xy'))

            keypoints = [
                (key_right_list[0], key_right_list[1]),(key_right_list[2], key_right_list[3]),
                (key_right_list[4], key_right_list[5]),(key_left_list[0], key_left_list[1]),
                (key_left_list[2], key_left_list[3]),(key_left_list[4], key_left_list[5]),
            ]
            transformed = train_transform(image=image, keypoints=keypoints)
            img_trans = transformed['image']
            keyps_trans = transformed['keypoints']

            if len(keyps_trans) == num_keyps:
                keys = []
                for i in range(num_keyps):
                    keys.append([(keyps_trans[i][0]*2+1)/Resize-1,(keyps_trans[i][1]*2+1)/Resize-1])
                keys = torch.Tensor(keys)

                mp_right = calc_mi(keyps_trans[0][0],keyps_trans[1][0],keyps_trans[2][0],'right')
                mp_left = calc_mi(keyps_trans[3][0],keyps_trans[4][0],keyps_trans[5][0],'left')
                print('GT mp right:{} | mp left:{}'.format(mp_right, mp_left))
                

                component.append(img_trans)# image
                component.append(keys)
                component = tuple(component)

                dataset.append(component)
            else:
                print('fail to include due to lack of cropped keyps:{}'.format(image_name.iloc[i].iat[0]))
    

    return dataset



def train(epoch):

    start = time.time()
    net.train()

    cpn_model = nn.DataParallel(net)

    for batch_index, (images, keyps) in enumerate(trainloader_hip):
        if args.gpu:
            images = images.cuda()
            keyps.cuda()

        optimizer.zero_grad()
        #CPN model
        coords, heatmaps = cpn_model(images)
        #DSNT model
        euc_losses = dsntnn.euclidean_losses(coords, keyps.cuda())
        # Per-location regularization losses
        reg_losses = dsntnn.js_reg_losses(heatmaps, keyps.cuda(), sigma_t=1.0)
        # Combine losses into an overall loss
        loss = dsntnn.average_loss(euc_losses + reg_losses)

        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(trainloader_hip) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch + len(images),
            total_samples=len(trainloader_hip.dataset)
        ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    return loss.item(), optimizer.param_groups[0]['lr']

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='CPN', help='net type')
    parser.add_argument('-BASE_DIR', type=str, default='../data', help='access data/label')
    parser.add_argument('-LABEL_DIR', type=str, default='stage_2_labels_processed_train.csv', help='access label file')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-batch', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.03, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()


    ###hipdata function
    _IMAGE_DIR = os.path.join(args.BASE_DIR, 'stage_2')
    label_file = pd.read_csv(os.path.join(args.BASE_DIR, 'label', args.LABEL_DIR))#the baseline

    trainset_hip = dataset_generate_hip(_IMAGE_DIR, label_file, Resize=224)
    trainloader_hip = torch.utils.data.DataLoader(trainset_hip, batch_size=args.batch, shuffle=True, num_workers=16) 
    ndata = trainset_hip.__len__()
    print('number of training data of hip for stage2 model: {}'.format(ndata)) 

    net = CoordRegressionNetwork().cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(trainset_hip)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'stage2', args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'stage2', args.net, settings.TIME_NOW)


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        loss, lr = train(epoch)

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

