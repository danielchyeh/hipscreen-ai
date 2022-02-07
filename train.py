# train.py
#!/usr/bin/env	python3
""" train rotation predictor network using pytorch"""

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
from utils import get_network, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
        level_stage, rotation_pairs, xypoint_norm, cal_ang


###hipdata settings####
import matplotlib.pyplot as plt
import pandas as pd
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np



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
            A.RandomCrop(width=rc_size, height=rc_size),
            A.Resize(224, 224),
            A.Affine(translate_percent=(-0.01,0.01),p=1.0),
            A.Rotate(limit=10),
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
        component = tuple(component)

        dataset.append(component)
    
    return dataset


def train(epoch):

    start = time.time()
    net.train()
    loss_func = nn.L1Loss()

    model = nn.DataParallel(net)

    for batch_index, (images, center_x, center_y, rot) in enumerate(trainloader_hip):

        if args.gpu:
            images = images.cuda()
            center_x = center_x.cuda()
            center_y = center_y.cuda()
            rot = rot.cuda()

        optimizer.zero_grad()
        outputs = model(images)

        #objective function
        loss = (loss_func(outputs[:,0], center_x.float()) + \
        loss_func(outputs[:,1], center_y.float())) + 2*loss_func(outputs[:,2], rot.float())

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
    parser.add_argument('-BASE_DIR', type=str, default='', help='access data/label')
    parser.add_argument('-LABEL_DIR', type=str, default='', help='access label file')
    parser.add_argument('-level_labels', nargs="+", default=['triradiate','tuberosity','crest','obturator'], help='type of level label for avg rot')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-batch', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.03, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    ###hipdata function
    _IMAGE_DIR = os.path.join(args.BASE_DIR, 'stage_1')
    label_file = pd.read_csv(os.path.join(args.BASE_DIR, 'label', args.LABEL_DIR))#the baseline

    net = get_network(args)

    trainset_hip = dataset_generate_hip(_IMAGE_DIR, label_file, args.level_labels)
    trainloader_hip = torch.utils.data.DataLoader(trainset_hip, batch_size=args.batch, shuffle=True, num_workers=16) 
    ndata = trainset_hip.__len__()
    print('number of hip training data for stage1 model (avg rotation predictor): {}'.format(ndata)) 


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(trainloader_hip)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()

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
