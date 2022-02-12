""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def calc_mi(med_head_x, lat_head_x, lat_acetabulum_x, side):
    """Calculate migration index (MI).
    For the right hip, MI = (lat acetabulum - lat head) / (med head - lat head)
    For the left hip first need to multiple each point by -1.
    Args:
        med_head_x: x coordinate of medial head
        lat_head_x: x coordinate of lateral head
        lat_acetabulum_x: x coordinate of lateral acetabulum
        side: side of hip from which points come (i.e., 'right' or 'left')
    Raise:
        ValueError if side not in ['right', 'left']
    Returns:
        Migration index in interval [0,1]
    """
    if side not in ['right', 'left']:
        raise ValueError(f'Side must be "right" or "left", not "{side}"')

    if side == 'left':
        med_head_x *= -1
        lat_head_x *= -1
        lat_acetabulum_x *= -1
    mi = (lat_acetabulum_x - lat_head_x) / (med_head_x - lat_head_x)
    
    # bound within [0, 1]
    mi = max(mi, 0)
    mi = min(mi, 1)
    return mi

def calc_rotation_diff(GT_LABELS, MODEL_PRED):
    model_img, rot_diff = [], []
    df_gt_rot, df_model_rot = pd.read_csv(GT_LABELS), pd.read_csv(MODEL_PRED)

    gt_img_name, gt_img_rot = df_gt_rot[['filename']], df_gt_rot[['degree_corr']]
    model_img_name, model_img_rot = df_model_rot[['file_name']], df_model_rot[['pred_rotation']]

    for i in range(len(model_img_name)):
        m_img, m_rot = model_img_name.iloc[i].iat[0], model_img_rot.iloc[i].iat[0]
        gt_img, gt_rot = gt_img_name.iloc[i].iat[0], gt_img_rot.iloc[i].iat[0]
        model_img.append(m_img)
        rot_diff.append(gt_rot - m_rot)

    return model_img, rot_diff

def visualization_highq(visual_path, mode, img_name, keyps, coords, mp_legend, resize=224, num_key=6):
    vout_path = visual_path+'/voutput_img_{}'.format(mode)
    if not os.path.exists(vout_path):
        os.makedirs(vout_path)

    image = Image.open(visual_path+'/base_img_{}/{}'.format(mode, img_name[0]))
    w, h = image.size
    #print(h,w)
    plt.figure(dpi=1200)#dpi=1200 represents 4K image lol!

    for k in range(num_key):
        k_gt_x, k_gt_y = ((keyps[0][k][0].item()+1)*resize-1)/2, ((keyps[0][k][1].item()+1)*resize-1)/2
        k_pred_x, k_pred_y = ((coords[0][k][0].item()+1)*resize-1)/2, ((coords[0][k][1].item()+1)*resize-1)/2
        rc_size_min = int(min(image.size[0:2]))
        k_gt_x, k_gt_y = k_gt_x*rc_size_min/resize, k_gt_y*rc_size_min/resize
        k_pred_x, k_pred_y = k_pred_x*rc_size_min/resize, k_pred_y*rc_size_min/resize
        
        rc_size_max = int(max(image.size[0:2]))
        if h > w:
            k_gt_y = k_gt_y + (rc_size_max - rc_size_min)/2
            k_pred_y = k_pred_y + (rc_size_max - rc_size_min)/2
        else:
            k_gt_x = k_gt_x + (rc_size_max - rc_size_min)/2
            k_pred_x = k_pred_x + (rc_size_max - rc_size_min)/2


        plt.plot(k_gt_x, k_gt_y, marker='o', color="blue", markersize=2)
        plt.plot(k_pred_x, k_pred_y, marker='o', color="red", markersize=2)

    plt.text(0, h*0.98, 'R-GT:{}  R-AI:{}                L-GT:{}  L-AI:{}'.format(int(mp_legend[0]), \
        int(mp_legend[2]),int(mp_legend[1]),int(mp_legend[3])), size=10,
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),))

    plt.imshow(image)
    plt.show()
    plt.axis('off')

    plt.savefig(os.path.join(vout_path, img_name[0]),bbox_inches='tight',pad_inches = 0)
    plt.close()
    #plt.imsave can achieve the samething

def gen_outputcsv_file(visual_path, mode, csv_file):
    header = ['X-ray_ID','gt_MP_right','gt_MP_left','model_MP_right','model_MP_left']
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    with open(visual_path+'/mp_error_{}.csv'.format(mode), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # write the data
        for files in csv_file:
            writer.writerow(files)




def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]



# def visualization_lowq(visual_path, mode, img_name, keyps, coords, resize=224, num_key=6):
#     vout_path = visual_path+'/voutput_img_{}'.format(mode)
#     if not os.path.exists(vout_path):
#         os.makedirs(vout_path)

#     image = Image.open(visual_path+'/base_img_{}/{}'.format(mode, img_name[0]))

#     for k in range(num_key):
#         plt.plot(((keyps[0][k][0].item()+1)*resize-1)/2,((keyps[0][k][1].item()+1)*resize-1)/2, marker='o', color="blue", markersize=2)
#         plt.plot(((coords[0][k][0].item()+1)*resize-1)/2,((coords[0][k][1].item()+1)*resize-1)/2, marker='o', color="red", markersize=2)

#     plt.imshow(image)
#     plt.show()

#     plt.savefig(os.path.join(vout_path, img_name[0]))
#     plt.close()