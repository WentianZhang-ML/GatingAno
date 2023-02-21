import argparse
import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import utils
import models
from train_func import *

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='IDRiD', help='Dataset [IDRiD/IDRiDc/ADAM/ADAMc]')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval_epoch", type=int, default=1)
parser.add_argument("--level", type=str,default='image', help='image level or pixel level')
parser.add_argument("--deta", type=float, default=0.02)
parser.add_argument("--opt", type=str, default='Adam', help='Adam/SGD')
parser.add_argument("--others", type=str,default='None', help='note extra information in log files')


# training settings
class BaselineConfig(object):
    dataset = {
        # pixel-level
        'IDRiD': '/apdcephfs/share_1290796/waltszhang/TMI/labels/pixel_level/Fundus_leison_3hog_Right/',
        'ADAM': '/apdcephfs/share_1290796/waltszhang/TMI/labels/pixel_level/ADAM/', 
        # image-level
        'IDRiDc': '/apdcephfs/share_1290796/waltszhang/TMI/labels/image_level/Fundus_leison_3hog/',
        'ADAMc': '/apdcephfs/share_1290796/waltszhang/TMI/labels/image_level/ADAMc/'
    }
    savedir = './ckpts/'
    batch_size_base = 32  
    lr_base = 1e-4 
    n_epochs_base = 200 
    eval_epoch = 1
    weight_decay_base = 5e-5


def train():
    args = parser.parse_args()
    record_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    config_ = BaselineConfig()
    args.save_dir = config_.savedir + args.dataset + '/' + args.model + '/Time_' + str(record_time) + 'lr_' + str(args.lr)+ '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger_ = utils.logg(config_, args, record_time)

    model = models.GatingAno(n_classes=2)
    model_d = models.Discriminator(in_channels=1)
    train_step(model, model_d, dataset_name=args.dataset, args=args, config=config_, logger=logger_, gating=True)

if __name__ == "__main__":
    train()
