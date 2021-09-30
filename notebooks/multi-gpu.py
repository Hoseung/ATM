import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets

import atm
import atm.simclr as simclr
import atm.simclr.resnet as models
from atm.simclr.utils import GaussianBlur
import argparse 

import logging
import os
import sys
import yaml

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Setup
do_parallel = False

args = argparse.Namespace()

args.data='./datasets' 
args.dataset_name=['cifar10', 'stl10', 'nair'][0]
args.arch='resnet50'
args.n_channels=1
args.workers=1
args.epochs=400


if args.dataset_name == 'cifar10':
    args.img_size =32
elif args.dataset_name == 'stl10':
    args.img_size = 96
elif args.dataset_name == "nair":
    args.img_size = 128
    args.n_channels = 1

if do_parallel:
    args.batch_size = 128
else:
    args.batch_size =1024

args.lr=0.02
args.weight_decay=0.0005
args.disable_cuda=False
args.fp16_precision=True
args.out_dim=[128,10][0]
args.log_every_n_steps=100
args.temperature=0.07
args.n_views = 2
args.device='cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:", args.device)

assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
# check if gpu training is available
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    #args.gpu_index = -1


from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from functools import partial
from astrobf.tmo import Mantiuk_Seidel

from atm.simclr.dataloader import ContrastiveLearningDataset, ContrastiveLearningViewGenerator
from atm.simclr.utils import save_config_file, accuracy, save_checkpoint
from atm.loader import TonemapImageDataset

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, n_channels=3):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim,
                                                        num_channels=n_channels),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim,
                                                        num_channels=n_channels)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


import time
from atm.simclr.models import SimCLR
            

def get_simclr_pipeline_transform(size, s=1, n_channels=3):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    if n_channels == 3:
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        _transforms = [transforms.RandomResizedCrop(size=size),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomApply([color_jitter], p=0.8),
                      transforms.RandomGrayscale(p=0.2),
                      GaussianBlur(kernel_size=int(0.1 * size)),
                      transforms.ToTensor()]
    elif n_channels == 1:
        _transforms = [transforms.RandomResizedCrop(size=size),
                      transforms.RandomHorizontalFlip(),
                      #GaussianBlur(kernel_size=int(0.1 * size)),
                      transforms.ToTensor()]#,
                      #transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True))]
    
    return transforms.Compose(_transforms)


class ContrastiveLearningDataset:
    def __init__(self, root_folder, n_channels_in=3, n_channels_out=3):
        self.root_folder = root_folder
        self.n_channels_in=n_channels_in
        self.n_channels_out=n_channels_out

    def get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        _transform = [transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()]
        if self.n_channels_in ==3 and self.n_channels_out == 1:
            _transform = _transform + [transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True))]
        return transforms.Compose(_transform)

    def get_dataset(self, dataset_name = 'cifar10', n_views=2):
        if dataset_name == 'cifar10':
            dataset_fn = lambda: datasets.CIFAR10(self.root_folder, 
                                              train=True,
                                              transform=ContrastiveLearningViewGenerator(
                                                  self.get_simclr_pipeline_transform(32), # image size = 32
                                                  n_views),
                                              download=True)
        else:
            raise NotImplementedError('Only cifar10 dataset is supported')

        return dataset_fn()

# Load data   
if args.dataset_name == "nair":
    import pickle
    from astrobf.utils.misc import load_Nair

    ddir = "../../tonemap/bf_data/Nair_and_Abraham_2010/"
    fn = ddir + "all_gals.pickle"
    all_gals = pickle.load(open(fn, "rb"))
#all_gals = all_gals[1:] # Why the first galaxy image is NaN?
    good_gids = np.array([gal['img_name'] for gal in all_gals])

# Catalog
    cat_data = load_Nair(ddir + "catalog/table2.dat")
    cat = cat_data[cat_data['ID'].isin(good_gids)] # pd

    tmo_params = {'b': 6.0,  'c': 3.96, 'dl': 9.22, 'dh': 2.45}

    train_dataset = TonemapImageDataset(all_gals, partial(Mantiuk_Seidel, **tmo_params),
                                        labels=cat['TT'].to_numpy(),
                                        train=True, 
                                        transform=ContrastiveLearningViewGenerator(
                                            get_simclr_pipeline_transform(args.img_size, n_channels=args.n_channels)
                                        ))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
else: 
    dataset = ContrastiveLearningDataset(args.data, n_channels_in=3, n_channels_out=args.n_channels)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, n_channels=args.n_channels)
 # SimCLR때는 out_dim = 128이므로, skwnddp ResNet에 transfer할 때는 마지막 fc를 10개 짜리를 새로 붙인 뒤 다시 training 해줘야함.
if do_parallel:
    model = nn.DataParallel(model)#, output_device=1) # split works into different devices. 1 deals with the output, 0 does the rest.
    # The commented part causes an error:
    # Expected all tensors to be on the same device, but found at least two devices,
    # cuda:1 and cuda:0! (when checking arugment for argument target in method wrapper_nll_loss_forward)

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)


np.seterr(divide='ignore')
#  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
#with torch.cuda.device(args.gpu_index):
simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
simclr.train(train_loader)
