import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.transforms import transforms

import atm
import atm.simclr as simclr
import atm.simclr.resnet as models

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
args.arch='resnet18'
args.workers=1
args.epochs=200
args.img_size =128
args.n_channels=1

if do_parallel:
    args.batch_size = 128
else:
    args.batch_size = 256

args.lr=0.02
args.weight_decay=0.0005
args.disable_cuda=False
args.fp16_precision=True
args.out_dim=10
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

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
class TonemapImageDataset(VisionDataset):
    def __init__(self, 
                 data_array, 
                 tmo,
                 labels: Optional = None, 
                 train: bool=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,):
        self._array = data_array
        self._good_gids = np.array([gal['img_name'] for gal in data_array])
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.tmo = tmo
        self._bad_tmo=False

    def _apply_tm(self, image):
        try:
            return self.tmo(image)
        except ZeroDivisionError:
            print("division by zero. Probably bad choice of TM parameters")
            self._bad_tmo=True
            return image

    def _to_8bit(self, image):
        """
        Normalize per image (or use global min max??)
        """

        image = (image - image.min())/image.ptp()
        image *= 255
        return image.astype('uint8')        
    
    def __len__(self) -> int:
        return len(self._array)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        For super
        """
        image, _segmap, weight = self._array[idx]['data']
        image[~_segmap.astype(bool)] = 0#np.nan # Is it OK to have nan?
        image[image < 0] = 0

        image = self._to_8bit(self._apply_tm(image))
        image = Image.fromarray(image)
        target = self.img_labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target

from atm.simclr.utils import save_config_file, accuracy, save_checkpoint
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
class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = './runs/'+timestr + f"_{self.args.dataset_name}_{self.args.arch}_{self.args.n_channels}_{self.args.batch_size}"
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, checkpoint_freq=100):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            
            if epoch_counter % checkpoint_freq == checkpoint_freq -1 or epoch_counter == self.args.epochs-1:
                # save model checkpoints
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'dataset':self.args.dataset_name,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'batchsize': self.args.batch_size,
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        logging.info("Training has finished.")
            

def get_simclr_pipeline_transform(size, s=1, n_channels=3):
    """Return a set of data augmentation transformations as described in the SimCLR paper.
       What about normalization?????"""
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
                      transforms.ToTensor(),
                      transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True))]
    
    return transforms.Compose(_transforms)


# Load data   
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
                                        get_simclr_pipeline_transform(128, n_channels=args.n_channels)
                                    ))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, n_channels=args.n_channels)
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
