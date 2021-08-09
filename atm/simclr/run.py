import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from .dataloader import ContrastiveLearningDataset
from .models import ResNetSimCLR, SimCLR

def main():
    args = argparse.Namespace()

    args.data='./datasets' 
    args.dataset_name='cifar10'
    args.n_views = 2
    args.workers=1
    args.arch='resnet50'
    args.out_dim=128
    args.wd=0.0005
    args.batch_size=256 
    args.device='cuda'

    dataset = ContrastiveLearningDataset(args.data) 
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # backbone model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    # Train
    # model is saved at the end of train.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader) 