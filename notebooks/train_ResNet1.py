import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.transforms import transforms
import argparse

from atm.simclr import resnet
from torch.utils.tensorboard import SummaryWriter
#from tensorflow import summary
from atm.simclr.utils import save_config_file, accuracy, save_checkpoint

#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')


args = argparse.Namespace()

args.data='./datasets'
args.dataset=['cifar10', 'stl10', 'nair'][0]
args.arch='resnet18'
args.workers=1
args.epochs=200
args.img_size =128
args.n_channels=1
args.weight_decay=0.0008

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# check if gpu training is available
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    args.device = torch.device('cpu')

print("Using...", args.device)


model = resnet.resnet50(pretrained=False, num_classes=10)
model.to(args.device)

#print(model)


if args.dataset == 'cifar10':
    img_size = 32
    args.batch_size = 256
    args.lr = 0.001
elif args.dataset == 'stl10':
    img_size = 96
    args.batch_size = 64
    args.lr = 0.001

transform_train = transforms.Compose(
    [transforms.RandomCrop(img_size),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
     transforms.Normalize((0.5), (0.2))])

transform_test = transforms.Compose(
    #[transforms.RandomHorizontalFlip(p=0.5),
     [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
     transforms.Normalize((0.5), (0.2))])


if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
elif args.dataset == 'stl10':
    trainset = torchvision.datasets.STL10(root='./data', split='train',
                                        download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data', split='test',
                                       download=True, transform=transform_test)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)


checkpoint_name=f'Resnet_ch1_{args.dataset}_bn{args.batch_size}_{args.epochs}_'

import torch.optim as optim

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = Config(
    trainloader = train_loader,
    testloader = test_loader,
    model = model,
    device = args.device,
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    criterion= nn.CrossEntropyLoss().to(args.device),
    globaliter = 0
)


#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# do Training

import time
class train_test():
    def __init__(self, config):
        self.trainloader = config.trainloader
        self.testloader = config.testloader
        self.model = config.model
        self.device = config.device
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.globaliter = config.globaliter
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = './runs/'+timestr + f"_{args.dataset}_{args.arch}_{args.n_channels}_{args.batch_size}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, epochs, log_interval):
        self.model.train()
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                self.globaliter += 1
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if i % log_interval == log_interval -1 :
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlearningLoss: {:.6f}\twhole_loss: {:.6f} '.format(
                        epoch, i*len(inputs), len(self.trainloader.dataset),
                        100. * i*len(inputs) / len(self.trainloader.dataset), 
                        running_loss / log_interval,
                        loss.item()))
                    running_loss = 0.0

                    #with train_summary_writer.as_default():
                    #    summary.scalar('train_loss', loss.item() , step = self.globaliter)
                self.writer.add_scalar('train_loss', loss.item() , global_step = self.globaliter)
            # evaluate
            with torch.no_grad():
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0
                acc = []
                for k, data in enumerate(self.testloader, 0):
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                   
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss += self.criterion(outputs, labels).item()
                    acc.append(100 * correct/total)

            print('\nTest set : Average loss:{:.4f}, Accuracy: {}/{}({:.5f}%)\n'.format(
                  test_loss, correct, total, 100 * correct/total))
            self.writer.add_scalar('test_loss', test_loss , global_step = self.globaliter)
            self.writer.add_scalar('accuracy', 100 * correct/total, global_step = self.globaliter)

            lr_sche.step()

        save_checkpoint({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': config.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name+f'{epoch}.pth'))
        self.writer.close()

ready_to_train=train_test(config)
lr_sche = optim.lr_scheduler.StepLR(config.optimizer, step_size=5000, gamma=0.5) # 20 step마다 lr조정
log_interval = 58

ready_to_train.train(args.epochs, log_interval)


