import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision.transforms import transforms
from atm.simclr import resnet
from torch.utils.tensorboard import SummaryWriter
#from tensorflow import summary


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# check if gpu training is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

print("Using...", device)


model = resnet.resnet50(pretrained=False, num_classes=10)
model.to(device)

#print(model)

dataset = 'stl10'

if dataset == 'cifar10':
    img_size = 32
    batch_size = 128
    lr = 0.001
elif dataset == 'stl10':
    img_size = 96
    batch_size = 64
    lr = 0.001

transform_train = transforms.Compose(
    [transforms.RandomCrop(img_size),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
     transforms.Normalize((0.5), (0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
     transforms.Normalize((0.5), (0.5))])

n_epochs=400

if dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
elif dataset == 'stl10':
    trainset = torchvision.datasets.STL10(root='./data', split='train',
                                        download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data', split='test',
                                       download=True, transform=transform_test)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


checkpoint_name=f'Resnet_ch1_{dataset}_bn{batch_size}_{n_epochs}_'

import torch.optim as optim

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

config = Config(
    trainloader = train_loader,
    testloader = test_loader,
    model = model,
    device = device,
    optimizer = torch.optim.Adam(model.parameters(), lr=lr),
    criterion= nn.CrossEntropyLoss().to(device),
    globaliter = 0
)



#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# do Training
tb = SummaryWriter() 
print(tb.log_dir)

#train_summary_writer = summary.create_file_writer(train_log_dir)
#test_summary_writer =  summary.create_file_writer(test_log_dir)

class train_test():
    def __init__(self, config):
        self.trainloader = config.trainloader
        self.testloader = config.testloader
        self.model = config.model
        self.device = config.device
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.globaliter = config.globaliter

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
                    tb.add_scalar('train_loss', loss.item() , global_step = self.globaliter)
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

                print('\nTest set : Average loss:{:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
                    test_loss, correct, total, 100 * correct/total
                ))
                #with test_summary_writer.as_default():
                #    summary.scalar('test_loss', test_loss , step = self.globaliter)
                #    summary.scalar('accuracy', 100 * correct/total , step = self.globaliter)
                tb.add_scalar('test_loss', test_loss , global_step = self.globaliter)
                tb.add_scalar('accuracy', 100 * correct/total, global_step = self.globaliter)

            lr_sche.step()

        tb.close()
        save_checkpoint({
        'epoch': n_epochs,
        'state_dict': model.state_dict(),
        'optimizer': config.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(tb.log_dir, checkpoint_name+f'{epoch}.pth'))

"""

for epoch in range(n_epochs):
    running_loss = 0.0
    total_correct = 0.0
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_correct+= get_num_correct(preds, labels)
    
    #print('[%d, %5d] loss: %.4f' %
    #      (epoch + 1, running_loss / 2000))
    if epoch % 100 == 99:    # print every 2000 mini-batches
        save_checkpoint({
        'epoch': n_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join('./', checkpoint_name+f'{epoch}.pth'))

    tb.add_scalar("Loss", running_loss, epoch)
    tb.add_scalar("Correct", total_correct, epoch)
    tb.add_scalar("Accuracy", total_correct/ len(trainset), epoch)

    #tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
    #tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
    #tb.add_histogram("conv2.bias", model.conv2.bias, epoch)
    #tb.add_histogram("conv2.weight", model.conv2.weight, epoch)        
"""

ready_to_train=train_test(config)
lr_sche = optim.lr_scheduler.StepLR(config.optimizer, step_size=5000, gamma=0.5) # 20 step마다 lr조정
log_interval = 58

ready_to_train.train(n_epochs, log_interval)


