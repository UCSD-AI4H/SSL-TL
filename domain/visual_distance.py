from utils import autoaugment as auto
import math

from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import os
from torchvision import transforms
import torchvision
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
import cv2

import logging
logger = logging.getLogger('AutoML')
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
fh = logging.FileHandler('test.log')
logger.addHandler(fh)

MODEL_DICT = {
    'resnet50': torchvision.models.resnet50
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


def normalize_dataset():
    for i in range(1, 1000):
        Images = cv2.imread("path/proxy_distance/SUN/SUN%s.png" % i)
        image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        X.append(((hist / 255).flatten()))
        Y.append(1)

        X = np.array(X)
        Y = np.array(Y)

class DomainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = ['0', '1']
        self.num_cls = len(self.classes)
        self.img_list = []

        self.source = 'path1'


        self.target = 'path2'



        source_list = [[os.path.join(self.source,item), 0] for item in os.listdir(self.source)]
        target_list = [[os.path.join(self.flo, item), 1] for item in os.listdir(self.flo)][0:999]



        self.img_list += source_list
        self.img_list += target_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            # print(img_path)
            # print(int(self.img_list[idx][1]))
            sample = {'img': image,
                      'label': int(self.img_list[idx][1])}


        except:
            print('corrupt image')
            img_path = self.img_list[0][0]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            sample = {'img': image,
                      'label': int(self.img_list[idx][1])}

        return sample


class simple_classfifer(torch.nn.Module):
    def __init__(self):
        super(simple_classfifer, self).__init__()
        layer1 = torch.nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(21632, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, LOSS_FUNC, device):
    model.train()
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # try:
    for index, batch_samples in enumerate(tqdm(train_loader)):
        # logger.info(images)
        # logger.info(target)
        images, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images[0].size(0))
        acc1 = accuracy(output, target, topk=(1,2))
        top1.update(acc1[0], images[0].size(0))


        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d]\tIter [%d/%d]\tAvg Loss: %.4f\tLoss: %.4f\tAvg Acc1: %.4f\tAcc1: %.4f'
                       % (epoch + 1, 100, index + 1, len(train_loader), losses.avg, loss.item(), top1.avg, acc1[0]))

        if index  != 0:

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

    # except:
    #         print('enumerate error')


    return losses.avg

def test(model, test_loader,nb_classes, LOSS_FUNC, device):
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        # for index, (images, im_id, target, tax_ids) in enumerate(tqdm(test_loader)):
        try:
            for index, batch_samples in enumerate(tqdm(test_loader)):
                # logger.info(images)
                # logger.info(target)
                images, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
                output = model(images)
                _, preds = torch.max(output, 1)
                loss = LOSS_FUNC(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 2))

                losses.update(loss.item(), images[0].size(0))
                top1.update(acc1[0], images[0].size(0))
                top5.update(acc5[0], images[0].size(0))
                for t, p in zip(target.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                y_score = F.softmax(output, dim=1)
                y_scores.append(y_score.cpu().numpy())
                y_true.append(target.long().cpu().numpy())
        except:
            print('corrupt image')

    # print('y_true_o',y_true)
    # y_true = np.concatenate(y_true,axis=0)
    # print('y_true',y_true)
    # print('y_scores', y_scores)
    # print('nb_classes', nb_classes)
    aucs = 1
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Val Loss {losses.avg:.3f}'
          .format(top1=top1, top5=top5, losses=losses))
    e = 1-top1.avg/100
    print('generize error',e)
    PAD = 2*(1-2*e)
    print('PAD', PAD)
    return top1.avg, top5.avg, confusion_matrix, losses.avg, aucs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # print('output', output)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        # print('pred',pred)
        # print('target',target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # print(res)
        return res

def auc(y_scores, y_true, nb_class):
    '''Return a list of AUC for each class'''
    y_true = np.eye(nb_class)[y_true]
    aucs = []
    for c in range(nb_class):

        AUC = roc_auc_score(y_true[:,c], y_scores[:,c])
        aucs.append(AUC)
    return aucs

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_trans = transforms.Compose(
                                 [
                                  transforms.Resize(256),
                                  transforms.CenterCrop((224,224)),
                                  transforms.ToTensor(),
                                  normalize
                                 ]
                             )

    trainset = DomainDataset(root_dir='path_covid/COVID-CT/Images-processed',
                              transform=transforms.Compose(
                                  [transforms.Resize(256),
                                   transforms.RandomHorizontalFlip(),
                                   auto.ImageNetPolicy(),
                                   transforms.RandomResizedCrop(224, scale=(0.5, 1.2)),
                                   transforms.ToTensor(),
                                   normalize
                                   ]))

    origin_len = len(trainset)
    trainset, testset = torch.utils.data.random_split(trainset, [int(0.7 * len(trainset)),
                                                           origin_len - int(
                                                               0.7 * len(trainset))])

    train_loader = DataLoader(trainset,
                              batch_size=16,
                              num_workers=8,
                              shuffle=True,pin_memory=True)
    val_loader = DataLoader(testset, batch_size=128,num_workers=8,pin_memory=True, shuffle=True)
    test_loader = DataLoader(testset,batch_size=16)


    LOSS_FUNC = nn.CrossEntropyLoss().to(device)


    # Random Initialize
    print('Random Initialize')
    model = simple_classfifer()
    # model = torchvision.models.resnet50(num_classes=2, pretrained=False)

    # Dataparallel for multiple GPU usage
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer and learning rate scheduler

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # print('start test')
    # for index, (images, target) in enumerate(tqdm(val_loader)):
    #     print(target)

    total_epoch = 10
    metric = []
    acc1, acc5, confusion_matrix, val_loss, aucs = test(model, val_loader, 2, LOSS_FUNC, device)
    for epoch in range(1, total_epoch):
        train_loss = train(model, train_loader, optimizer, 3, epoch, LOSS_FUNC, device)
        acc1, acc5, confusion_matrix, val_loss, aucs = test(model, val_loader, 2, LOSS_FUNC, device)
        metric.append(acc1)

        # Save train/val loss, acc1, acc5, confusion matrix(F1, recall, precision), AUCs
        record = {
            'epoch': epoch + 1,
            'train loss': train_loss,
            'val loss': val_loss,
            'acc1' : acc1,
            'acc5' : acc5,
            'confusion matrix':confusion_matrix,
            'AUCs': aucs
        }
        # torch.save(record, os.path.join('./','recordEpoch{}.pth.tar'.format(epoch)))
        # Only save the model with highest top1 acc
        if np.max(metric) == acc1:
            checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(checkpoint, os.path.join('./','nih_flo_best.pth.tar'))
            print("Model Saved")

    print('...........Testing..........')
    load_resume(args, model, optimizer, os.path.join('./','nih_flo_best.pth.tar'))
    acc1, acc5, confusion_matrix, val_loss, aucs = test(model, test_loader, 2, LOSS_FUNC, device)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    print("Learning rate adjust to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_resume(args, model, optimizer, load_path):
    if load_path:
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))

            checkpoint = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))

if __name__ == '__main__':
    print("Start training")
    main()