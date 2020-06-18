import torch.utils.data as data
import torch
from PIL import Image
import os
import json
import numpy as np
import random
import tarfile
import io
import csv


def default_loader(path):
    return Image.open(path).convert('RGB')


class NIHDataset(data.Dataset):
    def __init__(self, root, ann_file, transforms=None, categoriesid=None):

        self.transforms = transforms
        self.images = []
        self.labels = []
        if categoriesid:
            self.categoriesid = categoriesid
        else:
            self.categoriesid = {}
        self.categories_cnt = {}
        category_cnt = 0
        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file, 'r', newline='') as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                self.images.append(os.path.join(root, row[0]))
                if row[1] not in self.categoriesid:
                    self.categoriesid[row[1]] = category_cnt
                    category_cnt += 1
                if self.categoriesid[row[1]] not in self.categories_cnt:
                    self.categories_cnt[self.categoriesid[row[1]]] = 0
                self.categories_cnt[self.categoriesid[row[1]]] += 1
                self.labels.append(self.categoriesid[row[1]])
        print('Number of class {}\t Number of samples {}'.format(len(self.categoriesid.keys()), len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.labels[index]
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def get_label(self, index):
        target = self.labels[index]
        return target

    def get_categoriesid(self):
        return self.categoriesid

    def get_categories_weight(self):
        weight = []
        sum_weight = 0
        for key in self.categories_cnt.keys():
            weight.append(1/len(self.categoriesid.keys()) / self.categories_cnt[key])
            sum_weight += 1 / self.categories_cnt[key]
        # for i in range(len(weight)):
        #     weight[i]/=sum_weight
        self.weight = torch.FloatTensor(weight)
        print(self.categoriesid)
        print(self.categories_cnt)
        print(self.weight)

        return self.weight
