from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np

def read_mat():
    dataFile = 'path/imagelabels.mat'
    data = scio.loadmat(dataFile)

def get_imagenet_labels():
    """Return list of imagnet labels

    Returns:
        [list(str)] -- list of imagnet labels
    """
    with open('../imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
    imagenet_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return imagenet_labels

def get_flower_labels():
    fname = 'path/flo_labels.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        s = [i[:-1].split(',') for i in f.readlines()]
    flo_labels = [s[k][1] for k in range(len(s))]
    return flo_labels

def get_inat_labels():
    # fname = 'path/inat_label.txt'
    # with open(fname, 'r+', encoding='utf-8') as f:
    #     s = [i.lower().replace('\n','') for i in f.readlines()]
    # return s

    "8000"
    with open('path/categories.json', 'r') as f:
        data = json.load(f)
    name_list = [i['name'] for i in data]

    filename = open('path/inat_8000categories.txt', 'w')
    for i in name_list:
        filename.write(i)
        filename.write('\n')
    filename.close()
    return name_list

def get_cal_labels():
    fname = 'path/cal_label.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        s = [i.replace('-',' ').replace('\n','') for i in f.readlines()]
    return s

def get_sun_labels():
    fname = 'path/ClassName.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        s = [i.replace('_',' ').replace('\n','').split('/')[-1] for i in f.readlines()]
    return s

def get_nih_labels():
    fname = 'path/nih.txt'
    with open(fname, 'r+', encoding='utf-8') as f:
        n = [i.replace('\n','').split(',')[-1] for i in f.readlines()]
    with open(fname, 'r+', encoding='utf-8') as f:
        s = [i.split(',')[0] for i in f.readlines()]
    return s, n