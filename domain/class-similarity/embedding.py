from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np
from cosine_similarity import *
from embedding import *
from get_avelabel import *
import scipy.io as scio

def read_mat():
    dataFile = 'path/imagelabels.mat'
    data = scio.loadmat(dataFile)

def get_imagenet_labels():
    """Return list of imagnet labels

    Returns:
        [list(str)] -- list of imagnet labels
    """
    with open('imagenet_class_index.json', 'r') as f:
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


def label_to_embedding(label, word2emb):
    """label to glove """
    # for idx, word in enumerate(label):
    #     if word not in word2emb:
    #         return None
    #     glove_v = word2emb[word]

    # try:
    #     if label not in word2emb:
    #         return None
    #     else:
    #         glove_v = word2emb[label]
    #         return glove_v
    # except:
    #     print('label corrupt', label)
    if isinstance(label, list):
        label_key = label[0]
    else:
        label_key = label
    if label_key not in word2emb:
        return None
    else:
        glove_v = word2emb[label_key]
        return glove_v

def imagenet_embedding(word2emb):
    source_vectors = {}
    source = get_imagenet_labels()
    target = get_imagenet_labels()
    for i, label in enumerate(source):
        imagenet_label = label.replace('_', ' ').split(' ')
        if len(imagenet_label) > 1:
            vector_average = 0
            for word in imagenet_label:
                vector_add = label_to_embedding(word, word2emb)
                if vector_add is not None:
                    vector_average = vector_average + vector_add
            if not isinstance(vector_average, int):
                vector_average = vector_average / len(imagenet_label)
                source_vectors[i] = np.array(vector_average).tolist()
        else:
            source_v = label_to_embedding(imagenet_label, word2emb)
            if source_v is not None:
                source_vectors[i] = np.array(source_v).tolist()
        print(i)

    with open("path/imagenet_glove.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")

def COVID_embedding(word2emb):
    p_emb = label_to_embedding('pneumonia', word2emb)
    ### 349
    n_emb_add = label_to_embedding('not', word2emb)
    n_emb = (p_emb+n_emb_add)/2
    ### 398
    ##total 747

    return p_emb, n_emb

def phe_embedding(word2emb):
    p_emb = label_to_embedding('pneumonia', word2emb)
    ### 3875 +8 +390 = 4273
    n_emb_add = label_to_embedding('not', word2emb)
    n_emb = (p_emb+n_emb_add)/2
    ### 1341 +8 +234 = 1583
    ##total 5856
    return p_emb, n_emb

def luna_embedding(word2emb):
    p_emb = (label_to_embedding('lung', word2emb) + label_to_embedding('cancer', word2emb))/2
    ### 785
    n_emb = (label_to_embedding('not', word2emb) + label_to_embedding('lung', word2emb) + label_to_embedding('cancer', word2emb))/3
    ### 70720
    ##total 71505

    return p_emb, n_emb

def embedding(word2emb):
    source_vectors = {}
    source = get_cal_labels()
    for i, label in enumerate(source):
        imagenet_label = label.replace('_', ' ').split(' ')
        if len(imagenet_label) > 1:
            vector_average = 0
            for word in imagenet_label:
                vector_add = label_to_embedding(word, word2emb)
                if vector_add is not None:
                    vector_average = vector_average + vector_add
            if not isinstance(vector_average, int):
                vector_average = vector_average / len(imagenet_label)
                source_vectors[i] = np.array(vector_average).tolist()
        else:
            source_v = label_to_embedding(imagenet_label, word2emb)
            if source_v is not None:
                source_vectors[i] = np.array(source_v).tolist()

    with open("flo_glove.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")


def inat_embedding(word2emb):
    source_vectors = {}
    source = get_inat_labels()
    for i, label in enumerate(source):
        imagenet_label = label.lower().split(' ')
        if len(imagenet_label) > 1:
            vector_average = 0
            for word in imagenet_label:
                vector_add = label_to_embedding(word, word2emb)
                if vector_add is not None:
                    vector_average = vector_average + vector_add
            if not isinstance(vector_average, int):
                vector_average = vector_average / len(imagenet_label)
                source_vectors[i] = np.array(vector_average).tolist()
        else:
            source_v = label_to_embedding(imagenet_label, word2emb)
            if source_v is not None:
                source_vectors[i] = np.array(source_v).tolist()
        print('a')
    with open("inat_glove8000.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")


def cal_embedding(word2emb):
    source_vectors = {}
    source = get_cal_labels()
    original_cal_label = {}
    for i, label in enumerate(source):
        imagenet_label = label.lower().split(' ')
        if len(imagenet_label) > 1:
            vector_average = 0
            for word in imagenet_label:
                vector_add = label_to_embedding(word, word2emb)
                if vector_add is not None:
                    vector_average = vector_average + vector_add
            if not isinstance(vector_average, int):
                vector_average = vector_average / len(imagenet_label)
                source_vectors[i] = np.array(vector_average).tolist()
                original_cal_label[i] = imagenet_label
        else:
            source_v = label_to_embedding(imagenet_label, word2emb)
            if source_v is not None:
                source_vectors[i] = np.array(source_v).tolist()
                original_cal_label[i] = imagenet_label
    print('a')
    with open("cal_glove.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")
    with open("cal_label_vertorized.json", "w") as f:
        json.dump(original_cal_label, f)
        print("loading finished")


def nih_embedding(word2emb):
    source_vectors = {}
    number_vectors = {}
    source, n = get_nih_labels()
    for i, label in enumerate(source):
        imagenet_label = label.lower().replace('-', ' ').split(' ')
        if len(imagenet_label) > 1:
            vector_average = 0
            for word in imagenet_label:
                vector_add = label_to_embedding(word, word2emb)
                if vector_add is not None:
                    vector_average = vector_average + vector_add
            if not isinstance(vector_average, int):
                vector_average = vector_average / len(imagenet_label)
                source_vectors[i] = np.array(vector_average).tolist()
        else:
            source_v = label_to_embedding(imagenet_label, word2emb)
            if source_v is not None:
                source_vectors[i] = np.array(source_v).tolist()
        number_vectors[i] =  n[i]
        print('a')
    with open("nih_glove.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")
    return number_vectors