from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np
from cosine_similarity import *
from embedding import *
from get_avelabel import *
import scipy.io as scio

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

    with open("imagenet_glove.json", "w") as f:
        json.dump(source_vectors, f)
        print("loading finished")

def COVID_embedding(word2emb):
    p_emb = label_to_embedding('pneumonia', word2emb)
    ### 349
    n_emb_add = label_to_embedding('not', word2emb)
    n_emb = (p_emb+n_emb_add)/2
    ### 398
    ##total 747

    # no = 785 / 71505 * 349 / 747 * cos_sim(p_emb, lp) + 785 / 71505 * 398 / 747 * cos_sim(n_emb,lp) + 70720 / 71505 * 398 / 747 * cos_sim(
    #     n_emb, ln) + +70720 / 71505 * 349 / 747 * cos_sim(p_emb, ln)
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

    ###luna pne
    # no = 785 / 71505 * 4273 / 5856 * cos_sim(p_emb, lp) + 785 / 71505 * 1583 / 5856 * cos_sim(n_emb,lp) + 70720 / 71505 * 1583 / 5856 * cos_sim(
    #     n_emb, ln) + +70720 / 71505 * 4273 / 5856 * cos_sim(p_emb, ln)
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



glove_file = 'path/glove/glove.6B.300d.txt'

word2emb = {}
with open(glove_file, 'r') as f:
    entries = f.readlines()
emb_dim = len(entries[0].split(' ')) - 1
print('embedding dim is %d' % emb_dim)

"""glove to np.array"""
for entry in entries:
    vals = entry.split(' ')
    word = vals[0]
    vals = list(map(float, vals[1:]))
    word2emb[word] = np.array(vals)

number_vectors = nih_embedding(word2emb)
with open('/Users/jkooy/research/nips/proxy_distance/inat_glove8000.json', 'r') as f:
    data = json.load(f)

# p_emb, n_emb =  COVID_embedding(word2emb)
p_emb, n_emb =  luna_embedding(word2emb)
# lp, ln = luna_embedding(word2emb)

no = 0
for index, i in enumerate(data.values()):
    no += int(number_vectors[index])/sum*349/747*cos_sim(p_emb, i) + int(number_vectors[index])/sum*398/747*cos_sim(n_emb, i)

# no = 0
# for index, i in enumerate(data.values()):
#     no += int(number_vectors[index])/sum*4273/5856*cos_sim(p_emb, i) + int(number_vectors[index])/sum*1583/5856*cos_sim(n_emb, i)



# no = 0
# max_h = 0
# for index_value, i in enumerate(data3.values()):
#     max_temp = min(cos_sim(lp, i), cos_sim(ln, i))
#     no += 1 / len(data3) * 4273 / 5856 * cos_sim(lp, i) + 1 / len(data3) * 1583 / 5856 * cos_sim(ln, i)
#     if max_temp < max_h:
#         max_h =max_temp
#         index = index_value