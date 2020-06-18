from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

# with open('path/imagenet_glove.json', 'r') as f:
#     data = json.load(f)
#
# source_vectors = data['0']
# target_vectors = data['1']
# simple_sim = cos_sim(source_vectors, target_vectors)