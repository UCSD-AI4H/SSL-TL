import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

DATA_LS_0 = []
DATA_LS_1 = []

TOTAL_NUM = 574
RATIO = 0.8

TRAIN_LS = []
VAL_LS = []

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing


def LUNA_split():
    raw_dir = './trainB_raw' # files with .mhd files
    to_dir = '../luna_new_tl/'
    to_dir_ssl = '../luna_new_ssl/data/'
    data = pd.read_csv('candidates_V2.csv')
    cand_path = './candidates_V2.csv'
    cand = pd.read_csv(cand_path)
    
    
    img_list = [img_name for img_name in os.listdir(raw_dir) if img_name.endswith('mhd')]
    # os.makedirs(to_dir)
    cnt = 0
    train_dir = True
    for meta_name in tqdm(img_list):
        cnt = cnt + 1
        if cnt < TOTAL_NUM * RATIO:
            train_dir = True
        else:
            train_dir = False
        
        ct_name = meta_name[:-4] 
        print(meta_name)


        meta_path_from = os.path.join(raw_dir,meta_name)
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(meta_path_from)
        # Select slices in 1/5-4/5
        sliced_list = list(range(int(numpyImage.shape[0] / 5), int(4 * numpyImage.shape[0] / 5)))
        
        
        this_cand = cand.loc[cand['seriesuid'] == ct_name, :]
        label = this_cand[['class']]
        stretchedVoxelCoord = np.absolute(this_cand[['coordZ', 'coordY', 'coordX']] - numpyOrigin)
        voxelCoord = stretchedVoxelCoord / numpySpacing
        table = label.join(voxelCoord)
        ls_label1 = table.loc[table['class'] == 1, :].reset_index(drop=True)
        idx_label1 = ls_label1['coordZ'].tolist()
        idx_label1 = [int(x) for x in idx_label1]
        
        for i, idx in enumerate(idx_label1):
            print('label1: ', idx)
            if idx in sliced_list:
                sliced_list.remove(idx)
            # Slice
            image = numpyImage[idx]     
            image = normalizePlanes(image)
            
            # Save
            if train_dir == True:
                to_dir = '../luna_new_tl/train/1/'   
            else:
                to_dir = '../luna_new_tl/val/1/'   
            img_path_to = os.path.join(to_dir, '{}_{}.png'.format(ct_name, idx))

            plt.imsave(img_path_to, image, cmap=plt.cm.gray)
            img_path_to_ssl = os.path.join(to_dir_ssl, '{}_{}.png'.format(ct_name, idx))

            plt.imsave(img_path_to_ssl, image, cmap=plt.cm.gray)
            

        for i in sliced_list:
            if i % 100 == 0:
                print(i)
                
            if train_dir == True:
                to_dir = '../luna_new_tl/train/0/'   
            else:
                to_dir = '../luna_new_tl/val/0/'
  
            img_path_to = os.path.join(to_dir, '{}_{}.png'.format(ct_name, i))
            img_path_to_ssl = os.path.join(to_dir_ssl, '{}_{}.png'.format(ct_name, i))
            img = numpyImage[i]
            img = normalizePlanes(img)

            plt.imsave(img_path_to, img, cmap=plt.cm.gray)
            plt.imsave(img_path_to_ssl, img, cmap=plt.cm.gray)



def normalizePlanes(npzarray):

    maxHU = 50.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray


if __name__ == '__main__':
    LUNA_split()

