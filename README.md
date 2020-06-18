# Transfer Learning or Self-supervised Learning? A Tale of Two Pretraining Paradigms
Pytorch code and models for paper

Transfer Learning or Self-supervised Learning? A Tale of Two Pretraining Paradigms

This repository contains code and pre-trained models used in the paper and 2 demos to demonstrate:
1) Code for a comprehensive study between SSL and TL regarding which one works better under
    - domain difference between source15and target tasks,
    - the amount of pretraining data
    - class imbalance in source data
    - usage of target data for additional pretraining
2) Code to calculate domain distance between source domain and target domain in term of (1)Visual distance and (2)Class similarity

## Dependencies:
- Python (3.7)
- Pytorch (1.5.0)
- Tensorboard (1.14.0)
- scikit-learn
- https://github.com/ufoym/imbalanced-dataset-sampler

## Datasets
In the paper, we used data from 5 source and 4 target datasets:
* Source:
    - [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/)
    - [SUN397](https://groups.csail.mit.edu/vision/SUN/)
    - [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md)
    - [LUNA16](https://luna16.grand-challenge.org/data/)
    - [ChestX-ray8](https://nihcc.app.box.com/v/ChestXray-NIHCC)
* Target:
    - [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
    - [Oxford Flower 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
    - [COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)
    - [Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## File orgnization
```
    - ssl (Self-supervise pretraining)
        - moco (MoCo pretraining)
    - tl (Supervised pretraining)
    - finetune (Fintune on Target tasks)
    - dataset (Datasplit for Caltech256)
    - domain (visual domain distance & label similarity)
```
