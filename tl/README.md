## Supervised Pretraining

This repository performs supervised pretraining on

1. Source tasks
    - ImageNet: train_imagenet.py
    - SUN397: train_SUN.py
    - Chest-Ray8: train_NIH.py
    - iNatualist: train_inat.py
    - LUNA16: train_luna.py



### File orgnization
```
    - model
        - ResNet.py: define backbone model
    - utils
        - $DATASET-NAME$.py: define dataloader
        - ...
    - train_$DATASET-NAME$.py: start finetuning
```


### Unsupervised Training

To start finetuning on a specific dataset:
```
python train_$DATASET-NAME$.py \
  -model_name resnet50 \
  --lr 0.001 \
  --batch-size 128 \
  --pretrained None\    [None:      Randon initialization;
                         Transfer:  TL pretrained;
                         MoCo:      SSL pretrained;
                         Resume:    Resume checkpoint]
  [each dataset has diffent file orgnization, also specify the data-root]
```


