## Finetuning

This repository performs supervised finetuning on

1. Target tasks
    - Caltech256: train_caltech.py
    - Pnuemonia: train_pnuemonia.py
    - COVID-CT: train_covid.py
    - Flower102: train_flower102.py

2. Source(SUN397)+Target tasks
    - SUN+Caltech256: train_combine_SUN_caltech.py
    - SUN+Pnuemonia: train_combine_SUN_pne.py
    - SUN+COVID-CT: train_combine_SUN_covid.py
    - SUN+Flower102: train_combine_SUN_flower.py

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


