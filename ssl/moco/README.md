## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a PyTorch implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722) and [MoCo v2 paper](https://arxiv.org/abs/2003.04297). It is folked form the official implementation.

It contains MoCo on
1. Source tasks
    - ImageNet: main_moco_source_imagenet.py
    - SUN397: main_moco_source_SUN.py
    - Chest-Ray8: main_moco_source_NIH.py
    - iNatualist: main_moco_source_inat.py
    - LUNA16: main_moco_source_luna.py
2. Target tasks
    - Caltech256: main_moco_target_caltech.py
    - Pnuemonia: main_moco_target_pnuemonia.py
    - COVID-CT: main_moco_target_covid.py
    - Flower102: main_moco_target_flower.py

3. Source(SUN397)+Target tasks
    - SUN+Caltech256: main_moco_SUN_caltech.py
    - SUN+Pnuemonia: main_moco_SUN_pnemonia.py
    - SUN+COVID-CT: main_moco_SUN_covid.py
    - SUN+Flower102: main_moco_SUN_flower.py

### File orgnization
```
    - model
        - ResNet.py: define backbone model
    - utils
        - $DATASET NAME$.py: define dataloader
        - ...
    - moco
        - builder.py: defines the moco model
        - loader.py: special data preprocessing for moco
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco_XXX.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```



### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Reference
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```