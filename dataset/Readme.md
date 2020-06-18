# Dataset Preprocessing
1. luna_preprocess.py
    - Extract CT slices from LUNA raw dataset and save to png images

2. caltech_split.py
    - Since Caltech256 do not have official data split, we randomly draw 30:25:25=train:val:test for each class
    - caltech256_train_0.csv/caltech256_val_0.csv/caltech256_test_0.csv contains the train/val/test list of images in each category. move caltech_split.py into the folder which contains '256_ObjectCategories' then run it.

