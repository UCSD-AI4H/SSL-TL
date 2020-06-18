import csv
import os
import shutil

dataset='./256_ObjectCategories'


n=0
train_file='caltech256_train_{}.csv'.format(n)
val_file='caltech256_val_{}.csv'.format(n)
test_file='caltech256_test_{}.csv'.format(n)

new_dataset='./256_ObjectCategories_{}'.format(n)

if not os.path.isdir(new_dataset):
    os.mkdir(new_dataset)

if not os.path.isdir(os.path.join(new_dataset,'train')):
    os.mkdir(os.path.join(new_dataset,'train'))

if not os.path.isdir(os.path.join(new_dataset,'val')):
    os.mkdir(os.path.join(new_dataset,'val'))

if not os.path.isdir(os.path.join(new_dataset,'test')):
    os.mkdir(os.path.join(new_dataset,'test'))

for cate in ['train','val','test']:
    cate_file = 'caltech256_{}_{}.csv'.format(cate, n)
    with open(cate_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            cls_name = row[0]
            file_name = row[1]
            if not os.path.isdir(os.path.join(new_dataset, cate, cls_name)):
                os.mkdir(os.path.join(new_dataset, cate, cls_name))
            shutil.copyfile(os.path.join(dataset, cls_name, file_name),
                            os.path.join(new_dataset, cate, cls_name, file_name))

