from kaggle.api.kaggle_api_extended import KaggleApi
import scipy.io
import os
import shutil
import numpy as np
from collections import defaultdict

api = KaggleApi()
api.authenticate()

root = "data"

if os.path.exists(f'{root}/cars_dataset'):
    shutil.rmtree(f'{root}/cars_dataset')

def get_stanford_cars(split: str):
    if not os.path.exists(f'{root}/stanford_cars'):
        os.makedirs(f'{root}/stanford_cars', exist_ok=True)
        api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path=root, unzip=True)
    if split == 'test':
        p = f'{root}/stanford_cars/cars_test_annos_withlabels.mat'
        imgpath = f'{root}/stanford_cars/cars_test'
    else:
        p = f'{root}/stanford_cars/devkit/cars_train_annos.mat'
        imgpath = f'{root}/stanford_cars/cars_train'
    data = scipy.io.loadmat(p)
    da = data["annotations"][0]
    samples = [{"bbox": [int(a[0]), int(a[1]), int(a[2]), int(a[3])], "class": int(a[4]) - 1, "fname": str(a[5][0])} for a in da]
    return samples, imgpath

os.makedirs(f'{root}/cars_dataset/train/images', exist_ok=True)
os.makedirs(f'{root}/cars_dataset/test/images', exist_ok=True)

train_samples, train_imgpath = get_stanford_cars('train')
test_samples, test_imgpath = get_stanford_cars('test')

print("✅ Stanford Cars dataset loaded successfully!")

class_counts_train = defaultdict(int)
train_labels = []
for sample in train_samples:
    c = sample["class"]
    class_subdir = f'{root}/cars_dataset/train/images/{c}'
    os.makedirs(class_subdir, exist_ok=True)
    class_counts_train[c] += 1
    new_fname = str(class_counts_train[c]).zfill(5) + '.jpg'
    src = os.path.join(train_imgpath, sample["fname"])
    dst = os.path.join(class_subdir, new_fname)
    shutil.copy(src, dst)
    train_labels.append({"class": c, "fname": f"{c}/{new_fname}"})

print("✅ Images copied successfully for training set!")

class_counts_test = defaultdict(int)
test_labels = []
for sample in test_samples:
    c = sample["class"]
    class_subdir = f'{root}/cars_dataset/test/images/{c}'
    os.makedirs(class_subdir, exist_ok=True)
    class_counts_test[c] += 1
    new_fname = str(class_counts_test[c]).zfill(5) + '.jpg'
    src = os.path.join(test_imgpath, sample["fname"])
    dst = os.path.join(class_subdir, new_fname)
    shutil.copy(src, dst)
    test_labels.append({"class": c, "fname": f"{c}/{new_fname}"})

print("✅ Images copied successfully for test set!")

np.save(f'{root}/cars_dataset/train/labels.npy', train_labels)
np.save(f'{root}/cars_dataset/test/labels.npy', test_labels)

print("✅ Labels saved successfully!")
print("✅ Dataset prepared successfully!")
