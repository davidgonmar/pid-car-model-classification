import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

class CarsDataset(Dataset):
    """
    A custom PyTorch Dataset based on the Stanford Cars dataset.
    See ./scripts/prepare_dataset.py for details.
    Expected directory structure:
    <root>/
        └── cars_dataset/
            ├── train/
            │   ├── images/
            │   │   ├── 0/
            │   │   │   ├── 00001.jpg
            │   │   │   └── ...
            │   │   ├── 1/
            │   │   │   ├── 00001.jpg
            │   │   │   └── ...
            │   │   └── ...
            │   └── labels.npy
            └── test/
                ├── images/
                │   ├── 0/
                │   │   ├── 00001.jpg
                │   │   └── ...
                │   ├── 1/
                │   │   ├── 00001.jpg
                │   │   └── ...
                │   └── ...
                └── labels.npy

    Args:
        root (str): The top-level directory (e.g. 'data').
        split (str): Either 'train' or 'test'.
        transform (callable, optional): Transform to be applied to each image.
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.dataset_dir = os.path.join(self.root, 'cars_dataset', self.split)

        label_path = os.path.join(self.dataset_dir, 'labels.npy')
        self.labels = np.load(label_path, allow_pickle=True) 

        # json meta
        self.meta_path = os.path.join(self.root, 'cars_dataset', 'meta.json')
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                l = json.load(f)
                self.num_classes = l['total_number_of_classes']
                self.class_meta = l['classes']
        else:
            raise FileNotFoundError(f"Meta file {self.meta_path} not found.")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        class_id = item['class']
        fname = item['fname']
        img_path = os.path.join(self.dataset_dir, 'images', fname)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_id
    

