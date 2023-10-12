import scipy.io
import collections
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# Class conversion parameters (same as Cityscapes)
ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
           'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle']


def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

class SynthiaUntranslatedDataset(data.Dataset):
    ''' Dataset for training and testing on Cityscapes'''

    def __init__(self, mode, resolution):
        self.synthia_root = "/raid/datasets/synthia/RAND_CITYSCAPES"
        assert mode == 'train'
        self.mode = mode
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        images = listdir(join(self.synthia_root, 'RGB'))
        for img in images:
            self.files[mode].append({
                "image": join(self.synthia_root, 'RGB', img),
                "label": join(self.synthia_root, 'GT', 'LABELS', img[:-4] + '_labelTrainIds.png')
            })

    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        datafiles = self.files[self.mode][index]
        img = Image.open(datafiles["image"]).convert('RGB')
        #img = img.resize(self.resolution[::-1])
        img = img.resize((896, 896))
        img = img_transform(img)

        lb = Image.open(datafiles["label"]).convert('P')
        lb = lb.resize(self.resolution[::-1])
        lb = np.array(lb)
        #lb = remap_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]