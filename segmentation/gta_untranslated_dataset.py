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
from torchvision.transforms.functional import to_pil_image

from cityscapes_segmentation_dataset import remap_labels_to_train_ids

class GTAUntranslatedDataset(data.Dataset):
    ''' Dataset for training on GTA images and testing on Cityscapes'''

    def __init__(self, mode, resolution):
        self.image_root = "/raid/datasets/GTA/images"
        self.labels_root = "/raid/datasets/GTA/labels"
        self.cityscapes_root = "/raid/datasets/cityscapes"
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        if self.mode == 'train':
            # Load all GTA images and labels
            images = [f for f in listdir(self.image_root)]
            labels = [join(self.labels_root, f) for f in images]
            images = [join(self.image_root, i) for i in images]
            for i in range(len(images)):
                self.files[self.mode].append({
                    "image": images[i],
                    "label": labels[i]
                })
        else:
            cities = listdir(join(self.cityscapes_root, 'gtFine', self.mode))
            for c in cities:
                all_files = [f for f in listdir(join(self.cityscapes_root, 'gtFine', self.mode, c)) if
                             'labelIds' in f]
                for f in all_files:
                    self.files[self.mode].append({
                        "image": join(self.cityscapes_root, 'leftImg8bit', self.mode, c, f[:-20] + '_leftImg8bit.png'),
                        "label": join(self.cityscapes_root, 'gtFine', self.mode, c, f)
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
        lb = remap_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]