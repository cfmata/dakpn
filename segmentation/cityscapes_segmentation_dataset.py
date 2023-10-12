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

# Class conversion parameters
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

class CityscapesSegDataset(data.Dataset):
    ''' Dataset for training and testing on Cityscapes'''

    def __init__(self, mode, resolution):
        self.cityscapes_root = "/raid/datasets/cityscapes"
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

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
        img = img.resize((720, 720))
        img = img_transform(img)

        lb = Image.open(datafiles["label"]).convert('P')
        lb = lb.resize(self.resolution[::-1])
        lb = np.array(lb)
        lb = remap_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]

def class_percentage_pixels(resolution):
    """
    Counts percentage of pixels belonging to each class in the Cityscapes val dataset
    @param resolution in [height, width]
    """
    val_set = CityscapesSegDataset('val', resolution)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            shuffle=False,
                            drop_last=True)
    class_to_percentage = dict({"255": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0})
    for i, data in enumerate(val_loader):
        _, lb, name = data
        unique_classes = np.unique(lb)
        for cl in unique_classes:
            class_to_percentage[str(cl)] += torch.sum(lb == cl)
    tot_pixels = sum(class_to_percentage.values())
    for k,v in class_to_percentage.items():
        class_to_percentage[k] = class_to_percentage[k] / tot_pixels
        class_to_percentage[k] *= 100
    print(class_to_percentage)
    assert sum(class_to_percentage[k] for k in class_to_percentage.keys()) == 100

if __name__ == "__main__":
    class_percentage_pixels(resolution=[512, 512])