import collections
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join, exists

import torch.utils.data as data
from torchvision import transforms

ignore_label = 255
id2label = {0: ignore_label, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}
color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
celebA_classes = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

class CelebA(data.Dataset):
    def __init__(self, mode, resolution):
        self.root = "/raid/datasets/celebA/CelebAMask-HQ"
        self.mode = mode
        assert self.mode in ["train", "val", "test"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        ids = listdir(join(self.root, '{}_img'.format(self.mode)))
        for id in ids:
            img_name = join(self.root, '{}_img'.format(self.mode), str(int(id.split('.')[0]))+'.jpg')
            lb_name = join(self.root, '{}_label'.format(self.mode), str(int(id.split('.')[0]))+'.png')
            if exists(img_name) and exists(lb_name):
                self.files[self.mode].append({
                    "image": img_name,
                    "label": lb_name
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
        img = img.resize(self.resolution[::-1])
        img = img_transform(img)

        lb = Image.open(datafiles["label"]).convert('P')
        lb = lb.resize(self.resolution[::-1])
        lb = np.array(lb)
        lb = remap_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]