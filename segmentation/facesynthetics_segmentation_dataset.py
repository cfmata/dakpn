import os
import collections
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Class conversion parameters
#BACKGROUND = 0
#SKIN = 1
#NOSE = 2
#RIGHT_EYE = 3
#LEFT_EYE = 4
#RIGHT_BROW = 5
#LEFT_BROW = 6
#RIGHT_EAR = 7
#LEFT_EAR = 8
#MOUTH_INTERIOR = 9
#TOP_LIP = 10
#BOTTOM_LIP = 11
#NECK = 12
#HAIR = 13
#BEARD = 14
#CLOTHING = 15
#GLASSES = 16
#HEADWEAR = 17
#FACEWEAR = 18
ignore_label = 255
facesynthetics_classes = ['skin', 'nose', 'right_eye', 'left_eye', 'right_brow']
id2label = {0: ignore_label, 1: 1, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 9, 8: 8, 9: 10, 10: 11, 11: 12, 12: 17, 13: 13, 14: 13, 15: 18, 16: 3, 17: 14, 18: ignore_label}
color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
celebA_classes = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

class FaceSynthetics(data.Dataset):
    def __init__(self, mode, resolution):
        self.root = "/raid/datasets/FaceSynthetics"
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        ids = [f for f in os.listdir(self.root) if len(f) == 10]
        ids.sort()
        if self.mode == 'train':
            ids = ids[:-10000]
        else:
            ids = ids[-10000:]
        for id in ids:
            self.files[self.mode].append({
                "image": os.path.join(self.root, id),
                "label": os.path.join(self.root, id.split('.')[0]+'_seg.png')
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

class TranslatedFaceSynthetics(data.Dataset):
    """Loads translations of all FaceSynthetics images for training and real celebA images for testing"""
    def __init__(self, mode, resolution, dataset):
        if dataset == 'facesynthetics':
            self.images_root = "/raid/datasets/FaceSynthetics"
        elif dataset == 'cyclegan':
            self.images_root = "/raid/cristinam/pytorch-CycleGAN-and-pix2pix/results/celeba_facesynthetics/test_latest/images"
        elif dataset == 'vsait':
            self.images_root = "/raid/cristinam/vsait/checkpoints/faceparsing/vsait_adapt/images"
        elif dataset == 'dakpn':
            self.images_root = "/raid/cristinam/da_kpn_experiments/faceparsing/da_kpn_no_encoder_iter_67000_translations_facesynthetics"
        else:
            print("Dataset ", dataset, " not implemented")
            raise NotImplementedException
        self.labels_root = "/raid/datasets/FaceSynthetics"
        self.celeba_root = "/raid/datasets/celebA/CelebAMask-HQ"
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        if mode == 'train':
            # load synthetics
            if dataset == 'cyclegan':
                ids = [f for f in os.listdir(self.images_root) if f[-9:] == "_fake.png"]
                for id in ids:
                    self.files[self.mode].append({
                        "image": os.path.join(self.images_root, id),
                        "label": os.path.join(self.labels_root, id.split('_')[0] + '_seg.png')
                    })
            else:
                ids = os.listdir(self.images_root)
                for id in ids:
                    self.files[self.mode].append({
                        "image": os.path.join(self.images_root, id),
                        "label": os.path.join(self.labels_root, id[:-4] + '_seg.png')
                    })
        elif mode == 'val':
            # load real for testing
            ids = listdir(join(self.celeba_root, '{}_img'.format(self.mode)))
            for id in ids:
                img_name = join(self.celeba_root, '{}_img'.format(self.mode), str(int(id.split('.')[0])) + '.jpg')
                lb_name = join(self.celeba_root, '{}_label'.format(self.mode), str(int(id.split('.')[0])) + '.png')
                if os.path.exists(img_name) and os.path.exists(lb_name):
                    self.files[self.mode].append({
                        "image": img_name,
                        "label": lb_name
                    })

    def __len__(self):
        return len(self.files[self.mode])

    def remap_facesynthetics_labels_to_train_ids(self, arr):
        ignore_label = 255
        id2label = {0: ignore_label, 1: 1, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 9, 8: 8, 9: 10, 10: 11, 11: 12, 12: 17,
                    13: 13, 14: 13, 15: 18, 16: 3, 17: 14, 18: ignore_label}
        out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
        for id, label in id2label.items():
            out[arr == id] = int(label)
        return out

    def remap_celeba_labels_to_train_ids(self, arr):
        ignore_label = 255
        id2label = {0: ignore_label, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                    13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}
        out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
        for id, label in id2label.items():
            out[arr == id] = int(label)
        return out

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
        if self.mode == 'train':
            lb = self.remap_facesynthetics_labels_to_train_ids(lb)
        elif self.mode == 'val':
            lb = self.remap_celeba_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]