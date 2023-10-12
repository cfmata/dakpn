import os
import collections
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join, exists

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


def remap_celeba_labels_to_train_ids(arr):
    ignore_label = 255
    id2label = {0: ignore_label, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
                14: 14, 15: 15, 16: 16, 17: 17, 18: 18}
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

def remap_face_synthetics_labels_to_train_ids(arr):
    ignore_label = 255
    id2label = {0: ignore_label, 1: 1, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 9, 8: 8, 9: 10, 10: 11, 11: 12, 12: 17, 13: 13,
                14: 13, 15: 18, 16: 3, 17: 14, 18: ignore_label}
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

class CelebAFaceSynthetics(data.Dataset):
    ''' Dataset for DA-KPN Face Synthetics->CelebA translation
    '''

    def __init__(self, mode, full_resolution, low_resolution, patch_scale_factor, fold_params):
        self.celeb_root = "/raid/datasets/celebA/CelebAMask-HQ"
        self.face_root = "/homes/datasets/FaceSynthetics"
        self.mode = mode
        assert self.mode in ["train", "val", "generate"]
        self.synth_files = collections.defaultdict(list)
        self.real_files = collections.defaultdict(list)
        self.full_resolution = full_resolution
        self.low_resolution = low_resolution
        self.patch_scale_factor = patch_scale_factor
        self.fold_params = fold_params

        if self.mode == 'train':
            # Load real files during training
            ids = listdir(join(self.celeb_root, '{}_img'.format(self.mode)))
            for id in ids:
                img_name = join(self.celeb_root, '{}_img'.format(self.mode), str(int(id.split('.')[0])) + '.jpg')
                lb_name = join(self.celeb_root, '{}_label'.format(self.mode), str(int(id.split('.')[0])) + '.png')
                if exists(img_name) and exists(lb_name):
                    self.real_files[self.mode].append({
                        "image": img_name,
                        "label": lb_name
                    })

        # load all synthetics
        ids = [f for f in os.listdir(self.face_root) if len(f) == 10]
        for id in ids:
            self.synth_files[self.mode].append({
                "image": os.path.join(self.face_root, id),
                "label": os.path.join(self.face_root, id.split('.')[0] + '_seg.png')
            })

    def __len__(self):
        return len(self.synth_files[self.mode])

    def __getitem__(self, index):
        '''
        During training:
            return small res synthetic image, full res synthetic patch, patch_index, full res real patch, synth name, real name
        During validation and generation:
            return small res synthetic image, full res synthetic patches, synth name
        '''
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Synthetics
        synth_datafiles = self.synth_files[self.mode][index]
        synth_img = Image.open(synth_datafiles["image"]).convert('RGB')
        low_res_synth_img = synth_img.resize(self.low_resolution[::-1])
        low_res_synth_img = img_transform(low_res_synth_img)
        full_res_synth_img = synth_img.resize(self.full_resolution[::-1])
        full_res_synth_img = img_transform(full_res_synth_img)
        full_res_synth_patches = nn.Unfold(**self.fold_params)(full_res_synth_img.unsqueeze(0))
        synth_patch_index = np.random.randint(0, full_res_synth_patches.shape[-1])
        full_res_synth_patch = full_res_synth_patches[:, :, synth_patch_index].reshape(3, self.fold_params['kernel_size'][0], self.fold_params['kernel_size'][1])

        # Load synthetic labels
        lb = Image.open(synth_datafiles["label"]).convert('P')
        lb = lb.resize(self.low_resolution[::-1])
        lb = np.array(lb)
        lb = remap_face_synthetics_labels_to_train_ids(lb)

        # Corresponding noise patches for transformation
        noise = torch.load('gaussian_3ch.pt')
        full_res_noise = noise.resize(3, self.full_resolution[0], self.full_resolution[1])
        full_res_noise_patches = nn.Unfold(**self.fold_params)(full_res_noise.unsqueeze(0))
        full_res_noise_patch = full_res_noise_patches[:, :, synth_patch_index].reshape(3, self.fold_params['kernel_size'][0], self.fold_params['kernel_size'][1])
        low_res_noise = F.interpolate(torch.load('gaussian_3ch.pt').unsqueeze(0), self.low_resolution, mode='bilinear', align_corners=False)[0]

        if self.mode == 'train':
            # Real
            real_datafiles = self.real_files[self.mode][index % len(self.real_files[self.mode])]
            real_img = Image.open(real_datafiles["image"]).convert('RGB')
            real_img = real_img.resize(self.full_resolution[::-1])
            real_img = img_transform(real_img)
            real_img_patches = nn.Unfold(**self.fold_params)(real_img.unsqueeze(0))
            real_patch_index = np.random.randint(0, real_img_patches.shape[-1])
            real_patch = real_img_patches[:, :, real_patch_index].reshape(3, self.fold_params['kernel_size'][0], self.fold_params['kernel_size'][1])

            return low_res_synth_img, lb, full_res_synth_patch, full_res_noise_patch, low_res_noise, synth_patch_index, real_patch, synth_datafiles["image"], real_datafiles["image"]
        else:
            return low_res_synth_img, lb, full_res_synth_patches, full_res_noise_patches, synth_datafiles["image"]