import os
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
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from segmentation.cityscapes_segmentation_dataset import palette

from segmentation.cityscapes_segmentation_dataset import remap_labels_to_train_ids

class CityscapesGTADataset(data.Dataset):
    ''' Dataset for GTA->Cityscapes translation
    '''

    def __init__(self, mode, low_resolution, patch_scale_factor, encoder_type):
        self.city_root = "/raid/datasets/cityscapes"
        self.gta_root = "/raid/datasets/GTA"
        self.mode = mode
        assert self.mode in ["train", "val", "generate"]
        self.synth_files = collections.defaultdict(list)
        self.real_files = collections.defaultdict(list)
        self.low_resolution = low_resolution
        self.patch_scale_factor = patch_scale_factor
        self.encoder_type = encoder_type

        if self.mode == 'train':
            # Load cityscapes
            cities = listdir(join(self.city_root, 'gtFine', self.mode))
            for c in cities:
                all_files = [f for f in listdir(join(self.city_root, 'gtFine', self.mode, c)) if
                             'labelIds' in f]
                for f in all_files:
                    self.real_files[self.mode].append({
                        "image": join(self.city_root, 'leftImg8bit', self.mode, c, f[:-20] + '_leftImg8bit.png'),
                        "label": join(self.city_root, 'gtFine', self.mode, c, f)
                    })

        # Load GTA
        if self.mode == 'generate':
            # load all GTA images
            img_names = os.listdir(join(self.gta_root, 'images'))
            for img in img_names:
                self.synth_files[self.mode].append({
                    "image": join(self.gta_root, 'images', img),
                    "label": join(self.gta_root, 'labels', img)
                })
        else:
            ids = scipy.io.loadmat(join(self.gta_root, 'split.mat'))['{0}Ids'.format(self.mode)]
            for i in ids:
                self.synth_files[self.mode].append({
                    "image": join(self.gta_root, 'images', str(i[0]).zfill(5)+'.png'),
                    "label": join(self.gta_root, 'labels', str(i[0]).zfill(5)+'.png')
                })

        # Debug: visualize 70x70 crops
        self.visualize_crops = False

    def __len__(self):
        return len(self.synth_files[self.mode])

    def __getitem__(self, index):
        '''
        During training:
            return small res synthetic image, encoder input image, full res synthetic patch, patch_index, full res real patch, synth name, real name, (original image height, original image width)
        During validation and generation:
            return small res synthetic image, encoder input image, full res synthetic patches, synth name, (original image height, original image width)
        '''
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Synthetics
        synth_datafiles = self.synth_files[self.mode][index]
        synth_img = Image.open(synth_datafiles["image"]).convert('RGB')
        w, h = synth_img.size
        fold_params = {"kernel_size": (h//self.patch_scale_factor, w//self.patch_scale_factor),
                       "stride": (h//self.patch_scale_factor, w//self.patch_scale_factor)}
        low_res_synth_img = synth_img.resize(self.low_resolution[::-1])
        low_res_synth_img = img_transform(low_res_synth_img)
        if self.encoder_type == 'segformer':
            # Segformer spatially downsamples by 4. To get desired low-resolution output dims, upsample by 4
            encoder_img = synth_img.resize([4 * self.low_resolution[0], 4 * self.low_resolution[1]][::-1])
            encoder_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(encoder_img)
        else:
            encoder_img = low_res_synth_img
        full_res_synth_img = img_transform(synth_img)
        full_res_synth_patches = nn.Unfold(**fold_params)(full_res_synth_img.unsqueeze(0))


        synth_patch_index = np.random.randint(0, full_res_synth_patches.shape[-1])
        full_res_synth_patch = full_res_synth_patches[:, :, synth_patch_index].reshape(3,
                                                                                       fold_params['kernel_size'][0],
                                                                                       fold_params['kernel_size'][1])

        # Load synthetic labels
        lb = Image.open(synth_datafiles["label"]).convert('P')
        lb = lb.resize(self.low_resolution[::-1])
        lb = np.array(lb)
        lb = remap_labels_to_train_ids(lb)

        # Corresponding noise patches for transformation
        noise = torch.load('gaussian_3ch.pt')
        full_res_noise = F.interpolate(noise.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)
        full_res_noise_patches = nn.Unfold(**fold_params)(full_res_noise)
        full_res_noise_patch = full_res_noise_patches[:, :, synth_patch_index].reshape(3,
                                                                                       fold_params['kernel_size'][0],
                                                                                       fold_params['kernel_size'][1])
        low_res_noise = F.interpolate(torch.load('gaussian_3ch.pt').unsqueeze(0), self.low_resolution, mode='bilinear',
                                      align_corners=False)[0]

        if self.mode == 'train':
            # Real
            real_datafiles = self.real_files[self.mode][index % len(self.real_files[self.mode])]
            real_img = Image.open(real_datafiles["image"]).convert('RGB')
            real_img = img_transform(real_img)
            real_img_patches = nn.Unfold(**fold_params)(real_img.unsqueeze(0))
            real_patch_index = np.random.randint(0, real_img_patches.shape[-1])
            real_patch = real_img_patches[:, :, real_patch_index].reshape(3, fold_params['kernel_size'][0], fold_params['kernel_size'][1])

            if self.visualize_crops:
                to_pil_image(transforms.RandomCrop((70, 70))(transforms.ToTensor()(synth_img))).save('debug/synth_crop{}.png'.format(synth_datafiles['image'].split('/')[-1][:-4]))
                to_pil_image(transforms.RandomCrop((70, 70))(transforms.ToTensor()(Image.open(real_datafiles["image"]).convert('RGB')))).save('debug/real_crop{}.png'.format(real_datafiles['image'].split('/')[-1][:-4]))

            return low_res_synth_img, encoder_img, lb, full_res_synth_patch, full_res_noise_patch, low_res_noise, synth_patch_index, real_patch, synth_datafiles["image"], real_datafiles["image"], (h, w)
        else:
            return low_res_synth_img, encoder_img, lb, full_res_synth_patches, full_res_noise_patches, synth_datafiles["image"], (h, w)