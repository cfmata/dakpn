import collections
import numpy as np
from PIL import Image, ImageFile
from os import listdir
from os.path import isfile, join

import torch.utils.data as data
from torchvision import transforms

from cityscapes_segmentation_dataset import remap_labels_to_train_ids
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GTATranslatedDataset(data.Dataset):
    '''
    Training set: GTA images translated into Cityscapes domain using CycleGAN or another method
    Validation set: 500 Cityscapes val images

    Cyclgan translations saved at /raid/cristinam/cyclegan_gta_translations/cityscapes_gta_cyclegan_id1/epoch_35/images
    DAKPN translations (frozen encoder ep 31) at /raid/cristinam/da_kpn_experiments/translations/da_kpn_no_encoder_ep31_translations_gta
    VSAIT translations saved at /raid/cristinam/vsait/checkpoints/gta_vsait_adapt/images
    '''

    def __init__(self, mode, resolution, dataset):
        if dataset == 'dakpn':
            # removed /raid/cristinam/da_kpn_experiments/translations/da_kpn_no_encoder_ep31_translations_gta/03139.png
            # also had trouble with 17165.png
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/da_kpn_no_encoder_ep31_translations_gta"
        elif dataset == 'da_kpn_source_enc':
            # ablation where encoder is trained on source dataset
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/da_kpn_source_encoder_iter103000_translations_gta"
        elif dataset == 'da_kpn_source_noaffine':
            # ablation where affine transform is removed
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/da_kpn_source_ablation_noaffine"
        elif dataset == 'da_kpn_source_noblur':
            # ablation where blur transform is remvoed
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/da_kpn_source_ablation_noblur"
        elif dataset == 'da_kpn_source_nonoise':
            # ablation where noise is remvoed
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/da_kpn_source_ablation_nonoise"
        elif dataset == 'cyclegan':
            self.translations_root = "/raid/cristinam/da_kpn_experiments/translations/cyclegan_gta_translations/cityscapes_gta_cyclegan_id1/epoch_35/images"
        elif dataset == 'vsait':
            self.translations_root = "/raid/cristinam/vsait/checkpoints/gta_vsait_adapt/images"
        else:
            print("Dataset ", dataset, " not implemented")
            raise NotImplementedException
        self.labels_root = "/raid/datasets/GTA/labels"
        self.cityscapes_root = "/raid/datasets/cityscapes"
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resolution = resolution
        self.files = collections.defaultdict(list)

        if self.mode == 'train':
            # Load all GTA translated images and labels
            #images = [f for f in listdir(self.translations_root) if f[-9:] == '_fake.png'] # use for cyclegan
            images = [f for f in listdir(self.translations_root)]
            #labels = [join(self.labels_root, f.split('_')[0]+'.png') for f in images] # use for cyclegan
            labels = [join(self.labels_root, f) for f in images]
            images = [join(self.translations_root, i) for i in images]
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
        img = img.resize((720, 720))
        img = img_transform(img)

        lb = Image.open(datafiles["label"]).convert('P')
        lb = lb.resize(self.resolution[::-1])
        lb = np.array(lb)
        lb = remap_labels_to_train_ids(lb)

        return img, lb, datafiles["image"]

