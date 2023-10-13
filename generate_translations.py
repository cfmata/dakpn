import os
import time
import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from segmentation.metrics import eval_metrics
from cityscapes_dataset import CityscapesGTADataset
from celeba_dataset import CelebAFaceSynthetics
from segmentation.cityscapes_segmentation_dataset import palette
from kpn_model import DA_KPN, apply_affine_transformation, apply_noise_transformation, local_gaussian_blur

# Hyperparameters
weights = '/raid/cristinam/da_kpn_experiments/saved_models/ablation_frozen_source_encoder/net_iter_103000'
translations_folder = '/nfs/ws3/hdd2/cristinam/translations'
model_type = 'fcn'
#weights = '/raid/cristinam/da_kpn_experiments/faceparsing/da_kpn_no_encoder/net_iter_67000'
#translations_folder = '/raid/cristinam/da_kpn_experiments/faceparsing/da_kpn_no_encoder_iter_67000_translations_facesynthetics'
low_resolution = [96, 160]
patch_scale_factor = 8
num_params = 12
num_classes = 19

# Data
data_set = CityscapesGTADataset('generate', low_resolution, patch_scale_factor, model_type)
data_loader = DataLoader(data_set,
                        batch_size=1,
                        shuffle=False)

if model_type == 'fcn':
    feature_model = fcn_resnet50(num_classes=19, aux_loss=True)
    enc_weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_sourceonly/net_epoch_10")
    #enc_weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_targetonly/net_epoch_90")
    #enc_weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/faceparsing_targetonly_lowres/net_iter_57000")
elif model_type == 'segformer':
    # TODO: Convert to 19-class version
    feature_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    enc_weights = torch.load(
        "/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer/net_epoch_17")  # Source-trained encoder on GTA
enc_weights = {k[7:]: v for k, v in enc_weights.items()}
feature_model.load_state_dict(enc_weights, strict=True)
for param in feature_model.parameters():
    param.requires_grad = False
feature_model.cuda()

# Load model and weights
model = DA_KPN(num_params=num_params)
model.load_state_dict(torch.load(weights), strict=True)
model.eval()
model.cuda()
with torch.no_grad():
    for i, data in tqdm(enumerate(data_loader)):
        # data gives small res synthetic image, full res synthetic patches, synth image name
        low_res_synth_img, encoder_img, seg_lb, full_res_synth_patches, full_res_noise_patches, synth_name, (original_h, original_w) = data
        low_res_synth_img, encoder_img, seg_lb, full_res_synth_patches, full_res_noise_patches = low_res_synth_img.cuda(), encoder_img.cuda(), seg_lb.cuda(), full_res_synth_patches.cuda(), full_res_noise_patches.cuda()
        num_patches = full_res_synth_patches.shape[-1]

        if model_type == 'fcn':
            feats = feature_model(low_res_synth_img)['out'] # load features from pretrained model
        elif model_type == 'segformer':
            feats = feature_model(encoder_img)['logits']

        # get low res params
        params, features = model.kpn_forward(feats)

        # unfold params into patches
        low_res_fold_params = {"kernel_size": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor),
                               "stride": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor)}
        params_low_res_patches = nn.Unfold(**low_res_fold_params)(params).permute(0, 2, 1).reshape(1, num_patches, num_params, low_res_fold_params['kernel_size'][0], low_res_fold_params["kernel_size"][1])
        full_transformed = torch.zeros(3, original_h//patch_scale_factor, original_w//patch_scale_factor, num_patches)

        for j in range(num_patches):
            transformed_patch = apply_affine_transformation(params_low_res_patches[:, j, :, :, :],
                                                            full_res_synth_patches.reshape(1, 3, original_h//patch_scale_factor,
                                                                                           original_w//patch_scale_factor, -1)[:, :, :,
                                                            :, j])
            transformed_patch = local_gaussian_blur(transformed_patch, params_low_res_patches[:, j, :, :, :])
            transformed_patch = apply_noise_transformation(params_low_res_patches[:, j, :, :, :], transformed_patch,
                                                            full_res_noise_patches[:, :, :, j].reshape(1, 3, original_h//patch_scale_factor, original_w//patch_scale_factor))
            full_transformed[:, :, :, j] = transformed_patch[0]

        fold_transformed = nn.Fold(output_size=(original_h,original_w), kernel_size=(original_h//patch_scale_factor, original_w//patch_scale_factor), stride=(original_h//patch_scale_factor, original_w//patch_scale_factor))(full_transformed.reshape(1, -1, num_patches))
        if not os.path.exists(translations_folder):
            os.makedirs(translations_folder)
        to_pil_image(torch.clamp(fold_transformed[0], 0, 1)).save(os.path.join(translations_folder, synth_name[0].split('/')[-1]))