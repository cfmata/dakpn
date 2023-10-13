import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from segmentation.metrics import eval_metrics
from cityscapes_dataset import CityscapesGTADataset
from segmentation.cityscapes_segmentation_dataset import palette
from kpn_model import DA_KPN, apply_affine_transformation, apply_noise_transformation, local_gaussian_blur
from celeba_dataset import CelebAFaceSynthetics

# Hyperparameters
weight_save_dir = '/raid/cristinam/da_kpn_experiments/saved_models/segformer/'
model_type = 'segformer'
eval_mode = 'iter' # iter or epoch
checkpoints = list(range(0, 25000, 1000))
low_resolution = [96, 160]
patch_scale_factor = 8
# kernel_size = (90, 160)
# stride = (90, 160)
# fold_params = {"kernel_size": kernel_size,
#                "stride": stride,
#                }
num_params = 12
num_classes = 19
save_preds = True
save_features = True
save_parameters = True
eval_iou = False

# Data
val_set = CityscapesGTADataset('val', low_resolution, patch_scale_factor, model_type)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        shuffle=False)
print("Created dataset")
if model_type == 'fcn':
    feature_model = fcn_resnet50(num_classes=19, aux_loss=True)
elif model_type == 'segformer':
    # TODO: Convert to 19-class version
    feature_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")

#weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_targetonly/net_epoch_90")
#weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/faceparsing_targetonly_lowres/net_iter_57000")
weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer/net_epoch_17")
weights = {k[7:]: v for k, v in weights.items()}
feature_model.load_state_dict(weights, strict=True)
for param in feature_model.parameters():
    param.requires_grad = False
feature_model.cuda()
print("loaded model")
for ckpt in checkpoints:
    print("Checkpoint ", ckpt)
    # Load model and weights
    model = DA_KPN(num_params=num_params)
    model.load_state_dict(torch.load(os.path.join(weight_save_dir, 'net_{0}_{1}'.format(eval_mode, ckpt))), strict=True)
    model.eval()
    model.cuda()
    synth_disc_acc = 0
    if eval_iou:
        preds, labels = [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i > 1:
                break
            # data gives small res synthetic image, full res synthetic patches, synth image name
            low_res_synth_img, encoder_img, seg_lb, full_res_synth_patches, full_res_noise_patches, synth_name, (original_h, original_w) = data
            low_res_synth_img, encoder_img, seg_lb, full_res_synth_patches, full_res_noise_patches = low_res_synth_img.cuda(), encoder_img.cuda(), seg_lb.cuda(), full_res_synth_patches.cuda(), full_res_noise_patches.cuda()
            num_patches = full_res_synth_patches.shape[-1]
            if model_type == 'fcn':
                feats = feature_model(encoder_img)['out']
            elif model_type == 'segformer':
                feats = feature_model(encoder_img)['logits']
            # get low res params
            params, features = model.kpn_forward(feats)
            if eval_iou:
                pred = features.argmax(dim=1)[0]
                preds.append(pred)
                labels.append(seg_lb[0])

            # unfold params into patches
            low_res_fold_params = {"kernel_size": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor),
                                   "stride": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor)}
            params_low_res_patches = nn.Unfold(**low_res_fold_params)(params).permute(0, 2, 1).reshape(1, num_patches, num_params, low_res_fold_params['kernel_size'][0], low_res_fold_params["kernel_size"][1])
            full_transformed = torch.zeros(3, 90, 160, num_patches)

            for j in range(num_patches):
                transformed_patch = apply_affine_transformation(params_low_res_patches[:, j, :, :, :], full_res_synth_patches.reshape(1, 3, original_h//patch_scale_factor, original_w//patch_scale_factor, -1)[:, :, :, :, j])
                transformed_patch = local_gaussian_blur(transformed_patch, params_low_res_patches[:, j, :, :, :])
                transformed_patch = apply_noise_transformation(params_low_res_patches[:, j, :, :, :], transformed_patch,
                                                               full_res_noise_patches[:, :, :, j].reshape(1, 3, original_h//patch_scale_factor, original_w//patch_scale_factor))

                full_transformed[:, :, :, j] = transformed_patch[0]
                synth_disc_out = model.disc_forward(transformed_patch)
                synth_disc_acc += torch.sum(synth_disc_out < 0.5) / synth_disc_out.flatten().shape[0]
            if save_preds:
                # Save full transformed image
                fold_transformed = nn.Fold(output_size=(original_h, original_w), kernel_size=(original_h//patch_scale_factor, original_w//patch_scale_factor), stride=(original_h//patch_scale_factor, original_w//patch_scale_factor))(full_transformed.reshape(1, -1, num_patches))
                to_pil_image(torch.clamp(fold_transformed[0], 0, 1)).save(os.path.join('preds', '{0}_{1}_'.format(eval_mode, ckpt)+synth_name[0].split('/')[-1]))
            if save_features:
                # Save features from network
                features = features.argmax(dim=1)[0]
                output_np = features.byte().cpu().numpy()
                r = Image.fromarray(output_np)
                r.putpalette(palette)
                if r.mode != 'RGB':
                    r = r.convert('RGB')
                r.save('preds/features_{0}_{1}_'.format(eval_mode, ckpt)+synth_name[0].split('/')[-1])
            if save_parameters:
                print("Saving parameter")
                # Save parameter channels
                full_res_params = nn.Upsample((original_h, original_w), mode='bilinear', align_corners=False)(params)
                for p in range(num_params):
                    save_image(full_res_params[0, p, :, :], 'preds/param_{0}_{1}_{2}_'.format(eval_mode, ckpt, p)+synth_name[0].split('/')[-1])
    if eval_iou:
        preds = torch.stack(preds).cpu().numpy()
        labels = torch.stack(labels).cpu().numpy()
        print("Calculating IOU")
        pxAcc, clsAccs, ious = eval_metrics(preds, labels, num_classes=num_classes,
                                            ignore_index=255, nan_to_num=1)
        print("Pixel Accuracy ", pxAcc)
        print("Class Accuracy ", np.mean(clsAccs))
        print("Mean IOU", np.mean(ious))
    synth_disc_acc = synth_disc_acc / (len(val_loader) * patch_scale_factor**2)
    print("discriminator accuracy on synthetics validation ", eval_mode, ckpt, synth_disc_acc)