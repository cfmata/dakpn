import os
import time
import wandb
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from kpn_model import DA_KPN, apply_affine_transformation, apply_noise_transformation, local_gaussian_blur
from cityscapes_dataset import CityscapesGTADataset
from segmentation.metrics import eval_metrics
from segmentation.cityscapes_segmentation_dataset import palette
from celeba_dataset import CelebAFaceSynthetics

# Hyperparameters
use_wandb = False
if use_wandb:
    wandb.init(project="kpn_vidseg")
max_epochs = 100
weight_save_dir = '/raid/cristinam/da_kpn_experiments/saved_models/segformer'
epoch_save_rate = 1
iter_save_rate = 1000
low_resolution = [96, 160]
patch_scale_factor = 8
num_params = 12
batch_size = 8
num_classes = 19
encoder_type = 'segformer'
visualize_train = False
debug_dir = '/homes/cristinam/image_translation/debug'
save_preds = False
pred_dir = '/homes/cristinam/image_translation/preds'

# Dataset loading
train_set = CityscapesGTADataset('train', low_resolution, patch_scale_factor, encoder_type)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
val_set = CityscapesGTADataset('val', low_resolution, patch_scale_factor, encoder_type)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        shuffle=False)

# Load model and weights
model = DA_KPN(num_params=num_params)
model.cuda()
if use_wandb:
    wandb.watch(model)

# Load frozen pretrained model for semantic feature identity
if encoder_type == 'fcn':
    feature_model = fcn_resnet50(num_classes=19, aux_loss=True)
    #weights = torch.load("/raid/cristinam/da_kpn_experiments/segmentation/faceparsing_targetonly_lowres/net_iter_57000")
    #weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_targetonly/net_epoch_90")
    weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_sourceonly/net_epoch_10")
    weights = {k[7:]: v for k, v in weights.items()}
    feature_model.load_state_dict(weights, strict=True)
else:
    # TODO: Convert to 19-class version
    feature_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    weights = torch.load("/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer/net_epoch_17") # Source-trained encoder on GTA
    #weights = torch.load(
    #    "/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer_synthia/net_epoch_2")  # Source-trained encoder on Synthia
    weights = {k[7:]: v for k,v in weights.items()}
    feature_model.load_state_dict(weights, strict=True)
for param in feature_model.parameters():
    param.requires_grad = False
feature_model.cuda()


# Optimizer and loss
optimizer = optim.Adam(list(model.kpn.parameters()), lr=0.0001)
disc_optimizer = optim.Adam(list(model.discriminator.parameters()), lr=0.0001)
id_loss_fn = nn.MSELoss()
disc_loss_fn = nn.BCEWithLogitsLoss()
seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

for epoch in range(max_epochs):
    model.train()
    for i, data in enumerate(train_loader):
        low_res_synth_img, encoder_img, seg_lb, full_res_synth_patch, full_res_noise_patch, low_res_noise, synth_patch_index, real_patch, synth_name, real_name, (original_h, original_w) = data
        low_res_synth_img, encoder_img, seg_lb, full_res_synth_patch, full_res_noise_patch, low_res_noise = low_res_synth_img.cuda(), encoder_img.cuda(), seg_lb.long().cuda(), full_res_synth_patch.cuda(), full_res_noise_patch.cuda(), low_res_noise.cuda()

        if encoder_type == 'fcn':
            feats = feature_model(encoder_img)['out']
        else:
            feats = feature_model(encoder_img)['logits']

        # get low res params
        params, features = model.kpn_forward(feats)
        # unfold params into patches
        low_res_fold_params = {"kernel_size": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor),
                               "stride": (low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor)}
        params_low_res_patch = nn.Unfold(**low_res_fold_params)(params)
        params_low_res_patch = torch.stack([params_low_res_patch[i, :, synth_patch_index[i]] for i in range(batch_size)]).reshape(batch_size, num_params, low_resolution[0] // patch_scale_factor, low_resolution[1] // patch_scale_factor)

        # apply full res patch transformation
        transformed = apply_affine_transformation(params_low_res_patch, full_res_synth_patch)
        transformed = local_gaussian_blur(transformed, params_low_res_patch)
        transformed = apply_noise_transformation(params_low_res_patch, transformed, full_res_noise_patch)

        # get discriminator results
        disc_out_synth = model.disc_forward(transformed)
        disc_out_real = model.disc_forward(real_patch.cuda())

        # Multiscale: apply low res whole image transformation
        transformed_whole = apply_affine_transformation(params, low_res_synth_img)
        transformed_whole = local_gaussian_blur(transformed_whole, params)
        transformed_whole = apply_noise_transformation(params, transformed_whole, low_res_noise)
        disc_out_synth_whole = model.disc_forward(transformed_whole)

        optimizer.zero_grad()
        disc_optimizer.zero_grad()

        # Transformation identity loss
        identity_loss = id_loss_fn(transformed, full_res_synth_patch)
        print("identity loss ", identity_loss)
        if use_wandb:
            wandb.log({"Color Identity Loss": identity_loss})

        '''
        # Feature identity loss
        feature_loss = id_loss_fn(features, feature_model(low_res_synth_img)['out'].cuda())
        print("feature loss ", feature_loss)
        if use_wandb:
            wandb.log({"Feature Identity Loss": feature_loss})
        '''

        '''
        # Segmentation loss
        seg_loss = seg_loss_fn(features, seg_lb)
        print("segmentation loss ", seg_loss)
        if use_wandb:
            wandb.log({"Segmentation Loss": seg_loss})
        '''

        # Discriminator losses
        synth_disc_loss = disc_loss_fn(disc_out_synth, torch.zeros(disc_out_synth.shape).cuda()) + disc_loss_fn(disc_out_synth_whole, torch.zeros(disc_out_synth_whole.shape).cuda())
        real_disc_loss = disc_loss_fn(disc_out_real, torch.ones(disc_out_real.shape).cuda())
        disc_loss = synth_disc_loss + real_disc_loss
        print("discriminator loss ", disc_loss)
        synth_disc_patch_acc = torch.sum(disc_out_synth <= 0.5) / disc_out_synth.flatten().shape[0]
        synth_disc_whole_acc = torch.sum(disc_out_synth_whole <= 0.5) / disc_out_synth_whole.flatten().shape[0]
        real_disc_acc = torch.sum(disc_out_real > 0.5) / disc_out_real.flatten().shape[0]
        print("discriminator acc synth ", synth_disc_patch_acc, " whole synth ", synth_disc_whole_acc, " real ", real_disc_acc)
        if use_wandb:
            wandb.log({"Synthetic Discriminator Loss": synth_disc_loss,
                       "Real Discriminator Loss": real_disc_loss,
                       "Synthetic Discriminator Acc": synth_disc_patch_acc,
                       "Synthetic Discriminator Whole Acc": synth_disc_whole_acc,
                       "Real Discriminator Acc": real_disc_acc})

        total_loss = identity_loss + disc_loss #+ seg_loss #+ feature_loss
        total_loss.backward()
        optimizer.step()
        disc_optimizer.step()

        num_iter = epoch*len(train_loader) + i
        if num_iter % iter_save_rate == 0:
            torch.save(model.state_dict(), os.path.join(weight_save_dir, 'net_iter_{0}'.format(num_iter)))
            '''
            model.eval()
            with torch.no_grad():
                preds, labels = [], []
                for j, data in enumerate(val_loader):
                    if j % 1000 == 0:
                        print("Validating ", j)
                    low_res_synth_img, seg_lb, _, _, _ = data
                    low_res_synth_img, seg_lb = low_res_synth_img.cuda(), seg_lb.long().cuda()
                    feats = feature_model(low_res_synth_img)['out']
                    _, pred = model.kpn_forward(feats)
                    pred = pred.argmax(dim=1)[0]
                    preds.append(pred)
                    labels.append(seg_lb[0])
                    
                    # This section saves features
                    r = to_pil_image(pred.byte().cpu().numpy())
                    r.putpalette(palette)
                    if r.mode != 'RGB':
                        r = r.convert('RGB')
                    r.save('preds/pred_{0}.png'.format(j))
                    l = to_pil_image(seg_lb[0].byte().cpu().numpy())
                    l.putpalette(palette)
                    if l.mode != 'RGB':
                        l = l.convert('RGB')
                    l.save('preds/lb_{0}.png'.format(j))
                    
                preds = torch.stack(preds).cpu().numpy()
                labels = torch.stack(labels).cpu().numpy()
                print("Calculating IOU")
                pxAcc, clsAccs, ious = eval_metrics(preds, labels, num_classes=num_classes,
                                                    ignore_index=255, nan_to_num=1)
                print("Pixel Accuracy ", pxAcc)
                print("Class Accuracy ", np.mean(clsAccs))
                print("Mean IOU", np.mean(ious))
                if use_wandb:
                    wandb.log({"IOU": np.mean(ious)})
            model.train()
            '''

    if epoch % epoch_save_rate == 0:
        torch.save(model.state_dict(), os.path.join(weight_save_dir, 'net_epoch_{0}'.format(epoch)))

if use_wandb:
    wandb.finish()