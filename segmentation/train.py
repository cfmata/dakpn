"""
Trains a model for semantic segmentation on (possibly translated) dataset
"""
import os
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from metrics import eval_metrics
from gta_untranslated_dataset import GTAUntranslatedDataset
from gta_translated_dataset import GTATranslatedDataset
from synthia_segmentation_dataset import SynthiaUntranslatedDataset
from cityscapes_segmentation_dataset import CityscapesSegDataset
from celeba_faceparsing_dataset import CelebA
from facesynthetics_segmentation_dataset import FaceSynthetics, TranslatedFaceSynthetics

# Hyperparameters
use_wandb = True
if use_wandb:
    wandb.init(project="kpn_vidseg")
num_classes = 19
resolution = [180, 180]
batch_size = 8
max_epochs = 50
epoch_save_rate = 1
iter_save_rate = 1000
weight_save_dir = "/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer_on_gta_translations"
model_type = 'segformer'

# Datasets
train_set = GTATranslatedDataset('train', resolution, 'da_kpn_source_enc')
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
print(len(train_set), " training")
val_set = CityscapesSegDataset('val', resolution)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        shuffle=False,
                        drop_last=True)
print(len(val_set), " validation")

if model_type == 'fcn':
    # Next block loads COCO pretrained FCN model
    model = fcn_resnet50(num_classes=num_classes, aux_loss=True)
    weights = torch.load('/home/cristinam/.cache/torch/hub/checkpoints/fcn_resnet50_coco-1167a1af.pth')
    del weights['classifier.4.weight']
    del weights['classifier.4.bias']
    del weights['aux_classifier.4.weight']
    del weights['aux_classifier.4.bias']
    model.load_state_dict(weights, strict=False)
    model = nn.DataParallel(model.cuda())
    optimizer = optim.SGD(model.parameters(), 0.004)
    lr_scheduler = None
else:
    # Initialize with Imagenet-pretrained backbone
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    model.decode_head.classifier = nn.Conv2d(768, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model = nn.DataParallel(model.cuda())
    base_lr = 0.00006 * (batch_size / 8)
    optimizer = optim.AdamW(model.parameters(), base_lr)
    lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max_epochs)

loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

for epoch in range(max_epochs):
    if epoch == 0:
        model.train()
    for i, data in enumerate(train_loader):
        images, labels, _ = data
        images, labels = images.cuda(), labels.long().cuda()
        out = model(images)
        if model_type == 'fcn':
            loss = loss_fn(out['out'], labels) + 0.4 * loss_fn(out['aux'], labels)
        else:
            loss = loss_fn(out.logits, labels)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_wandb:
            wandb.log({"Total Loss": loss})

        num_iter = epoch*len(train_loader) + i
        if num_iter % iter_save_rate == 0:
            torch.save(model.state_dict(), os.path.join(weight_save_dir, 'net_iter_{0}'.format(num_iter)))

    if epoch % epoch_save_rate == 0:
        torch.save(model.state_dict(), os.path.join(weight_save_dir, 'net_epoch_{0}'.format(epoch)))

    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for j, data in enumerate(val_loader):
            images, lb, _ = data
            if model_type == 'fcn':
                pred = model(images)['out']
            else:
                pred = model(images).logits
            pred = pred.argmax(dim=1)[0]
            preds.append(pred)
            labels.append(lb[0])
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
    # Step scheduler after each epoch
    lr_scheduler.step()

torch.save(model.state_dict(), os.path.join(weight_save_dir, 'net_epoch_{0}'.format(max_epochs)))