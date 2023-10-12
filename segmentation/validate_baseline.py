"""
Calculate Deeplabv3 or FCN-score for either: 1. gta->cityscapes translations using Cityscapes-pretrained model or
2. FaceSynthetics->CelebA translations
Pretrained model taken from https://github.com/VainF/DeepLabV3Plus-Pytorch
"""
import sys
import time
import numpy as np
sys.path.append('/raid/cristinam/DeepLabV3Plus-Pytorch')
import network.modeling
import torch
from torch.utils.data import DataLoader
from modelscore_dataset import GTAModelScoreDataset, FaceSyntheticsModelScoreDataset
from metrics import eval_metrics
from torchvision.models.segmentation import fcn_resnet50

num_classes = 19
resolution= [512, 512] # height, width
data = "GTA" # 'FaceSynthetics' or 'GTA'
model_type = "fcn_resnet50" # 'deeplab' or 'fcn_resnet50'
adaptation = 'fastcut' # 'da_kpn', 'cyclegan', 'vsait', 'source', 'da_kpn_source_enc', 'da_kpn_source_noaffine', 'da_kpn_source_noblur', 'da_kpn_source_nonoise', 'cut', 'fastcut'

if data == "GTA":
    dataset = GTAModelScoreDataset(resolution, adaptation)
else:
    dataset = FaceSyntheticsModelScoreDataset(resolution, adaptation)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
print("Number validation images ", len(dataloader))

if model_type == "deeplab":
    # Load deeplabv3 mobilenet
    model = network.modeling.deeplabv3plus_mobilenet(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
    if data == "GTA":
        model.load_state_dict(torch.load('/raid/cristinam/DeepLabV3Plus-Pytorch/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')['model_state'])
    else:
        print("Need pretrained Deeplab weights for CelebA dataset")
        raise NotImplementedError
elif model_type == "fcn_resnet50":
    # Load fcn-resnet50 that we pretrained ourselves
    model = fcn_resnet50(num_classes=num_classes, aux_loss=True)
    if data == "GTA":
        weights = torch.load('/raid/cristinam/da_kpn_experiments/saved_models/segmentation/cityscapes_segmentation_targetonly/net_epoch_90')
    else:
        weights = torch.load('/raid/cristinam/da_kpn_experiments/saved_models/segmentation/faceparsing_targetonly_ignorebg/net_epoch_30')
    weights = {k[7:]:v for k,v in weights.items()}
    model.load_state_dict(weights, strict=True)
else:
    print("ERROR: model type ", model_type, " not recognized")
model.cuda()
model.eval()

print("model and dataset done")

val_preds = torch.zeros(len(dataloader), resolution[0], resolution[1])
val_labels = torch.zeros(len(dataloader), resolution[0], resolution[1])
print("created placeholders")
with torch.no_grad():
    t0 = time.time()
    for i, data in enumerate(dataloader):
        images, labels, _ = data
        images = images.cuda()
        if model_type == "fcn_resnet50":
            pred = model(images)['out']
        else:
            pred = model(images)
        val_preds[i] = torch.argmax(pred, dim=1)[0]
        val_labels[i] = labels[0]
        if i % 1000 == 0 and i > 0:
            print(i)
    val_preds = val_preds.cpu().numpy()
    #val_preds = val_preds.numpy()
    print("Calculating IOU")
    pxAcc, clsAccs, ious = eval_metrics(val_preds, val_labels.cpu().numpy(), num_classes=num_classes, ignore_index=255, nan_to_num=1)
    #pxAcc, clsAccs, ious = eval_metrics(val_preds, val_labels.numpy(), num_classes=num_classes, ignore_index=255, nan_to_num=1)
    t1 = time.time()
    print("Time for evaluation", t1 - t0)
    print("Pixel Accuracy ", pxAcc)
    print("Class Accuracy ", np.mean(clsAccs))
    print("Mean IOU", np.mean(ious))