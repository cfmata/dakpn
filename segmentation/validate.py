import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from metrics import eval_metrics
from cityscapes_segmentation_dataset import CityscapesSegDataset, palette
from celeba_faceparsing_dataset import CelebA
from facesynthetics_segmentation_dataset import FaceSynthetics

# Hyperparameters
num_classes = 19
resolution = [180, 180] #[720, 1280]
save_preds = False
model_type = 'segformer' # segformer or fcn
ckpt_type = 'epoch'
ckpts = list(range(2, 3))
weight_save_dir = "/raid/cristinam/da_kpn_experiments/saved_models/segmentation/segformer_synthia"
print("Evaluating weights from ", weight_save_dir)
# Dataset
val_set = CityscapesSegDataset('val', resolution)
val_loader = DataLoader(val_set,
                        batch_size=1,
                        shuffle=False,
                        drop_last=True)

for ckpt in ckpts:
    # Load model
    if model_type == 'fcn':
        model = fcn_resnet50(num_classes=num_classes, aux_loss=True)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    weights = torch.load(os.path.join(weight_save_dir, 'net_{0}_{1}'.format(ckpt_type, ckpt)))
    weights = {k[7:]: v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)

    model.cuda()
    model.eval()
    with torch.no_grad():
        #preds, labels = torch.zeros(2993, 512,512), torch.zeros(2993, 512, 512)
        preds, labels = torch.zeros(len(val_set), resolution[0], resolution[1]), torch.zeros(len(val_set), resolution[0], resolution[1])
        for j, data in enumerate(val_loader):
            images, lb, name = data
            images, lb = images.cuda(), lb.cuda()
            if model_type == 'fcn':
                pred = model(images)['out']
            else:
                outputs = model(images)
                pred = outputs.logits
            pred = pred.argmax(dim=1)[0]
            preds[j] = pred

            #lb[lb == 255] = 0
            #r = Image.fromarray(lb[0].byte().cpu().numpy())
            #r.putpalette(palette)
            #if r.mode != 'RGB':
            #    r = r.convert('RGB')
            #r.save('celeba_lb/{0}.png'.format(name[0].split('/')[-1][:-4]))

            if save_preds:
                if not os.path.exists('preds/{0}_{1}'.format(ckpt_type, ckpt)):
                    os.makedirs('preds/{0}_{1}'.format(ckpt_type, ckpt))
                output_np = pred.byte().cpu().numpy()
                r = Image.fromarray(output_np)
                r.putpalette(palette)
                if r.mode != 'RGB':
                    r = r.convert('RGB')
                r.save('preds/{0}_{1}/{2}.png'.format(ckpt_type, ckpt, name[0].split('/')[-1][:-4]))
            labels[j] = lb[0]

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        print("Calculating IOU at {0} {1}".format(ckpt_type, ckpt))
        pxAcc, clsAccs, ious = eval_metrics(preds, labels, num_classes=num_classes,
                                            ignore_index=255, nan_to_num=1)
        print("Pixel Accuracy ", pxAcc)
        print("Class Accuracy ", np.mean(clsAccs))
        print("Mean IOU", np.mean(ious))
        #print("All class IOUs ", ious)