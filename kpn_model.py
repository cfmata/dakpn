import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image

from discriminators import NLayerDiscriminator, grad_reverse

def rgb_to_hsv(img):
    """Convert an RGB image to HSV color space
    Reference: https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py
    """
    #M, argM = torch.max(img, dim=0), torch.argmax(img, dim=0)
    #m = torch.min(img, dim=0)
    #C = torch.abs(M - m)
    #assert C != 0
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).cuda()
    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + 1e-7))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + 1e-7))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + 1e-7))[
        img[:, 0] == img.max(1)[0]]) % 6
    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + 1e-7)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]
    return torch.stack([hue, saturation, value], dim=1)

def hsv_to_rgb(img):
    """Convert HSV image to RGB color space
    Reference: https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py
    """
    C = img[:, 2, :, :] * img[:, 1, :, :]
    X = C * (1 - torch.abs((img[:, 0, :, :] * 6) % 2 - 1))
    m = img[:, 2, :, :] - C

    R_hat = C * (img[:, 0, :, :] < 1/6).int().float()
    R_hat += X * ((1/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 2/6)).int().float()
    R_hat += 0 * ((2/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 3/6)).int().float()
    R_hat += 0 * ((3/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 4/6)).int().float()
    R_hat += X * ((4/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 5/6)).int().float()
    R_hat += C * ((5/6 < img[:, 0, :, :]) & (img[:, 0, :, :] <= 6/6)).int().float()

    G_hat = X * (img[:, 0, :, :] < 1/6).int().float()
    G_hat += C * ((1/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 2/6)).int().float()
    G_hat += C * ((2/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 3/6)).int().float()
    G_hat += X * ((3/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 4/6)).int().float()
    G_hat += 0 * ((4/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 5/6)).int().float()
    G_hat += 0 * ((5/6 < img[:, 0, :, :]) & (img[:, 0, :, :] <= 6/6)).int().float()

    B_hat = 0 * (img[:, 0, :, :] < 1/6).int().float()
    B_hat += 0 * ((1/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 2/6)).int().float()
    B_hat += X * ((2/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 3/6)).int().float()
    B_hat += C * ((3/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 4/6)).int().float()
    B_hat += C * ((4/6 < img[:, 0, :, :]) & (img[:, 0, :, :] < 5/6)).int().float()
    B_hat += X * ((5/6 < img[:, 0, :, :]) & (img[:, 0, :, :] <= 6/6)).int().float()

    R, G, B = (R_hat + m), (G_hat + m), (B_hat + m)

    return torch.stack([R, G, B], dim=1)

def apply_affine_transformation(low_res_params, patches):
    _, _, patch_h, patch_w = patches.shape
    # resize low res params to same size as patches
    params = nn.Upsample((patch_h, patch_w), mode='bilinear', align_corners=False)(low_res_params)

    # convert patches to hsv color space
    hsv_patch = rgb_to_hsv(patches)
    # affine color transformation
    color_weights, color_bias = params[:, :3, :, :], params[:, 3:6, :, :]
    color_transformed = color_weights * hsv_patch + color_bias
    color_transformed = hsv_to_rgb(color_transformed)
    return color_transformed

def apply_blur_transformation(low_res_params, patch):
    # resize low res params
    params = nn.Upsample((90, 160), mode='bilinear', align_corners=False)(low_res_params)
    b, c, h, w = patch.shape
    # construct gaussian blur kernel
    sigma = params[:, 6:9, :, :]
    kernel = torch.stack([torch.stack(
        [(1 / (2 * math.pi * sigma ** 2)) * math.e ** ((-(i ** 2 + j ** 2) / (2 * sigma ** 2))) for i in range(25)]) for
                          j in range(25)])
    kernel = kernel.permute(2, 3, 0, 1, 4, 5).reshape(b, c, 25, 25, -1)

    # Unfold synth patch into 25x25
    fold_params = {'kernel_size': (25, 25),
                   'stride': 1,
                   }
    unfolded = nn.Unfold(**fold_params)(F.pad(patch, (12, 12, 12 ,12), mode='replicate')).reshape(b, c, 25, 25, -1)
    # apply weights
    blurred = kernel*unfolded
    # sum dy
    sum1 = blurred[:, :, :, 0, :] + blurred[:, :, :, 1, :] + blurred[:, :, :, 2, :] + blurred[:, :, :, 3, :] + \
    blurred[:, :, :, 4, :] + blurred[:, :, :, 5, :] + blurred[:, :, :, 6, :] + blurred[:, :, :, 7, :] + \
    blurred[:, :, :, 8, :] + blurred[:, :, :, 9, :] + blurred[:, :, :, 10, :] + blurred[:, :, :, 11, :] + \
    blurred[:, :, :, 12, :] + blurred[:, :, :, 13, :] + blurred[:, :, :, 14, :] + blurred[:, :, :, 15, :] + \
    blurred[:, :, :, 16, :] + blurred[:, :, :, 17, :] + blurred[:, :, :, 18, :] + blurred[:, :, :, 19, :] + \
    blurred[:, :, :, 20, :] + blurred[:, :, :, 21, :] + blurred[:, :, :, 22, :] + blurred[:, :, :, 23, :] + \
    blurred[:, :, :, 24, :]
    # sum dx
    sum2 = sum1[:, :, 0, :] + sum1[:, :, 1, :] + sum1[:, :, 2, :] + sum1[:, :, 3, :] + sum1[:, :, 4, :] + \
    sum1[:, :, 5, :] + sum1[:, :, 6, :] + sum1[:, :, 7, :] + sum1[:, :, 8, :] + sum1[:, :, 9, :] + sum1[:, :, 10, :] + \
    sum1[:, :, 11, :] + sum1[:, :, 12, :] + sum1[:, :, 13, :] + sum1[:, :, 14, :] + sum1[:, :, 15, :] + sum1[:, :, 17,
                                                                                                        :] + \
    sum1[:, :, 18, :] + sum1[:, :, 19, :] + sum1[:, :, 20, :] + sum1[:, :, 21, :] + sum1[:, :, 22, :] + sum1[:, :, 23,
                                                                                                        :] + \
    sum1[:, :, 24, :]
    sum2 = sum2.reshape(b, 3, 90, 160)

    #recovery_mask = nn.Fold(output_size=(90, 160), **fold_params)(nn.Unfold(**fold_params)(torch.ones(1, 3, 90, 160).cuda()))
    #sum2 /= recovery_mask
    #return sum2
    return nn.Fold(output_size=(90+24, 160+24), kernel_size=25, stride=1)(blurred.reshape(1, -1, 14400))


def gaussian_kernels(stds, size=25):
    """ Takes a series of std values of length N
        and integer size corresponding to kernel side length M
        and returns a set of gaussian kernels with those stds in a (N,M,M) tensor

    Args:
        stds (Tensor): Flat list tensor containing std values.
        size (int): Size of the Gaussian kernel.
    Returns:
        Tensor: Tensor containing a unique 2D Gaussian kernel for each value in the stds input.
    """
    # 1. create input vector to the exponential function
    n = (torch.arange(0, size) - (size - 1.0) / 2.0).unsqueeze(-1).cuda()
    var = 2 * (stds ** 2).unsqueeze(-1) + 1e-8  # add constant for stability

    # 2. compute gaussian values with exponential
    kernel_1d = torch.exp((-n ** 2) / var.t()).permute(1, 0)
    # 3. outer product in a batch
    kernel_2d = torch.bmm(kernel_1d.unsqueeze(2), kernel_1d.unsqueeze(1))
    # 4. normalize to unity sum
    kernel_2d /= kernel_2d.sum(dim=(-1, -2)).view(-1, 1, 1)

    return kernel_2d

def local_gaussian_blur(input, low_res_params, kernel_size=25):
    """Blurs image with dynamic Gaussian blur.
    https://github.com/mikonvergence/LocalGaussianBlur/blob/main/src/LocalGaussianBlur.py
    Args:
        input (Tensor): The image to be blurred (C,H,W).
        modulator (Tensor): The modulating signal that determines the local value of kernel variance (H,W).
        kernel_size (int): Size of the Gaussian kernel.
    Returns:
        Tensor: Locally blurred version of the input image.
    """
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    b, c, h, w = input.shape
    pad = int((kernel_size - 1) / 2)
    modulator = nn.Upsample((h, w), mode='bilinear', align_corners=False)(low_res_params)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad, pad, pad, pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    #'''
    kernel_r = gaussian_kernels(modulator[:, 6, :, :].flatten())
    kernel_g = gaussian_kernels(modulator[:, 7, :, :].flatten())
    kernel_b = gaussian_kernels(modulator[:, 8, :, :].flatten())
    kernels_rgb = torch.stack([kernel_r, kernel_g, kernel_b], 1).reshape(b, h*w, c, -1)
    #'''
    '''
    kernel = gaussian_kernels(modulator[:, 6, :, :].flatten())
    kernels_rgb = torch.stack(3*[kernel], 1).reshape(b, h*w, c, -1)
    '''
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size, kernel_size))
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b, c, -1, h * w)
    x2 = kernels_rgb.view(b, h * w, c, -1).permute(0, 2, 3, 1)
    y = (x1 * x2).sum(2)
    # 5. Fold and return
    return torch.nn.functional.fold(y, (h, w), (1, 1))

def apply_noise_transformation(low_res_params, patch, noise):
    _, _, patch_h, patch_w = patch.shape
    # resize low res params to same size as patches
    params = nn.Upsample((patch_h, patch_w), mode='bilinear', align_corners=False)(low_res_params)

    # scale and add gaussian noise
    noise = params[:, 9:12, :, :] * noise
    noisy = torch.clamp(patch + noise, min=0, max=1)
    #noisy = torch.sigmoid(patch + noise)
    return noisy

class DA_KPN(nn.Module):
    def __init__(self, num_params):
        super(DA_KPN, self).__init__()
        #self.encoder = torchvision.models.segmentation.deeplabv3_resnet101()
        #self.encoder.load_state_dict(
        #    torch.load('/raid/cristinam/pretrained_models/deeplabv3_resnet101_coco-586e9e4e.pth'),
        #    strict=False)

        #self.encoder = fcn_resnet50(num_classes=19, aux_loss=True)
        #weights = torch.load("/raid/cristinam/da_kpn_experiments/segmentation/cityscapes_segmentation_targetonly/net_epoch_100")
        #weights = {k[7:]: v for k, v in weights.items()}
        #self.encoder.load_state_dict(weights, strict=True)

        # TODO: Change to 19 classes instead of 1000
        self.kpn = nn.Conv2d(19, num_params, 1)
        #self.kpn = nn.Conv2d(1000, num_params, 1)

        # Initial KPN params should give identity
        nn.init.constant_(self.kpn.weight.data, 0)
        nn.init.constant_(self.kpn.bias.data[:3], 1)
        nn.init.constant_(self.kpn.bias.data[3:6], 0)
        nn.init.constant_(self.kpn.bias.data[6:9], 1)
        nn.init.constant_(self.kpn.bias.data[9:12], 0)
        self.discriminator = NLayerDiscriminator(3)

    def kpn_forward(self, features):#x):
        #features = self.encoder(x)['out']
        parameters = self.kpn(features)
        return parameters, features

    def disc_forward(self, x):
        reverse_feature = grad_reverse(x)
        disc_out = self.discriminator(reverse_feature)
        return disc_out