#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from models.resnet_fpn import fpn
from utils.metrics import ConfusionMatrix
from PIL import Image
import os

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
    transforms.ToTensor(),
])

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1

    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(int(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]    #height
    y = shape[1]    #width
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def global2patch(images, labels_glb, p_size, ids):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(ids); patch_ones = np.ones(p_size); label_patches = []
    for i in range(len(ids)):
        image_info = os.listdir(os.path.join('data/patches', ids[i]))[0].split('_')
        patches_list = os.listdir(os.path.join('data/patches', ids[i]))
        patches_list.sort()
        h, w = int(image_info[-2]), int(image_info[-1].split('.')[0])

        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])

        patches.append([None] * (len(patches_list)))
        label_patches.append([None] * (len(patches_list)))
        coordinates.append([(0, 0)] * (len(patches_list)))

        for j, patch_name in enumerate(patches_list):
            x, y = int(patch_name.split('_')[-4]), int(patch_name.split('_')[-3])
            patch = Image.open(os.path.join('data/patches', ids[i], patch_name))
            coordinates[i][j] = (x, y)
            patches[i][j] = transforms.functional.crop(patch, 0, 0, p_size[0], p_size[1])
            label_patches[i][j] = labels_glb[i]

    return patches, coordinates, sizes, ratios, label_patches

def patch2global(patches, n_class, sizes, coordinates, p_size):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    predictions = [np.zeros((n_class, 1)) for size in sizes ]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))
            predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[i][j]
    return predictions

def template_patch2global(size_g, size_p, n, step):
    template = np.zeros(size_g)
    coordinates = [(0, 0)] * n ** 2
    patch = np.ones(size_p)
    step = (size_g[0] - size_p[0]) // (n - 1)
    x = y = 0
    i = 0
    while x + size_p[0] <= size_g[0]:
        while y + size_p[1] <= size_g[1]:
            template[x:x+size_p[0], y:y+size_p[1]] += patch
            coordinates[i] = (1.0 * x / size_g[0], 1.0 * y / size_g[1])
            i += 1
            y += step
        x += step
        y = 0
    return Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda(), coordinates

def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    return {'image': image, 'label': label, 'id': id}

def collate_test(batch):
    image = [ b['image'] for b in batch ] # w, h
    id = [ b['id'] for b in batch ]
    return {'image': image, 'id': id}


def model(n_class, mode=1, evaluation=False, path_g=None, path_g2l=None, path_l2g=None):
    model = fpn(n_class)
    model = nn.DataParallel(model)
    model = model.cuda()

    if (mode == 2 and not evaluation) or (mode == 1 and evaluation):
        # load fixed basic global branch
        partial = torch.load(path_g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if (mode == 3 and not evaluation) or (mode == 2 and evaluation):
        partial = torch.load(path_g2l)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    global_fixed = None
    if mode == 3:
        # load fixed basic global branch
        global_fixed = fpn(n_class)
        global_fixed = nn.DataParallel(global_fixed)
        global_fixed = global_fixed.cuda()
        partial = torch.load(path_g)
        state = global_fixed.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        global_fixed.load_state_dict(state)
        global_fixed.eval()

    if mode == 3 and evaluation:
        partial = torch.load(path_l2g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if mode == 1 or mode == 3:
        model.module.resnet_local.eval()
        model.module.fpn_local.eval()
    else:
        model.module.resnet_global.eval()
        model.module.fpn_global.eval()
    
    return model, global_fixed

def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode == 1 or mode == 3:
        # train global
        optimizer = torch.optim.Adam([
                {'params': model.module.resnet_global.parameters(), 'lr': learning_rate},
                {'params': model.module.resnet_local.parameters(), 'lr': 0},
                {'params': model.module.fpn_global.parameters(), 'lr': learning_rate},
                {'params': model.module.fpn_local.parameters(), 'lr': 0},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    else:
        # train local
        optimizer = torch.optim.Adam([
                {'params': model.module.resnet_global.parameters(), 'lr': 0},
                {'params': model.module.resnet_local.parameters(), 'lr': learning_rate},
                {'params': model.module.fpn_global.parameters(), 'lr': 0},
                {'params': model.module.fpn_local.parameters(), 'lr': learning_rate},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    return optimizer