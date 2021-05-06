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

class Evaluator(object):
    def __init__(self, n_class, size_g, size_p, sub_batch_size=6, mode=1, test=False):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed):
        with torch.no_grad():
            images = sample['image']
            ids = sample['id']
            if not self.test:
                labels = sample['label'] # PIL images
                labels_glb = masks_transform(labels)

            width, height = images[0].size

            if width > self.size_g[0] or height > self.size_g[1]:
                images_global = resize(images, self.size_g) # list of resized PIL images
            else:
                images_global = list(images)

            if self.mode == 2 or self.mode == 3:
                images_local = [ image.copy() for image in images ]
                scores_local = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
                scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode == 2 or self.mode == 3:
                            images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                            images_local[b] = transforms.functional.hflip(images_local[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90)
                            if self.mode == 2 or self.mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90)

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global) # b, c, h, w

                    if self.mode == 2 or self.mode == 3:
                        patches, coordinates, sizes, ratios, label_patches = global2patch(images, labels_glb, self.size_p, ids)
                        predicted_patches = np.zeros((len(images), 4))
                        predicted_ensembles = np.zeros((len(images), 4))
                        outputs_global = [ None for i in range(len(images)) ]
                    if self.mode == 1:
                        # eval with only resized global image ##########################
                        if flip:
                            outputs_global += np.flip(np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global, _ = model.forward(images_glb, None, None, None)
                        ################################################################

                    if self.mode == 2:
                        # eval with patches ###########################################
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))
                                
                                # patch predictions          
                                for pred_patch, pred_ensemble in zip(torch.max(output_patches.data, 1)[1].data, torch.max(output_ensembles.data, 1)[1].data):
                                    predicted_patches[i][int(pred_patch)] += 1
                                    predicted_ensembles[i][int(pred_ensemble)] += 1

                                j += patches_var.size()[0]
                            outputs_global[i] = output_global

                        outputs_global = torch.cat(outputs_global, dim=0)
                        ###############################################################

                    if self.mode == 3:
                        # eval global with help from patches ##################################################
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                _, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))

                                for pred_patch in torch.max(output_patches.data, 1)[1].data:
                                    predicted_patches[i][int(pred_patch)] += 1
                            
                                j += self.sub_batch_size
                        # go through global image

                        tmp, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)

                        if flip:
                            outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global = tmp
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                fl = model.module.generate_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))
                                fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                                fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                                output_ensembles = model.module.ensemble(fl, fg) # include cordinates

                                # ensemble predictions
                                for pred_ensemble in torch.max(output_ensembles.data, 1)[1].data:
                                    predicted_ensembles[i][int(pred_ensemble)] += 1

                                j += self.sub_batch_size
                        ###################################################

            _, predictions_global = torch.max(outputs_global.data, 1)

            if not self.test:
                self.metrics_global.update(labels_glb, predictions_global)

            if self.mode == 2 or self.mode == 3:
                # patch predictions ###########################
                predictions_local = predicted_patches.argmax(1)
                if not self.test:
                    self.metrics_local.update(labels_glb, predictions_local)
                ###################################################
                predictions = predicted_ensembles.argmax(1)
                if not self.test:
                    self.metrics.update(labels_glb, predictions)
                return predictions, predictions_global, predictions_local
            else:
                return None, predictions_global, None
