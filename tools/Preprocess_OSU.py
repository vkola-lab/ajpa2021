"""This script creates patches using masks for all input images in a folder.
Input: data folder with the following images:
    1) unannotated original image - ex. S16-415-A1_TRICHR.ndpi, ndpi or svs files
    2) annotated mask image -   ex. S16-415-A1_TRICHR.ndpi-annotations.png
Output: folder with name of image (ex. 'S16-415-A1') containing patches for the unannotated images.
"""

import os
import numpy as np
import cv2
import PIL
from PIL import Image, ImageStat
import openslide
PIL.Image.MAX_IMAGE_PIXELS = 10000000000

import time

PATCH_SIZE = 224
STRIDE = 224

def find_aoi(img_annot):
    """find area of interest from annotated mask image"""
    img_annot = img_annot.convert('L')
    width, height = img_annot.size
    
    # downsample original image 
    width_r = width/2560
    height_r = height/1920
    annot_resize = img_annot.resize((2560,1920))
    annot_resize = np.array(annot_resize)
    annot_resize = (annot_resize > 128) * 255

    # find area of interest
    xy = np.where(annot_resize == 255)
    x_resize_min = xy[0].min()
    x_resize_max = xy[0].max()
    y_resize_min = xy[1].min()
    y_resize_max = xy[1].max()

    # get original bounding box
    x_min, x_max = int(x_resize_min * height_r), int(x_resize_max * height_r)
    y_min, y_max = int(y_resize_min * width_r),  int(y_resize_max * width_r)

    return [x_min, y_min, x_max, y_max]

def patch_gen(bbox, img_ori, patch_path):
    """"creates patches given the following parameters:
        bbox = bounding box of area of interest
        img_ori = unannotated original image
        patch_path = folder containing image pathes"""
    width, height = img_ori.dimensions
    aoi_h = bbox[2] - bbox[0]
    aoi_w = bbox[3] - bbox[1]
    wp = int((aoi_w - PATCH_SIZE) / STRIDE + 1)
    hp = int((aoi_h - PATCH_SIZE) / STRIDE + 1)
    total = wp * hp
    cnt = 0
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)
    for w in range(wp):
        for h in range(hp):
            y = bbox[1] + w * STRIDE 
            x = bbox[0] + h * STRIDE
            cnt += 1
            if y > width or x > height:
                continue
            crop = img_ori.read_region((y,x), 0, (PATCH_SIZE,PATCH_SIZE))
            crop = crop.convert('RGB')
            # filters out 1) background patches that have no kidney in them 2) kidney cortex less than half
            if sum(ImageStat.Stat(crop).stddev)/3 < 18 or sum(ImageStat.Stat(crop).median)/3 > 200:
                continue
            crop.save('{}/{}_{}.png'.format(patch_path,str(y),str(x)))
            if cnt % 1000 == 0:
                print('{}/{}'.format(str(cnt), str(total)))


start_time_total = time.time()

dir_name_ori = '/scratch2/zheng/kidney_fibrosis_patch_based/osu_data/Originals/'
dir_name_annot = '/scratch2/zheng/kidney_fibrosis_patch_based/osu_data/Annotations Only/'
filenames = os.listdir(dir_name_ori)

for filename in filenames:
    path_ori = os.path.join(dir_name_ori, filename)
    path_annot = os.path.join(dir_name_annot, '{}-annotations.png').format(filename.rsplit(".", 1)[0])

    try:
        img_ori = openslide.OpenSlide(path_ori)
        img_annot = Image.open(path_annot)
    except:
        print("Can't open {}".format(filename))
        continue

    bbox = find_aoi(img_annot)
    patch_path = os.path.join("patches", filename.rsplit(".", 1)[0])
    patch_gen(bbox, img_ori, patch_path)

print("--- %s seconds ---" % (time.time() - start_time_total))
