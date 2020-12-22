import os
import PIL
from PIL import Image
from PIL import Image, ImageStat
PIL.Image.MAX_IMAGE_PIXELS = 10000000000

import numpy as np
import cv2

import openslide
import time

path_ori = '/scratch2/zheng/kidney_fibrosis_patch_based/osu_data/Originals_and_No_annt/'
path_ant = '/scratch2/zheng/kidney_fibrosis_patch_based/osu_data/Annotations Only/'
PATCH_SIZE = 224
stride = 224
DOWN_SIZE = 508

def preprocess_mask(path_ori, width_resize=2560, height_resize=1920):
    start_time = time.time()
    all_files = os.listdir(path_ori)
    for file_ in all_files:
        ori = openslide.OpenSlide(os.path.join(path_ori, file_))
        name = file_.split('.')[0]
        if os.path.exists(os.path.join(path_ant, '{}-annotations.png'.format(name))):
            annt = Image.open(os.path.join(path_ant, '{}-annotations.png'.format(name)))
            width, height = annt.size
            width_r = width/width_resize
            height_r = height/height_resize
            annt_resize = annt.resize((width_resize,height_resize))
            annt_resize_img = np.array(annt_resize)
            if not os.path.exists('data/annotations/'):
                os.makedirs('data/annotations/')
            cv2.imwrite('data/annotations/{}_annt.jpg'.format(name), annt_resize_img)
        print('{} finished'.format(name))
    print("--- %s seconds ---" % (time.time() - start_time_total))

def preprocess_global(path_ori, width_resize=2560, height_resize=1920):
    start_time = time.time()
    all_files = os.listdir(path_ori)
    for file_ in all_files:
        ori = openslide.OpenSlide(os.path.join(path_ori, file_))
        name = file_.split('.')[0]
        width, height = ori.dimensions
        x_resize_mins = []
        x_resize_maxs = []
        y_resize_mins = []
        y_resize_maxs = []
        if os.path.exists('data/annotations/{}_annt.jpg'.format(name)):
            annt_resized = cv2.imread('data/annotations/{}_annt.jpg'.format(name), 0)
            height_annt, width_annt = annt_resized.shape
            ret, score = cv2.threshold(annt_resized, 128, 1, 0)
            nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(score.astype(np.uint8), connectivity=4)

            for k in range(1,nLabels):
                size = stats[k, cv2.CC_STAT_AREA]
                if size < 10000: continue
                segmap = np.zeros(annt_resized.shape, dtype=np.uint8)
                segmap[labels==k] = 255
                np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
                x_resize_min, x_resize_max = min(np_contours[:,1]), max(np_contours[:,1])
                y_resize_min, y_resize_max = min(np_contours[:,0]), max(np_contours[:,0])
                x_resize_mins.append(x_resize_min)
                x_resize_maxs.append(x_resize_max)
                y_resize_mins.append(y_resize_min)
                y_resize_maxs.append(y_resize_max)
            
            width_r = width/width_resize
            height_r = height/height_resize

            if len(x_resize_maxs) != 0:
                x_resize_min, x_resize_max, y_resize_min, y_resize_max = min(x_resize_mins), max(x_resize_maxs), min(y_resize_mins), max(y_resize_maxs)
            else:
                for k in range(1,nLabels):
                    size = stats[k, cv2.CC_STAT_AREA]
                    if size < 150: continue
                    segmap = np.zeros(annt_resized.shape, dtype=np.uint8)
                    segmap[labels==k] = 255
                    np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
                    x_resize_min, x_resize_max = min(np_contours[:,1]), max(np_contours[:,1])
                    y_resize_min, y_resize_max = min(np_contours[:,0]), max(np_contours[:,0])
                    x_resize_mins.append(x_resize_min)
                    x_resize_maxs.append(x_resize_max)
                    y_resize_mins.append(y_resize_min)
                    y_resize_maxs.append(y_resize_max)
                x_resize_min, x_resize_max, y_resize_min, y_resize_max = min(x_resize_mins), max(x_resize_maxs), min(y_resize_mins), max(y_resize_maxs)

            x_min, x_max = int(x_resize_min * height_r), int(x_resize_max * height_r)
            y_min, y_max = int(y_resize_min * width_r),  int(y_resize_max * width_r)

            crop = ori.read_region((y_min,x_min), 0, (y_max-y_min,x_max-x_min))
            crop = crop.resize((DOWN_SIZE,DOWN_SIZE))
            crop_np = np.array(crop)
            crop_np = cv2.cvtColor(crop_np, cv2.COLOR_RGBA2RGB)
            if not os.path.exists('data/downsample/'):
                os.makedirs('data/downsample/')
            cv2.imwrite('data/downsample/{}.png'.format(name), crop_np)
            print('{} finished'.format(name))
    print("--- %s seconds ---" % (time.time() - start_time))

def preprocess_patch(path_ori, width_resize=2560, height_resize=1920):
    start_time = time.time()
    all_files = os.listdir(path_ori)
    if not os.path.exists('data/patches/'):
        os.mkdir('data/patches/')

    for file_ in all_files:
        ori = openslide.OpenSlide(os.path.join(path_ori, file_))
        name = file_.split('.')[0]
        width, height = ori.dimensions
        x_resize_mins = []
        x_resize_maxs = []
        y_resize_mins = []
        y_resize_maxs = []
    
        if os.path.exists('data/annotations/{}_annt.jpg'.format(name)):
            if not os.path.exists('data/patches/' + name):
                os.mkdir('data/patches/' + name)
            annt_resized = cv2.imread('data/annotations/{}_annt.jpg'.format(name), 0)
            height_annt, width_annt = annt_resized.shape
            ret, score = cv2.threshold(annt_resized, 128, 1, 0)
            nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(score.astype(np.uint8), connectivity=4)

            for k in range(1,nLabels):
                size = stats[k, cv2.CC_STAT_AREA]
                if size < 10000: continue
                segmap = np.zeros(annt_resized.shape, dtype=np.uint8)
                segmap[labels==k] = 255
                np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
                x_resize_min, x_resize_max = min(np_contours[:,1]), max(np_contours[:,1])
                y_resize_min, y_resize_max = min(np_contours[:,0]), max(np_contours[:,0])
                x_resize_mins.append(x_resize_min)
                x_resize_maxs.append(x_resize_max)
                y_resize_mins.append(y_resize_min)
                y_resize_maxs.append(y_resize_max)
    
            width_r = width/width_resize
            height_r = height/height_resize

            if len(x_resize_maxs) != 0:
                x_resize_min, x_resize_max, y_resize_min, y_resize_max = min(x_resize_mins), max(x_resize_maxs), min(y_resize_mins), max(y_resize_maxs)
            else:
                for k in range(1,nLabels):
                    size = stats[k, cv2.CC_STAT_AREA]
                    if size < 150: continue
                    segmap = np.zeros(annt_resized.shape, dtype=np.uint8)
                    segmap[labels==k] = 255
                    np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
                    x_resize_min, x_resize_max = min(np_contours[:,1]), max(np_contours[:,1])
                    y_resize_min, y_resize_max = min(np_contours[:,0]), max(np_contours[:,0])
                    x_resize_mins.append(x_resize_min)
                    x_resize_maxs.append(x_resize_max)
                    y_resize_mins.append(y_resize_min)
                    y_resize_maxs.append(y_resize_max)
                x_resize_min, x_resize_max, y_resize_min, y_resize_max = min(x_resize_mins), max(x_resize_maxs), min(y_resize_mins), max(y_resize_maxs)

            x_min, x_max = int(x_resize_min * height_r), int(x_resize_max * height_r)
            y_min, y_max = int(y_resize_min * width_r),  int(y_resize_max * width_r)
        else:
            continue

        roi_h = x_max - x_min
        roi_w = y_max - y_min
        
        wp = int((roi_w - PATCH_SIZE) / stride + 1)
        hp = int((roi_h - PATCH_SIZE) / stride + 1)
        total = wp * hp
        cnt = 0

        for w in range(wp):
            for h in range(hp):
                y = y_min + w * stride
                x = x_min + h * stride
                cnt += 1
                if y > width or x > height:
                    continue
                crop = ori.read_region((y,x), 0, (PATCH_SIZE,PATCH_SIZE))
                crop = crop.convert('RGB')

                if sum(ImageStat.Stat(crop).stddev)/3 < 18 or sum(ImageStat.Stat(crop).median)/3 > 200:
                    continue
                if not os.path.exists('data/patches/{}'.format(name)):
                    os.mkdir('data/patches/{}'.format(name))
                crop.save('data/patches/{}/{}_{}_{}_{}_{}.png'.format(name,name,str(h),str(w),str(roi_h),str(roi_w)))
                if cnt % 1000 == 0:
                    print('{}/{}'.format(str(cnt), str(total)))
        print('{} finished'.format(name))
    print("--- %s seconds ---" % (time.time() - start_time_total))
if __name__ == "__main__":
    #preprocess_mask(path_ori)
    #preprocess_global(path_ori)
    preprocess_patch(path_ori)
