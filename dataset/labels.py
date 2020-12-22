import os
import random
import math

txt2num = ["minimal", "mild", "moderate", "severe"]

def majority_voting():
    gt_raw = open("data/OSU_raw.txt", "r")
    gt = open("data/OSU_gt.txt", "w")
    lines_raw = gt_raw.readlines()

    for line_raw in lines_raw:
        voting = {'minimal':0, 'mild':0, 'moderate':0, 'severe':0, 'none':-1}
        tmp = line_raw.replace("\n", "").split("\t")
        name = tmp[0]
        for index in range(1,6):
            voting[tmp[index]] += 1
        sorted_voting = sorted(voting.items(), key=lambda x: x[1], reverse=True)
        label = sorted_voting[0][0]
        if sorted_voting[0][1] == sorted_voting[1][1]:
            if sorted_voting[0][0] == 'mild' or sorted_voting[1][0] == 'mild':
                label = 'mild'
            else:
                label = sorted_voting[0][0]
        gt.writelines(name + '\t' + label +'\n')

def split_train_test(raw_folder, gt_train, gt_test, GT_names):
    all_list = list(range(len(raw_folder)))
    test_num = int(math.ceil(len(all_list) * 0.3))
    random.shuffle(all_list)
    test_list = all_list[0:test_num]
    train_list = all_list[test_num:]
    # if not os.path.exists(output_path + 'test/' + class_name):
    #     os.makedirs(output_path + 'test/' + class_name)
    # if not os.path.exists(output_path + 'train/' + class_name):
    #     os.makedirs(output_path + 'train/' + class_name)
    for ind in all_list:
        name = raw_folder[ind]
        label = GT_names[name]
        if ind in test_list:
            gt_test.writelines(name + ' ' + str(label) + '\n')
        else:
            gt_train.writelines(name + ' ' + str(label) + '\n')

def split_files():
    gt = open("data/OSU_gt.txt", "r")
    gt_train= open("data/OSU_gt_train.txt", "w+")
    gt_test = open("data/OSU_gt_test.txt", "w+")
    lines = gt.readlines()
    GT_all = {}
    for line in lines:
        name, label = line.replace("\n", "").split("\t")[0], line.replace("\n", "").split("\t")[1].lower()
        GT_all[name] = label

    GT_class = {'minimal':[], 'mild':[], 'moderate':[], 'severe':[]}
    for label in GT_all.keys():
        GT_class[GT_all[label]].append(label)
    print('minimal ' + str(len(GT_class['minimal'])))
    split_train_test(GT_class['minimal'],gt_train, gt_test, GT_all)
    print('mild ' + str(len(GT_class['mild'])))
    split_train_test(GT_class['mild'],gt_train, gt_test, GT_all)
    print('moderate ' + str(len(GT_class['moderate'])))
    split_train_test(GT_class['moderate'],gt_train, gt_test, GT_all)
    print('severe ' + str(len(GT_class['severe'])))
    split_train_test(GT_class['severe'],gt_train, gt_test, GT_all)
    return

def get_ids(path, valid_ids):
    gt = open(path, "r")
    lines = gt.readlines()
    ids = []
    ids_ = []
    for line in lines:
        name = line.replace("\n", "").split(" ")[0]
        #name = line.replace("\n", "").split("\t")[0]
        if name in valid_ids:
            ids.append(name)
        else:
            ids_.append(name)
    return ids, ids_

def get_gt(path, path_down='data/downsample'):
    valid_ids = []
    for file_ in os.listdir(path_down):
        valid_ids.append(file_.split('.')[0])
    gt = open(path, "r")
    lines = gt.readlines()
    GT = {}
    for line in lines:
        name, label = line.replace("\n", "").split("\t")[0], line.replace("\n", "").split("\t")[1].lower()
        #name, label = line.replace("\n", "").split(" ")[0], line.replace("\n", "").split(" ")[1].lower()
        GT[name] = txt2num.index(label)
    return GT, valid_ids

if __name__ == "__main__":
    #majority_voting()
    split_files()
