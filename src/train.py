#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset.kidney_fibrosis import KidneyFibrosis
from dataset.labels import get_gt, get_ids
from tensorboardX import SummaryWriter
from option import Options
from utils.lr_scheduler import LR_Scheduler
from utils.model_builder import model, get_optimizer, collate, collate_test
from utils.trainer import Trainer
from utils.evaluator import Evaluator

args = Options().parse()
n_class = args.n_class

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################

mode = args.mode # 1: train global; 2: train local from global; 3: train global from local
evaluation = args.evaluation
test = evaluation and False
print("mode:", mode, "evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

# get train, val, test
GT, VIDS = get_gt("data/OSU_gt.txt")
ids_train, _ = get_ids("data/OSU_gt_train.txt", VIDS)
ids_val, _ = get_ids("data/OSU_gt_test.txt", VIDS)
ids_test, _ = get_ids("data/OSU_gt_test.txt", VIDS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = KidneyFibrosis(os.path.join(data_path, "downsample"), ids_train, label=True, classdict=GT, transform=False)     # default: True
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_val = KidneyFibrosis(os.path.join(data_path, "downsample"), ids_val, label=True, classdict=GT)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
dataset_test = KidneyFibrosis(os.path.join(data_path, "downsample"), ids_test, label=False, classdict=GT)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=10, collate_fn=collate_test, shuffle=False, pin_memory=True)

##### sizes are (w, h) ##############################
# make sure margin / 32 is over 1.5 AND size_g is divisible by 4
size_g = (args.size_g, args.size_g) # resized global image
size_p = (args.size_p, args.size_p) # cropped local patch size
sub_batch_size = args.sub_batch_size # batch size for train local patches
###################################
print("creating models......")

path_g = os.path.join(model_path, args.path_g)
path_g2l = os.path.join(model_path, args.path_g2l)
path_l2g = os.path.join(model_path, args.path_l2g)
model, global_fixed = model(n_class, mode, evaluation, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)

###################################
num_epochs = args.num_epochs
learning_rate = args.lr
lamb_fmreg = args.lamb_fmreg

optimizer = get_optimizer(model, mode, learning_rate=learning_rate)

scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion = nn.CrossEntropyLoss()

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(criterion, optimizer, n_class, size_g, size_p, sub_batch_size, mode, lamb_fmreg)
evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode, test)

best_pred = 0.0

for epoch in range(num_epochs):
    trainer.set_train(model)
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train); train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        if evaluation: break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample_batched, model, global_fixed)
        train_loss += loss.item()
        score_train, score_train_global, score_train_local = trainer.get_scores()
        if mode == 1: tbar.set_description('Train loss: %.3f; global acc: %.3f' % (train_loss / (i_batch + 1), score_train_global["accuracy"]))
        else: tbar.set_description('Train loss: %.3f; agg acc: %.3f' % (train_loss / (i_batch + 1), score_train_global["accuracy"]))

    score_train, score_train_global, score_train_local = trainer.get_scores()
    trainer.reset_metrics()
    # torch.cuda.empty_cache()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: tbar = tqdm(dataloader_test)
            else: tbar = tqdm(dataloader_val)

            for i_batch, sample_batched in enumerate(tbar):
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model, global_fixed)
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                # use [1:] since class0 is not considered in deep_globe metric
                if mode == 1: tbar.set_description('global acc: %.3f' % (score_val_global["accuracy"]))
                else: tbar.set_description('agg acc: %.3f' % (score_val["accuracy"]))
                images = sample_batched['image']
                if not test:
                    labels = sample_batched['label'] # PIL images

                if not evaluation and not test:
                    if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                        writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)

            if test: break
            else:
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()
                if mode == 1:
                    if score_val_global["accuracy"] > best_pred:
                        best_pred = score_val_global["accuracy"]
                        if not (test or evaluation):
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name + ".pth")

                if mode == 2:
                    if score_val["accuracy"] > best_pred:
                        best_pred = score_val["accuracy"]
                        if not (test or evaluation):
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name + ".pth")

                else:
                    if score_val["accuracy"] > best_pred: 
                        best_pred = score_val["accuracy"]
                        if not (test or evaluation):
                            print("saving model...")
                            torch.save(model.state_dict(), model_path + task_name + ".pth")

                log = ""
                log = log + 'epoch [{}/{}] acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train["accuracy"], score_val["accuracy"]) + "\n"
                log = log + 'epoch [{}/{}] Local  -- acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs,score_train_local["accuracy"], score_val_local["accuracy"]) + "\n"
                log = log + 'epoch [{}/{}] Global -- acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train_global["accuracy"], score_val_global["accuracy"]) + "\n"
                log = log + "train: " + str(score_train["accuracy"]) + "\n"
                log = log + "val:" + str(score_val["accuracy"]) + "\n"
                log = log + "Local train:" + str(score_train_local["accuracy"]) + "\n"
                log = log + "Local val:" + str(score_val_local["accuracy"]) + "\n"
                log = log + "Global train:" + str(score_train_global["accuracy"]) + "\n"
                log = log + "Global val:" + str(score_val_global["accuracy"]) + "\n"
                log += "================================\n"
                print(log)
                if evaluation: break

                f_log.write(log)
                f_log.flush()
                if mode == 1:
                    writer.add_scalars('accuracy', {'train accuracy': score_train_global["accuracy"], 'validation accuracy': score_val_global["accuracy"]}, epoch)
                else:
                    writer.add_scalars('accuracy', {'train accuracy': score_train["accuracy"], 'validation accuracy': score_val["accuracy"]}, epoch)

if not evaluation: f_log.close()