###########################################################################
# Copyright (c) 2020
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=4, help='classification classes')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--model_path', type=str, help='path to trained model')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1: train global branch only. 2: train local branch with fixed global branch. 3: train global branch with fixed local branch')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--size_g', type=int, default=508, help='size (in pixel) for downsampled global image')
        parser.add_argument('--size_p', type=int, default=508, help='size (in pixel) for cropped local image')
        parser.add_argument('--path_g', type=str, default="", help='name for global model path')
        parser.add_argument('--path_g2l', type=str, default="", help='name for local from global model path')
        parser.add_argument('--path_l2g', type=str, default="", help='name for global from local model path')
        parser.add_argument('--lamb_fmreg', type=float, default=0.15, help='loss weight feature map regularization')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        if args.mode == 1:
            args.num_epochs = 30
            args.lr = 5e-5
        elif args.mode == 2:
            args.num_epochs = 50
            args.lr = 2e-5
        else:
            args.num_epochs = 120
            args.lr = 5e-5
        if args.evaluation:
            args.num_epochs = 1
        return args
