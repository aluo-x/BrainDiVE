import argparse
import torch
import os
import random
import numpy as np

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        parser = self.parser
        parser.add_argument('--exp_name', default="subject_{}_neurips_split_v3") # We override this in the trainer anyways



        parser.add_argument('--save_loc', default="./results", type=str) # Where are weights saved
        parser.add_argument('--subject_id', default=["1"], nargs='+') # Put just a single subject here
        parser.add_argument('--gpus', default=1, type=int) # Number of GPUs to use. Only tested on one
        parser.add_argument('--neural_activity_path', default="/ocean/projects/soc220007p/aluo/data/cortex_subj_{}.npy")
        parser.add_argument('--image_path', default="/ocean/projects/soc220007p/aluo/data/image_data.h5py") # All images for all subjects in one h5py
        parser.add_argument('--double_mask_path', default="/ocean/projects/soc220007p/aluo/double_mask_HCP.pkl")
        parser.add_argument('--volume_functional_path', default="/ocean/projects/soc220007p/aluo/volume_to_functional.pkl")
        parser.add_argument('--early_visual_path', default="/ocean/projects/soc220007p/aluo/rois/subj0{}/prf-visualrois.nii.gz")

        # parser.add_argument('--neural_activity_path', default="/lab_data/tarrlab/afluo/NSD_zscored/cortex_subj_{}.npy")
        # parser.add_argument('--image_path', default="/lab_data/tarrlab/afluo/NSD_zscored/image_data.h5py")
        # parser.add_argument('--double_mask_path', default="/lab_data/tarrlab/afluo/NSD_zscored/double_mask_HCP.pkl")

        parser.add_argument('--epochs', default=100, type=int) # Total epochs to train for, we use 100
        parser.add_argument('--resume', default=0, type=bool_flag) # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--lr_init', default=3e-4, type=float)  # Starting learning rate for adam/adamw
        parser.add_argument('--lr_decay', default=5e-1, type=float)  # Learning rate decay rate, so at the end of training how much you want the last lr to be.


    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        torch.manual_seed(0)
        # random.seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt