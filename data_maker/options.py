import argparse
import os
import nibabel as nib
import numpy as np
from numpy.typing import DTypeLike

nsd_root = '/mnt/lexar/NSD'
output_dir = './results'
coco_root = '/mnt/lexar/COCO'
subjects = [1]
all_subjects = list(range(1, 9))


def load_from_nii(nii_file, dtype: DTypeLike = np.float32):
    if not os.path.isfile(nii_file):
        raise FileNotFoundError(f"NIfTI file not found: {nii_file}")
    return nib.load(nii_file).get_fdata(dtype=dtype)


def data_paths(nsd_root, output_dir):
    preprocessing = os.path.join(output_dir, 'pre_processing')
    zscored = os.path.join(preprocessing, 'NSD_zscored')
    rois = os.path.join(nsd_root, 'nsddata', 'ppdata', 'subj0{}', 'func1pt8mm', 'roi')
    return {
        'preprocessing_dir': preprocessing,
        'zscored_dir': zscored,
        'beta_root': os.path.join(nsd_root, 'nsddata_betas', 'ppdata'),
        'roi_root': rois,
        'neural_activity_path': os.path.join(zscored, 'cortex_subj_{}.npy'),
        'image_path': os.path.join(preprocessing, 'image_data.h5py'),
        'double_mask_path': os.path.join(preprocessing, 'double_mask_s1.pkl'),
        'volume_functional_path': os.path.join(preprocessing, 'volume_functional_mask_s1.pkl'),
        'early_visual_path': os.path.join(rois, 'prf-visualrois.nii.gz'),
        'checkpoint_path': os.path.join(output_dir, 'subject_{}_neurips_split_VIT_last_fully_linear', '00100.chkpt'),
    }


def bool_flag(value):
    if value in ('0', '1'):
        return value == '1'
    raise argparse.ArgumentTypeError('Boolean flags must be 0 or 1.')


class Options:
    def __init__(self, subjects=subjects):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--exp_name', default='subject_{}_neurips_split_v3')
        self.parser.add_argument('--nsd-root', '--nsd_root', dest='nsd_root', default=nsd_root)
        self.parser.add_argument('--output-dir', '--save_loc', dest='save_loc', default=output_dir)
        self.parser.add_argument('--coco-root', dest='coco_root', default=coco_root)
        self.parser.add_argument('--subject_id', '--subjects', dest='subject_id', default=[str(subject) for subject in subjects], nargs='+')
        self.parser.add_argument('--gpus', default=1, type=int)
        self.parser.add_argument('--neural_activity_path', default=None)
        self.parser.add_argument('--image_path', default=None)
        self.parser.add_argument('--double_mask_path', default=None)
        self.parser.add_argument('--volume_functional_path', default=None)
        self.parser.add_argument('--early_visual_path', default=None)
        self.parser.add_argument('--checkpoint', dest='checkpoint_path', default=None)
        self.parser.add_argument('--epochs', default=100, type=int)
        self.parser.add_argument('--resume', default=0, type=bool_flag)
        self.parser.add_argument('--batch_size', default=64, type=int)
        self.parser.add_argument('--lr_init', default=3e-4, type=float)
        self.parser.add_argument('--lr_decay', default=5e-1, type=float)

    def parse(self):
        args = self.parser.parse_args()
        try:
            args.subjects = [int(subject) for value in args.subject_id for subject in value.split(',')]
        except ValueError:
            self.parser.error("Subjects must be integers from 1 to 8.")
        if any(subject not in all_subjects for subject in args.subjects):
            self.parser.error("Subjects must be integers from 1 to 8.")
        args.subject_id = [str(subject) for subject in args.subjects]
        for name, default in data_paths(args.nsd_root, args.save_loc).items():
            if getattr(args, name, None) is None:
                setattr(args, name, default)
        print('| options')
        for name, value in vars(args).items():
            print('%s: %s' % (name, value))
        print()
        return args