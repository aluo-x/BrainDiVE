import pandas as pd
import numpy as np
import ast
import imageio
import os
import h5py

from PIL import Image
from options import Options

options = Options().parse()

nsd_root = options.nsd_root
coco_path = options.coco_root
output_dir = options.preprocessing_dir
os.makedirs(output_dir, exist_ok=True)

csv_obj = pd.read_csv(
    os.path.join(nsd_root, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv')
)


def coco_crop(img, cropbox_in):
    if type(cropbox_in) is str:
        cropbox = ast.literal_eval(cropbox_in)
    else:
        cropbox = cropbox_in

    top = int(img.shape[0] * cropbox[0])
    bottom = int(img.shape[0] * (1 - cropbox[1]))
    left = int(img.shape[1] * cropbox[2])
    right = int(img.shape[1] * (1 - cropbox[3]))

    return img[top:bottom, left:right]

def get_cocopath(coco_id_):
    filename = str(int(coco_id_)).zfill(12) + '.jpg'
    for partition in ['val2017', 'test2017', 'train2017']:
        path = os.path.join(coco_path, partition, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"COCO image {filename} not found under {coco_path}")

all_coco = list(csv_obj['cocoId'])
file = h5py.File(options.image_path, "w")
zz = 0
for i in all_coco:
    zz += 1
    if zz %100 == 0:
        print(zz)
    current_box = list(csv_obj[csv_obj['cocoId']==int(i)]['cropBox'])[0]
    example_img = imageio.v2.imread(get_cocopath(i))
    cropped_image = np.array(coco_crop(example_img, current_box))
    cropped_image = Image.fromarray(cropped_image)
    output = np.array(cropped_image.resize((256, 256), resample=Image.Resampling.BICUBIC))
    file.create_dataset(str(i).zfill(12), data=output[:])
file.close()
