import numpy.random
import torch
import os
import pickle
import numpy as np
import random
import h5py
import gc
from skimage.transform import resize
from data_maker.options import load_from_nii

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)

def check_between(start_count, end_count, check_idx):
    return (check_idx >= start_count) and (check_idx < end_count)

OPENAI_CLIP_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None]
OPENAI_CLIP_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None]

def normalize_image(input_ndarray):
    # print(input_ndarray.dtype, np.max(input_ndarray), np.min(input_ndarray), input_ndarray.shape)
    # exit()
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))*random.uniform(0.95, 1.05)/(255.0)
    # print(scaled_image.shape, OPENAI_CLIP_STD.shape, OPENAI_CLIP_MEAN.shape, "SHAPES")
    return (scaled_image-OPENAI_CLIP_MEAN)/OPENAI_CLIP_STD

def normalize_image_deterministic(input_ndarray):
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1))/(255.0)
    return (scaled_image-OPENAI_CLIP_MEAN)/OPENAI_CLIP_STD


class neural_loader(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        self.subject_id = arg_stuff.subject_id
        if isinstance(self.subject_id, int):
            self.subject_id = list(self.subject_id)
        self.neural_activity_path = arg_stuff.neural_activity_path
        self.image_path = arg_stuff.image_path
        self.double_mask_path = arg_stuff.double_mask_path
        self.volume_functional_path = arg_stuff.volume_functional_path
        self.early_visual_path = arg_stuff.early_visual_path

        with open(self.double_mask_path, "rb") as double_mask_object:
            self.double_mask = pickle.load(double_mask_object) # 1D bool mask that goes from cortical to ROI voxels

        with open(self.volume_functional_path, "rb") as volume_functional_object:
            self.volume_functional_mask = pickle.load(volume_functional_object) # 3D bool mask that goes from volume to ROI voxels


        self.transform = normalize_image

        self.all_keys = dict() # Maps subject id to valid COCO_ids
        self.num_stimulus = dict() # Maps subject id to number of stimulus
        self.neural_sizes = dict() # Maps subject id to number of voxels
        self.early_sizes = dict() # Maps subject id to number of neurons in early visual
        self.higher_sizes = dict() # Maps subject id to number of neurons in higher visual
        self.early_visual_mask = dict() # Maps subject id to a mask, the mask goes from 1D functional to 1D early visual
        self.higher_visual_mask = dict() # Maps subject id to a mask, the mask goes from 1D function to 1D higher visual

        print("Caching the image_ids, this will take a while...")
        self.image_data = None
        all_keys = {}

        os.makedirs(arg_stuff.save_loc, exist_ok=True)
        keys_path = os.path.join(arg_stuff.save_loc, 'all_keys.pkl')
        ###### Extract testing set
        if not os.path.exists(keys_path):
            for subject in self.subject_id:
                str_subject = str(subject)
                neural_data = h5py.File(self.neural_activity_path.format(str_subject), 'r')
                all_keys[str_subject] = [i for i in list(neural_data.keys()) if (not "mask" == i)]
                neural_data.close()
            with open(keys_path, "wb") as dict_saver:
                pickle.dump(all_keys, dict_saver)
        neural_data = None
        with open(keys_path, "rb") as dict_saver:
            all_keys = pickle.load(dict_saver)


        testing_set = set.intersection(*[set(_) for _ in list(all_keys.values())]) #903 COCO ids
        # print(len(testing_set))
        # exit()
        self.testing_set = sorted(testing_set)
        self.complete_keys = all_keys
        for subject in self.subject_id:
            str_subject = str(subject)
            neural_data = h5py.File(self.neural_activity_path.format(str_subject), 'r')
            self.all_keys[str_subject] = [i for i in list(neural_data.keys()) if ((not "mask" == i) and (not i in testing_set))]
            self.num_stimulus[str_subject] = len(self.all_keys[str_subject])
            self.neural_sizes[str_subject] = np.sum(self.double_mask[int(subject)-1])

            current_early_visual = load_from_nii(arg_stuff.early_visual_path.format(str_subject), dtype=np.float64).astype(np.int32)>0.5
            # It is a float array originally, 1 or 2 = V1, 3 or 4 = V2 etc
            # 3D volume originally

            self.early_sizes[str_subject] = int(np.sum(current_early_visual[self.volume_functional_mask[int(subject)-1]]))
            self.higher_sizes[str_subject] = int(self.neural_sizes[str_subject])-self.early_sizes[str_subject]
            self.early_visual_mask = current_early_visual[self.volume_functional_mask[int(subject)-1]]
            self.higher_visual_mask = np.logical_not(current_early_visual)[self.volume_functional_mask[int(subject)-1]]

            neural_data.close()
            neural_data = None
            gc.collect()
            # Pytorch will fail if you try to use multiprocessing with an open h5py
            # Zero it out
            setattr(self, "subj_{}_neural_data".format(str_subject), None)
            # setattr(self, "subj_{}_image_data".format(str_subject), None)
        self.all_subjects = sorted(list(self.all_keys.keys()))

    def __len__(self):
        # return total number of images
        # strictly speaking this is slightly different for each subject
        # Upper bound is 10000 total (train + test) per subject
        # Just return 10K since we will use a packed format
        if len(self.all_subjects)==1:
            return list(self.num_stimulus.values())[0]
        print("multi subject case", max(list(self.num_stimulus.values())))
        return max(list(self.num_stimulus.values()))

    def __getitem__(self, idx):
        loaded = False
        # while not loaded:
        # try:
        all_images = []
        all_neural = []
        for subject_idx in self.all_subjects:
            mask = self.double_mask[int(subject_idx)-1]

            subject_neural_h5py = getattr(self, "subj_{}_neural_data".format(subject_idx))
            subject_image_h5py = self.image_data

            if subject_neural_h5py is None:
                subject_neural_h5py = h5py.File(self.neural_activity_path.format(subject_idx), 'r')
            else:
                pass

            if subject_image_h5py is None:
                subject_image_h5py = h5py.File(self.image_path.format(subject_idx), 'r')
            else:
                pass
            # print(len(subject_neural_h5py), subject_idx)
            if idx > (self.num_stimulus[subject_idx]-1):
                curidx = random.randint(0, self.num_stimulus[subject_idx]-1)
            else:
                curidx = idx
            # print(curidx, subject_idx, "random")
            # print(curidx, self.num_stimulus[subject_idx], subject_idx)
            neural_key = self.all_keys[subject_idx][curidx]
            # assert mask.shape == subject_neural_h5py[neural_key][:].shape
            selected_neural = subject_neural_h5py[neural_key][:][mask]
            selected_early_visual = selected_neural[self.early_visual_mask]
            selected_higher_visual = selected_neural[self.higher_visual_mask]
            selected_image = subject_image_h5py[str(neural_key).zfill(12)][:]
            if not (self.transform is None):
                # print(np.max(selected_image), np.min(selected_image))
                selected_image = self.transform(selected_image)
            else:
                assert False
            all_images.append(np.copy(selected_image))
            # all_neural.append(np.copy(selected_neural))
            # print(selected_neural.shape, selected_early_visual.shape, selected_higher_visual.shape)
            all_neural.append(np.copy(np.concatenate((selected_early_visual, selected_higher_visual))))
        all_neural = np.concatenate(all_neural)
        # print(self.all_subjects)
        return_subjects = np.array([int(x) for x in self.all_subjects])
        return {"subject_id":torch.from_numpy(return_subjects), "neural_data": torch.from_numpy(all_neural), "image_data": torch.from_numpy(np.array(all_images))}

    def get_item_test(self, idx):
        loaded = False
        # while not loaded:
        # try:
        all_images = []
        all_neural = []
        for subject_idx in self.all_subjects:
            mask = self.double_mask[int(subject_idx) - 1]

            subject_neural_h5py = getattr(self, "subj_{}_neural_data".format(subject_idx))
            subject_image_h5py = self.image_data

            if subject_neural_h5py is None:
                subject_neural_h5py = h5py.File(self.neural_activity_path.format(subject_idx), 'r')
            else:
                pass

            if subject_image_h5py is None:
                subject_image_h5py = h5py.File(self.image_path.format(subject_idx), 'r')
            else:
                pass
            # print(len(subject_neural_h5py), subject_idx)
            # print(curidx, subject_idx, "random")
            # print(curidx, self.num_stimulus[subject_idx], subject_idx)
            curidx = idx
            neural_key = self.testing_set[curidx]
            self.eval_key = neural_key
            # assert mask.shape == subject_neural_h5py[neural_key][:].shape
            selected_neural = subject_neural_h5py[neural_key][:][mask]
            selected_early_visual = selected_neural[self.early_visual_mask]
            selected_higher_visual = selected_neural[self.higher_visual_mask]
            selected_image = subject_image_h5py[str(neural_key).zfill(12)][:]
            selected_image = normalize_image_deterministic(selected_image)
            all_images.append(np.copy(selected_image))
            # all_neural.append(np.copy(selected_neural))
            # print(selected_neural.shape, selected_early_visual.shape, selected_higher_visual.shape)
            all_neural.append(np.copy(np.concatenate((selected_early_visual, selected_higher_visual))))
        all_neural = np.concatenate(all_neural)
        # print(self.all_subjects)
        return_subjects = np.array([int(x) for x in self.all_subjects])
        return {"subject_id": torch.from_numpy(return_subjects), "neural_data": torch.from_numpy(all_neural),
                "image_data": torch.from_numpy(np.array(all_images))}
