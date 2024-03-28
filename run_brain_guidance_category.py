import sys
import os

sys.path.append("/ocean/projects/soc220007p/aluo/DiffusionInception")
os.environ["HF_HOME"] = "/ocean/projects/soc220007p/aluo/cache"

# print(random.getstate())
# exit()
import timm
import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from data_loader_multi_subj_split import neural_loader
from neurips_sag import mypipelineSAG
import pickle
import gc
import model_vit
import nibabel as nib
import os
import time

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()

print("Creating CLIP ViT")
# backbone = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
backbone = model_vit.feature_extractor_vit()
backbone.eval()
backbone.cuda()
assert not backbone.training
for name, param in backbone.named_parameters():
    param.requires_grad = False
print("Created CLIP ViT and moved to GPU")

NUM_TO_GENERATE = 1000
functional = []
functional_dict = {}

print("Constructing the 3D to valid cortex mask")
for s in [1,2,3,4,5,6,7,8]:
    selected = []
    for roi_strings in ["prf-visualrois.nii.gz","floc-bodies.nii.gz", "floc-faces.nii.gz", "floc-places.nii.gz", "floc-words.nii.gz", "food", "HCP"]:
        if (not (roi_strings == "food")) and (not (roi_strings == "HCP")):
            full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(s, roi_strings)
            all_roi = load_from_nii(full_path).astype(np.single)
            selected.append(all_roi>=0.5)

        elif roi_strings == "food":
            mask = np.load("/ocean/projects/soc220007p/aluo/subj_{}_food_mask.npy".format(s))
            mask2 = load_from_nii("/ocean/projects/soc220007p/aluo/rois/subj0{}/nsdgeneral.nii.gz".format(s))

            # Construct flat mask (cortex voxels)
            cortex_mask = mask2[mask2>-0.5]
            container = np.zeros(cortex_mask.shape)
            container[mask] = 1.0

            # Construct the 3D mask, then fill in the flat voxels
            original_shape = np.zeros(mask2.shape)
            original_shape[mask2>-0.5] = container
            selected.append(original_shape>=0.5)

        elif roi_strings == "HCP":
            hcp_mask = np.load("/ocean/projects/soc220007p/aluo/data/best_HCP.npy")
            nsdgeneral = load_from_nii("/ocean/projects/soc220007p/aluo/rois/subj0{}/nsdgeneral.nii.gz".format(s))
            container = np.zeros_like(nsdgeneral)
            full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}_MMP1.nii.gz".format(s, roi_strings)
            all_roi = load_from_nii(full_path).astype(np.int32)
            for i in hcp_mask[:45]:
                container[all_roi==i] += 1.0
            selected.append(container>=0.5)

    functional.append(np.logical_or.reduce(selected))
print("Completed loading of all subjects masks")
import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))

all_subjects = [1,2,3,4,5,6,7,8]
random.shuffle(all_subjects)
# We shuffle here so the slurm run tries to avoid collisions
# ["RSC", "PPA", "OPA", "FFA", "OFA"]
experiment_id = {"bodies":0, "faces":1, "places":2, "words":3, "food":4,
                 "RSC":5, "PPA":6, "OPA":7, "FFA":8, "OFA":9}

# subject = 1
# TODO remove
for subject in all_subjects:
    # subject = 2
    # TODO remove

    with open("/ocean/projects/soc220007p/aluo/DiffusionInception/random_seeds.pkl", "rb") as fff:
        all_subject_seeds = pickle.load(fff)
    subject_seeds = all_subject_seeds[subject]

    print("Starting subject {}".format(str(subject)))
    try:
        del dataset
    except:
        pass


    try:
        del brain_model
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()


    class myarg():
        def __init__(self):
            self.subject_id = [str(subject)]
            self.neural_activity_path = "/ocean/projects/soc220007p/aluo/data/cortex_subj_{}.npy"
            self.image_path = "/ocean/projects/soc220007p/aluo/data/image_data.h5py"
            self.double_mask_path = "/ocean/projects/soc220007p/aluo/double_mask_HCP.pkl"
            self.volume_functional_path = "/ocean/projects/soc220007p/aluo/volume_to_functional.pkl"
            self.early_visual_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/prf-visualrois.nii.gz"

    other_args = myarg()
    dataset = neural_loader(other_args)
    brain_model = model_vit.downproject_CLIP_split_linear(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes)
    weights = torch.load("/ocean/projects/soc220007p/aluo/DiffusionInception/results/subject_{}_neurips_split_VIT_last_fully_linear/00100.chkpt".format(subject))
    brain_model.load_state_dict(weights["network"], strict=True)
    brain_model.cuda()
    brain_model.eval()
    for name, param in brain_model.named_parameters():
        param.requires_grad = False
    assert not brain_model.training

    ############################## Some code to map from our early/higher voxels back to cortical voxels (HCP45 + functional)
    early_full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(subject, "prf-visualrois.nii.gz")
    early_visual = load_from_nii(early_full_path).astype(np.int32)
    volume_functional_mask = functional[subject - 1]  # 3D bool mask that goes from volume to ROI voxels
    early_vis_mask = torch.from_numpy((early_visual > 0.5)[volume_functional_mask])
    higher_vis_mask = torch.from_numpy((early_visual < 0.5)[volume_functional_mask])

    ##############
    # def split_to_roi_1d(input_tensor):
    #     input_tensor_flat = input_tensor.reshape(-1)
    #     rois = torch.zeros(higher_vis_mask.shape).float().to(input_tensor_flat.device)
    #     rois[early_vis_mask] = input_tensor_flat[:torch.sum(early_vis_mask)]
    #     rois[higher_vis_mask] = input_tensor_flat[torch.sum(early_vis_mask):]
    #     return rois
    ###############


    ######## Where is the memory leak??????
    try:
        del pipe
    except:
        pass

    try:
        del pipe2
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")

    # regions = ["bodies", "faces", "places", "words", "food"]
    regions = ["RSC", "PPA", "OPA", "FFA", "OFA"]
    random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
    random.shuffle(regions)
    # shuffle here to avoid slurm collision
    for region in regions:
        print("Starting S{} {}".format(subject, region))
        # region = "words"
        # region = "OFA"

        #TODO REMOVE
        random_seed_idx = experiment_id[region]
        region_seeds = list(subject_seeds[random_seed_idx][:NUM_TO_GENERATE].copy())
        random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
        random.shuffle(region_seeds)
        # roi_name = "floc-{}".format(region)

        try:
            del mask_tensor
        except:
            pass

        try:
            del region_mask_flat
            del region_mask
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # ["RSC", "PPA", "OPA", "FFA", "OFA"]
        meta_region_list = {"RSC":"places", "PPA":"places", "OPA":"places","FFA":"faces", "OFA":"faces"}
        region_sub_id_list = {"OPA":[1], "PPA":[2], "RSC":[3], "FFA":[2,3], "OFA":[1]}
        meta_region = meta_region_list[region]

        if meta_region in ["bodies", "faces", "places", "words"]:
            region_mask_string = "/ocean/projects/soc220007p/aluo/refined_roi/{}_S{}_t2.npy".format(meta_region, subject)
            # print("Loading {} ###################################".format(region_mask_string))
            region_mask = np.load(region_mask_string)
            roi_strings = "floc-{}.nii.gz".format(meta_region)
            roi_id_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(subject, roi_strings)
            # print(roi_id_path, "PATH #################")
            loaded_roi_ids = load_from_nii(roi_id_path).astype(np.int32)
            # print(loaded_roi_ids.shape, "SHAPE")
            all_sub_masks = []
            for desired_id in region_sub_id_list[region]:
                all_sub_masks.append(loaded_roi_ids==desired_id)
            all_sub_masks = np.logical_or.reduce(all_sub_masks)
            # print(np.sum(region_mask), "old region mask")
            # print(np.sum(region_mask),region_mask.shape)
            region_mask = np.logical_and(region_mask, all_sub_masks)
            # print(np.sum(region_mask), meta_region, region, subject, "new region mask")
            # exit()

            # roi_name = "floc-words"
            # for s in [SUBJ]:
            #     selected = []
            #     for roi_strings in ["{}.nii.gz".format(roi_name)]:
            #         full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(s, roi_strings)
            #         all_roi = load_from_nii(full_path).astype(np.int32)
            # #         all_roi = all_roi.astype(np.int8)
            #         good = all_roi>0.5
            #         good = all_roi==2
            #         good = all_roi>0.5
            #         bad_2 = all_roi<0.5



        elif meta_region in ["food"]:
            region_mask_string = "/ocean/projects/soc220007p/aluo/food_s{}.npy".format(subject)
            region_mask = np.load(region_mask_string)>0.5
        else:
            print("mistake!")
        # print(region, region_mask.shape, region_mask.dtype, np.unique(region_mask))
        # exit()
        region_mask_flat = region_mask[functional[subject-1]]
        mask_tensor = torch.from_numpy(region_mask_flat).bool().to("cuda")

        def loss_function_higher(image_input):
            image_features = backbone(image_input)
            predicted_voxels_higher = brain_model.forward_higher(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
            pred_response = torch.zeros(higher_vis_mask.shape).float().to("cuda")
            # pred_response[early_vis_mask] = torch.zeros(torch.sum(early_vis_mask)).float().to("cuda")
            pred_response[higher_vis_mask] = predicted_voxels_higher
            return -torch.mean(pred_response[mask_tensor])

        def loss_function_early(image_input):
            image_features = backbone(image_input)
            predicted_voxels_early = brain_model.forward_early(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
            pred_response = torch.zeros(higher_vis_mask.shape).float().to("cuda")
            pred_response[early_vis_mask] = predicted_voxels_early
            # pred_response[higher_vis_mask] = torch.zeros(torch.sum(higher_vis_mask)).to("cuda")
            return -torch.mean(pred_response[mask_tensor])

        current_folder = "/ocean/projects/soc220007p/aluo/scratchpath/images/{}/{}".format("S{}".format(subject), region + "_all_t2")
        # try:
        os.makedirs(current_folder, exist_ok=True)
        offset = 0
        pipe.brain_tweak = loss_function_higher
        for seed in region_seeds:
            offset += 1
            print("Starting {}".format(str(offset).zfill(5)))
            if offset % 20 == 0:
                print("S{}, {} region, {}/500".format(subject, region, offset))
                gc.collect()
            if offset % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            image_name = os.path.join(current_folder, "{}_{}.png".format(region, str(seed).zfill(12)))
            if os.path.exists(image_name):
                print("skipping")
                continue
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
            if os.path.exists(image_name):
                continue
            image.images[0].save(image_name, format="PNG", compress_level=6)