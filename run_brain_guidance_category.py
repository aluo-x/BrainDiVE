import os
import torch
import numpy as np
import pickle
import gc
import encoder_model_vit as model_vit
import random

from diffusers import DPMSolverMultistepScheduler

from data_maker.options import Options, load_from_nii
from brain_guide_pipeline import mypipelineSAG

options = Options().parse()
NUM_TO_GENERATE = 5
all_subjects = list(options.subjects)

print("Creating CLIP ViT")
# backbone = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
backbone = model_vit.feature_extractor_vit()
backbone.eval()
backbone.cuda()
assert not backbone.training
for name, param in backbone.named_parameters():
    param.requires_grad = False
print("Created CLIP ViT and moved to GPU")

random.shuffle(all_subjects)
# We shuffle here so the slurm run tries to avoid collisions
# ["RSC", "PPA", "OPA", "FFA", "OFA"]
experiment_id = {"bodies":0, "faces":1, "places":2, "words":3, "food":4,
                 "RSC":5, "PPA":6, "OPA":7, "FFA":8, "OFA":9}
pre_processing_path = options.preprocessing_dir

for subject in all_subjects:

    print("Starting subject {}".format(str(subject)))
    try:
        del brain_model
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()


    with open(options.double_mask_path, "rb") as f:
        double_mask = pickle.load(f)

    with open(options.volume_functional_path, "rb") as f:
        functional_mask = pickle.load(f)

    str_subject = str(subject)
    early_visual = load_from_nii(options.early_visual_path.format(str_subject), dtype=np.float64).astype(np.int32) > 0.5
    volume_functional_mask = functional_mask[subject - 1]
    
    neural_size = int(np.sum(double_mask[subject - 1]))
    early_size = int(np.sum(early_visual[volume_functional_mask]))
    higher_size = neural_size - early_size
    early_sizes = {str_subject: early_size}
    higher_sizes = {str_subject: higher_size}

    early_vis_mask = torch.from_numpy(early_visual[volume_functional_mask]).bool()
    higher_vis_mask = torch.from_numpy(np.logical_not(early_visual)[volume_functional_mask]).bool()

    brain_model = model_vit.downproject_CLIP_split_linear(num_early_output=early_sizes, num_higher_output=higher_sizes)
    weights = torch.load(options.checkpoint_path.format(subject))

    brain_model.load_state_dict(weights["network"], strict=True)
    brain_model.cuda()
    brain_model.eval()
    for name, param in brain_model.named_parameters():
        param.requires_grad = False
    assert not brain_model.training

    ####### Where is the memory leak??????
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

    repo_id = "Manojb/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")

    # regions = ["RSC", "PPA", "OPA", "FFA", "OFA"]
    regions = ["PPA", "FFA"]
    random.seed(os.urandom(5))
    random.shuffle(regions)
    tstat_file_list = {"faces": "floc_facestval.nii.gz",
                       "places": "floc_placestval.nii.gz",
                       "bodies": "floc_bodiestval.nii.gz",
                       "words": "floc_charactertsval.nii.gz",
                       "food": "floc_placestval.nii.gz"}
    # shuffle here to avoid slurm collision
    for region in regions:
        print("Starting S{} {}".format(subject, region))

        random_seed_idx = experiment_id[region]
        seed_rng = random.Random(5 + subject * 100 + random_seed_idx)
        region_seeds = seed_rng.sample(range(0, 2**32), NUM_TO_GENERATE)
        random.shuffle(region_seeds)
        
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
        
        meta_region_list = {"RSC":"places", "PPA":"places", "OPA":"places","FFA":"faces", "OFA":"faces"}
        region_sub_id_list = {"OPA":[1], "PPA":[2], "RSC":[3], "FFA":[2,3], "OFA":[1]}
        meta_region = meta_region_list[region]

        if meta_region in ["bodies", "faces", "places", "words"]:
            roi_strings = "floc-{}.nii.gz".format(meta_region)
            roi_id_path = os.path.join(options.roi_root.format(subject), roi_strings)
            loaded_roi_ids = load_from_nii(roi_id_path, dtype=np.float64).astype(np.int32)
            
            all_sub_masks = []
            for desired_id in region_sub_id_list[region]:
                all_sub_masks.append(loaded_roi_ids==desired_id)
            all_sub_masks = np.logical_or.reduce(all_sub_masks)
                     
            tstat_filename = tstat_file_list[meta_region]
            t_stats_path = os.path.join(os.path.dirname(options.roi_root.format(subject)), tstat_filename)
            if not os.path.exists(t_stats_path):
               raise FileNotFoundError("T-statistic map not found.")

            t_stats = load_from_nii(t_stats_path, dtype=np.float64).astype(np.int32)
            t_mask = t_stats > 2
            if all_sub_masks.shape != t_mask.shape:
               raise ValueError(
                  "ROI and t-statistic maps have different shapes:{} vs {}".format(all_sub_masks.shape, t_mask.shape))
            region_mask = np.logical_and(t_mask, all_sub_masks)

        elif meta_region in ["food"]:
            region_mask_string = os.path.join(pre_processing_path, 'food_s{}.npy'.format(subject))
            region_mask = np.load(region_mask_string)>0.5
        else:
            print("mistake!")

        region_mask_flat = region_mask[volume_functional_mask]
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

        output_dir = os.path.join(options.save_loc, 'diffusion')
        current_folder = os.path.join(output_dir, "S{}".format(subject), region + "_all_t2")
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
