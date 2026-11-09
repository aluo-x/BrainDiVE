import numpy as np
import os
from options import Options, load_from_nii

options = Options().parse()
beta_root = options.beta_root
output_dir = options.preprocessing_dir
subjects = options.subjects

all_subject_rois = dict()

for subject in subjects:
    print("Processing ", subject)
    roi_strings = "HCP_MMP1.nii.gz"
    raw_path = "{}/subj0{}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz".format(beta_root, subject)
    raw_data = load_from_nii(raw_path)

    full_path = os.path.join(options.roi_root.format(subject), roi_strings)
    all_roi = load_from_nii(full_path)
    all_rois_long = all_roi.astype(np.int64)
    unique_rois = np.unique(all_rois_long)
    unique_rois = unique_rois[unique_rois>0.5]

    roi_id = []
    roi_ncnsr = []

    for i in unique_rois:
        selected_mask = (all_rois_long==i)
        selected_ncnsr = raw_data[selected_mask]
        roi_id.append(i)
        roi_ncnsr.append(np.nanmean(selected_ncnsr))

    roi_id = np.array(roi_id)
    roi_ncnsr = np.array(roi_ncnsr)
    sorted_idx = np.argsort(roi_ncnsr)[::-1]
    sorted_roi_id = roi_id[sorted_idx]
    all_subject_rois[subject] = np.copy(sorted_roi_id[:])

roi_list = []
roi_ranks = []
for i in range(1,181):
    roi_list.append(i)
    roi_ranks.append([])
    for subj in subjects:
        roi_ranks[-1].append(all_subject_rois[subj].tolist().index(i))

roi_list = np.array(roi_list)
roi_ranks = np.array(roi_ranks)
print(roi_ranks)
print(roi_ranks.shape)
roi_ranks_mean = np.mean(roi_ranks, axis=1)
print(roi_ranks_mean.shape)
best = np.argsort(roi_ranks_mean)

os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "best_HCP.npy"),roi_list[best])
