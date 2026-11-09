import pickle
import os
import h5py
import numpy as np
from options import Options, load_from_nii

options = Options().parse()
output_dir = options.preprocessing_dir

EXPECTED_EARLY = 4657
EXPECTED_HIGHER = 22961
EXPECTED_TOTAL = EXPECTED_EARLY + EXPECTED_HIGHER


def make_food_volume(food_data, nsdgeneral):

    # Identify active cortical voxels
    active_voxels = nsdgeneral > -0.5

    # Flatten to 1D array containing only cortical voxel values
    cortex_mask = nsdgeneral[active_voxels]
    vector_size = cortex_mask.shape[0]

    # Food_data contains indices into the flattened cortical voxels
    indices = np.asarray(food_data).reshape(-1).astype(np.int64)
    food_vector = np.zeros(vector_size, dtype=bool)
    food_vector[indices] = True

    # Reconstruct 3D food mask
    food_volume = np.zeros(nsdgeneral.shape, dtype=bool)
    food_volume[active_voxels] = food_vector
    return food_volume


best_hcp_path = os.path.join(output_dir, 'best_HCP.npy')

if not os.path.isfile(best_hcp_path):
    raise FileNotFoundError(f"HCP ranking not found: {best_hcp_path}")

best_hcp = np.load(best_hcp_path).astype(np.int32)
top_hcp_rois = best_hcp[:45]

if len(top_hcp_rois) != 45:
    raise ValueError("Expected 45 HCP labels.")

volume_functional_masks = [None] * max(options.subjects)
double_masks = [None] * max(options.subjects)
for subject in options.subjects:
    print(f"\nProcessing subject {subject}")

    food_path = os.path.join(output_dir, f'food_s{subject}.npy')
    if not os.path.isfile(food_path):
        raise FileNotFoundError(f"Food mask not found: {food_path}")
    roi_dir = options.roi_root.format(subject)
    neural_path = options.neural_activity_path.format(subject)
    roi_masks = [
        ('prf-visualrois', load_from_nii(os.path.join(roi_dir, 'prf-visualrois.nii.gz'))>= 0.5,),
        ('floc-bodies', load_from_nii(os.path.join(roi_dir, 'floc-bodies.nii.gz'))>= 0.5,),
        ('floc-faces', load_from_nii(os.path.join(roi_dir, 'floc-faces.nii.gz'))>= 0.5,),
        ('floc-places', load_from_nii(os.path.join(roi_dir, 'floc-places.nii.gz'))>= 0.5,),
        ('floc-words', load_from_nii(os.path.join(roi_dir, 'floc-words.nii.gz'))>= 0.5,),
    ]

    hcp = load_from_nii(os.path.join(roi_dir, 'HCP_MMP1.nii.gz')).astype(np.int32)
    hcp_mask = np.isin(hcp, top_hcp_rois)
    roi_masks.append(('HCP top 45', hcp_mask))

    nsdgeneral = load_from_nii(os.path.join(roi_dir, 'nsdgeneral.nii.gz'))

    food_data = np.load(food_path, allow_pickle=False)
    food_volume = make_food_volume(food_data, nsdgeneral)
    roi_masks.append(('food', food_volume))

    with h5py.File(neural_path, 'r') as f:
        if 'mask' not in f:
            raise KeyError(f"No mask dataset in {neural_path}")
        neural_vector_mask = f['mask'][:].astype(bool)

    if neural_vector_mask.shape != hcp.shape:
        raise ValueError(f"Neural mask {neural_vector_mask.shape} does not match HCP {hcp.shape}")
 
    print(f"Neural-vector mask voxels: {np.count_nonzero(neural_vector_mask)}")
    print(f"Food volume voxels: {np.count_nonzero(food_volume)}")
    print("\nIndividual and cumulative contributions")

    combined_mask = np.zeros(neural_vector_mask.shape, dtype=bool)
 
    for name, mask in roi_masks:
        mask_in_neural = mask & neural_vector_mask
        combined_mask |= mask_in_neural

    actual_total = int(np.count_nonzero(combined_mask))
    print(f"Expected total:  {EXPECTED_TOTAL}")
    print(f"Actual total:    {actual_total}")

    volume_functional_mask = combined_mask.astype(bool)
    double_mask = combined_mask[neural_vector_mask.astype(bool)].astype(bool)
    volume_functional_masks[subject - 1] = volume_functional_mask
    double_masks[subject - 1] = double_mask

os.makedirs(output_dir, exist_ok=True)

with open(options.volume_functional_path, "wb") as f:
    pickle.dump(volume_functional_masks, f,
                protocol=pickle.HIGHEST_PROTOCOL)

with open(options.double_mask_path, "wb") as f:
    pickle.dump(double_masks, f,
                protocol=pickle.HIGHEST_PROTOCOL)

print(f"Volume mask: {volume_functional_mask.shape} {int(volume_functional_mask.sum())}")
print(f"Double mask: {double_mask.shape} {int(double_mask.sum())}")
