import os
import numpy as np
import h5py
import pickle
import gc

from scipy.io import loadmat
from options import Options, all_subjects, load_from_nii

options = Options().parse()

nsd_root = options.nsd_root
beta_root = options.beta_root
partition_root = os.path.join(options.preprocessing_dir, 'image_data_partitions.npy')
output_dir = options.zscored_dir

trials_per_sess = 750
sess_per_subj = 40
# hard coded values based on sessions that are missing for some subs
max_sess_each_subj = [40,40,32,30,40,32,40,30]

def get_session_inds_full():
    session_inds = np.repeat(np.arange(0,sess_per_subj), trials_per_sess)
    return session_inds

def load_from_hdf5(hdf5_file, keyname=None):
    data_set = h5py.File(hdf5_file, 'r')
    if keyname is None:
        keyname = list(data_set.keys())[0]
    values = np.copy(data_set[keyname])
    data_set.close()
    return values

def get_master_image_order():
    """
    Gather the "ordering" information for NSD images.
    masterordering gives zero-indexed ordering of indices (matlab-like to python-like), same for all subjects.
    consists of 30000 values in the range [0-9999], which provide a list of trials in order.
    The value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    get the image for that trial.
    """
    exp_design_file = os.path.join(nsd_root, 'nsddata','experiments','nsd','nsd_expdesign.mat')
    exp_design = loadmat(exp_design_file)

    image_order = exp_design['masterordering'].flatten() - 1

    return image_order


def load_betas(subject, sessions=[0], voxel_mask=None,  zscore_betas_within_sess=True, volume_space=True):

    """
    Load preprocessed voxel data for an NSD subject (beta weights).
    Always loading the betas with suffix 'fithrf_GLMdenoise_RR.
    Concatenate the values across multiple sessions.
    "sessions" is zero-indexed, add one to get the actual session numbers.
    """

    if volume_space:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')
    else:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', 'betas_fithrf_GLMdenoise_RR')

    print("Data is located in: %s..."%beta_subj_folder)

    if np.any((np.array(sessions)+1)>max_sess_each_subj[subject-1]):
        print("attempting to load sessions:")
        print(sessions+1)
        raise ValueError("trying to load sessions that do not exist for subject %d, only has up to session %d"\
                         %(subject, max_sess_each_subj[subject-1]))

    n_trials = len(sessions)*trials_per_sess

    for ss, se in enumerate(sessions):
        try:
            del betas
        except:
            pass

        try:
            del values
        except:
            pass

        gc.collect()
        if volume_space:

            # Load volume space nifti
            fn2load = os.path.join(beta_subj_folder, 'betas_session%02d.nii.gz'%(se+1))
            print("Loading from %s..."%fn2load)
            values = load_from_nii(fn2load).transpose((3,0,1,2))
            print("Raw data:")
            print(values.dtype, np.min(values), np.max(values), values.shape)

            betas = values.reshape((len(values), -1), order='C')

        else:
            # Surface space, concatenate the two hemispheres
            # Must be left then right to match ROI definitions.
            fn2load1 = os.path.join(beta_subj_folder, 'lh.betas_session%02d.hdf5'%(se+1))
            fn2load2 = os.path.join(beta_subj_folder, 'rh.betas_session%02d.hdf5'%(se+1))

            print("Loading from %s..."%fn2load1)
            values1 = load_from_hdf5(fn2load1)
            print("Raw data:")
            print(values1.dtype, np.min(values1), np.max(values1), values1.shape)

            print("Loading from %s..."%fn2load2)
            values2 = load_from_hdf5(fn2load2)
            print("Raw data:")
            print(values2.dtype, np.min(values2), np.max(values2), values2.shape)

            betas = np.concatenate((values1, values2), axis=1)

        # divide by 300 to convert back to percent signal change
        betas = betas.astype(np.float32) / 300

        print('Adjusted data (divided by 300):')
        print(betas.dtype, np.min(betas), np.max(betas), betas.shape)

        if voxel_mask is not None:
            betas = betas[:,voxel_mask]

        if zscore_betas_within_sess:
            print('z-scoring beta weights within this session...')
            mb = np.mean(betas, axis=0, keepdims=True)
            sb = np.std(betas, axis=0, keepdims=True)
            np.subtract(betas, mb, out=betas)
            np.add(sb, 1e-6, out=sb)
            np.divide(betas, sb, out=betas)
            np.nan_to_num(betas, copy=False)
            print ("mean = %.3f, sigma = %.3f" % (np.mean(mb), np.mean(sb)))
            del mb
            del sb
            gc.collect()

        if ss==0:
            n_vox = betas.shape[1]
            betas_full = np.zeros((n_trials, n_vox), dtype=np.single)

        betas_full[ss*trials_per_sess : (ss+1)*trials_per_sess, :] = betas

    return betas_full

def average_repetitions_streaming(subject, sessions, voxel_mask=None,
    zscore_betas_within_sess=True,
    volume_space=True,
    require_three_reps=True,
):
    """
    Normalize each session independently, then average trials by image.

    Only one session of beta data is loaded at a time.
    Returns (averaged_data_float32, sorted_unique_image_ids).
    """
    sessions = np.asarray(sessions, dtype=int)
    if sessions.size == 0:
        raise ValueError("No sessions selected.")
    if np.any(sessions < 0) or np.any(
        sessions + 1 > max_sess_each_subj[subject - 1]
    ):
        raise ValueError("Invalid session selected.")

    master_order = get_master_image_order()
    selected_orders = [
        master_order[se * trials_per_sess:(se + 1) * trials_per_sess]
        for se in sessions
    ]
    image_order = np.concatenate(selected_orders)
    unique_ims, inverse = np.unique(image_order, return_inverse=True)

    n_images = len(unique_ims)
    sums = None
    counts = np.zeros(n_images, dtype=np.int32)
    offset = 0

    for se, session_order in zip(sessions, selected_orders):
        print("\nProcessing session %d" % (se + 1))
        betas = load_betas(
            subject,
            sessions=[int(se)],
            voxel_mask=voxel_mask,
            zscore_betas_within_sess=zscore_betas_within_sess,
            volume_space=volume_space,
        )
        if sums is None:
            sums = np.zeros((n_images, betas.shape[1]), dtype=np.float32)

        session_inverse = inverse[offset:offset + len(session_order)]
        offset += len(session_order)

        # Add each trial to the corresponding image row.
        np.add.at(sums, session_inverse, betas)
        np.add.at(counts, session_inverse, 1)
        del betas
        gc.collect()

    print("Repetition-count distribution:", np.unique(counts, return_counts=True))
    if require_three_reps:
        # Keep only images observed exactly three times. This ensures that
        # every returned response is the mean of exactly three normalized trials.
        keep = counts == 3
        n_dropped = int(np.sum(~keep))
        if n_dropped:
            print("Excluding %d images without exactly 3 repetitions." % n_dropped)
        sums = sums[keep]
        counts = counts[keep]
        unique_ims = unique_ims[keep]

    if np.any(counts == 0):
        raise RuntimeError("Encountered an image with zero observations.")

    # Divide in place to avoid allocating another image-by-voxel array.
    np.divide(sums, counts[:, None], out=sums, casting="unsafe")
    print("Averaged data shape:", sums.shape, "dtype:", sums.dtype)
    return sums, unique_ims


def get_data_splits(subject, sessions=[0], voxel_mask=None, \
                    zscore_betas_within_sess=True, volume_space=True, \
                    shuffle_images=False):

    """
    Gather voxel data and the indices of training/testing images, for one NSD subject.
    Not actually loading images here, because all image features are pre-computed.
    Always leaving out the "shared1000" image subset as my validation set, and training within the rest of the data.
    Can specify a list of sessions to work with (don't have to be contiguous).
    Can specify whether to work in volume or surface space (set volume_space to True or False).
    Can also choose to shuffle images or generate random voxel data at this stage if desired.
    """

    # Load the experiment design file that defines full image order over 30,000 trials
    image_order = get_master_image_order()

    # Decide which sessions to work with here
    session_inds = get_session_inds_full()

    print(session_inds, session_inds.shape)
    print(np.unique(session_inds, return_counts=True))

    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)
    if np.any((sessions+1)>max_sess_each_subj[subject-1]):
        # adjust the session list that was entered, if the subject is missing some sessions.
        # will alter the list for both images and voxel data.
        print('subject %d only has up to session %d, will load these sessions:'%\
              (subject, max_sess_each_subj[subject-1]))
        sessions = sessions[(sessions+1)<=max_sess_each_subj[subject-1]]
        print(sessions+1)
        assert(len(sessions)>0)

    inds2use = np.isin(session_inds, sessions)
    session_inds = session_inds[inds2use]
    image_order = image_order[inds2use]

    unique_ims = np.unique(image_order)
    print(unique_ims)
    n_unique_ims = len(unique_ims)
    for uu, im in enumerate(unique_ims):
        inds = image_order==im;
        if np.sum(inds)<3:
            print(np.sum(inds), uu)

    # Load voxel data. When averaging repetitions, process one session at a time
    # so the complete trial-by-voxel matrix is never held in RAM.
    print('Loading data for sessions:')
    print(sessions + 1)

    voxel_data, unique_ims = average_repetitions_streaming(
        subject=subject,
        sessions=sessions,
        voxel_mask=voxel_mask,
        zscore_betas_within_sess=zscore_betas_within_sess,
        volume_space=volume_space,
        require_three_reps=True,
    )
    image_order = unique_ims
    session_inds = None
    
    print(voxel_data.shape, 'Voxel data shape; dtype:', voxel_data.dtype)
    assert voxel_data.shape[0] == len(image_order)

    # Get indices to split into training/validation set now
    subj_df = get_subj_df(subject)
    is_shared_image = np.array(subj_df['shared1000'])
    shared_1000_inds = is_shared_image[image_order]
    val_inds = shared_1000_inds

    is_trn, is_holdout, is_val = load_image_data_partitions(subject)
    is_trn = is_trn[image_order]
    is_val = is_val[image_order]
    is_holdout = is_holdout[image_order]
    assert(np.all(is_val==val_inds))
    holdout_inds = is_holdout

    if shuffle_images:
        print('\nShuffling image order')
        # shuffle each data partition separately
        for inds in [is_trn, is_holdout, is_val]:
            values = image_order[inds]
            image_order[inds] = values[np.random.permutation(len(values))]

    return voxel_data, image_order, val_inds, holdout_inds, session_inds

def get_subj_df(subject):
    """
    Get info about the 10,000 images that were shown to each subject.
    Note this is not the full ordered sequence of trials (which is 30,000 long)
    This is only the unique images
    (matches what is in nsd_stimuli/stimuli/nsd/S1_stimuli....h5py)
    """
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']

    nsd_meta_file = os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")

    ss=subject-1
    subject_df = stim_info.loc[subject_idx[ss,:]-1]

    return subject_df

def make_image_data_partitions(pct_holdout=0.10):

    subjects = np.array(all_subjects + [999])
    n_subjects = len(subjects)
    # fixed random seeds for each subject, to make sure shuffling is repeatable
    rndseeds = [171301, 42102, 490304, 521005, 11407, 501610, 552211, 450013, 824387]

    n_images_total = 10000
    is_trn = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_holdout = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_val = np.zeros((n_images_total,n_subjects),dtype=bool)

    for si, ss in enumerate(subjects):

        if ss==999:
            val_image_inds = np.arange(0,10000)<1000
            trn_image_inds = np.arange(0,10000)>=1000
        else:
            subject_df = get_subj_df(ss)
            val_image_inds = subject_df['shared1000']
            trn_image_inds = ~subject_df['shared1000']

        n_images_val = np.sum(val_image_inds)
        n_images_notval = np.sum(trn_image_inds);
        n_images_holdout = int(np.ceil(n_images_notval*pct_holdout))
        n_images_trn = n_images_notval - n_images_holdout

        # of the full 9000 image training set, holding out a random chunk
        inds_notval = np.where(trn_image_inds)[0]
        np.random.seed(rndseeds[si])
        np.random.shuffle(inds_notval)
        inds_trn = inds_notval[0:n_images_trn]
        inds_holdout = inds_notval[n_images_trn:]
        assert(len(inds_holdout)==n_images_holdout)

        is_trn[inds_trn,si] = 1
        is_holdout[inds_holdout,si] = 1
        is_val[val_image_inds,si] = 1

    np.save(partition_root, {'is_trn': is_trn, 'is_holdout': is_holdout, 'is_val': is_val, 'rndseeds': rndseeds})

def load_image_data_partitions(subject):

    print('loading train/holdout/val image list from %s'%partition_root)
    partitions = np.load(partition_root, allow_pickle=True).item()
    is_trn = partitions['is_trn'][:,subject-1]
    is_holdout = partitions['is_holdout'][:,subject-1]
    is_val = partitions['is_val'][:,subject-1]

    return is_trn, is_holdout, is_val

if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(partition_root):
        print("\nCreate data partition")
        make_image_data_partitions()
    else:
        print(f"Data partition already exists locally: {partition_root}")
 
    for subj in options.subjects:
        print("\nStarting subject", subj)
        subject_sessions = list(range(max_sess_each_subj[subj - 1]))
        outs = get_data_splits(subj, sessions=subject_sessions)

        # Print evaluation indices returned by get_data_splits()
        is_val = outs[2]
        eval_indices = np.where(is_val)[0]
        print(f"Number of evaluation images: {len(eval_indices)}")
        print(f"Evaluation indices:\n{eval_indices}")

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "subj_{}.npy".format(subj)), outs[0])
        np.save(os.path.join(output_dir, "subj_{}_order.npy".format(subj)), outs[1])
        del outs
        gc.collect()

