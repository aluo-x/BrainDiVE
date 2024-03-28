import torch
torch.backends.cudnn.benchmark = True

from encoder_dataloader import neural_loader
import os
import torch.distributed as dist
import model_vit
from time import time
from encoder_options import Options
torch.backends.cudnn.benchmark=True
import h5py
import numpy as np

def test_net(rank, other_args):
    output_device = rank
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    # other_args.subject_id = [1,2,3,4,5,6,7,8]
    dataset = neural_loader(other_args)
    feature_extractor = model_vit.feature_extractor_vit()
    feature_extractor.to(rank)
    feature_extractor.eval()
    projector = model_vit.downproject_CLIP_split_linear(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes)

    if rank == 0:
        print("Dataloader requires {} batches".format(len(dataset.testing_set)))


    loaded_weights = False
    if 1:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            return 1
        else:
            current_files = sorted(os.listdir(other_args.exp_dir))
            if len(current_files)>0:
                latest = current_files[-1]
                start_epoch = int(latest.split(".")[0]) + 1
                if rank == 0:
                    print("Identified checkpoint {} with new starting epoch {}".format(latest, start_epoch))

                map_location = 'cpu'
                weight_loc = os.path.join(other_args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print("Checkpoint loaded {}".format(weight_loc))
                projector.load_state_dict(weights["network"])
                loaded_weights = True
                if "opt" in weights:
                    load_opt = 1
        if loaded_weights is False:
            print("Resume indicated, but no weights found!")
            exit()

    _ = projector.to(rank)
    ddp_projector = projector
    ddp_projector.eval()
    print("TRAINING STATUS", ddp_projector.training)
    if rank == 0:
        old_time = time()
    # file_
    criterion = torch.nn.MSELoss()
    losses = 0.0
    offset = 0.0
    print("Saving to {}".format("/ocean/projects/soc220007p/aluo/DiffusionInception/results/eval/{}.h5py".format(cur_args.exp_name)))
    file = h5py.File("/ocean/projects/soc220007p/aluo/DiffusionInception/results/eval/{}.h5py".format(cur_args.exp_name), 'w')
    with torch.no_grad():
        print("Total {} for testing".format(len(dataset.testing_set)))
        for data_stuff_idx in range(len(dataset.testing_set)):
            if data_stuff_idx%50 == 0 and data_stuff_idx>0:
                print(data_stuff_idx, losses/offset)
            data_stuff = dataset.get_item_test(data_stuff_idx)
            neural_data = data_stuff["neural_data"][None].reshape(-1).to(output_device, non_blocking=True) # Flat tensor already
            image_data = data_stuff["image_data"][None].reshape(-1,3,224,224).to(output_device, non_blocking=True) # collapse along batch
            subj_order = data_stuff["subject_id"][None].reshape(-1).tolist() # collapse along batch
            features = feature_extractor(image_data)
            predicted = ddp_projector(features[0][0], features[0][1], features[1], subj_order).reshape(-1)
            losses += criterion(predicted, neural_data)
            offset += 1.0
            file.create_dataset("{}_gt".format(dataset.eval_key),data=neural_data.cpu().numpy().astype(np.single))
            file.create_dataset("{}_eval".format(dataset.eval_key), data=predicted.cpu().numpy().astype(np.single))
    file.close()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    cur_args.exp_name = "subject_{}_neurips_split_VIT_last_fully_linear"
    exp_name = cur_args.exp_name
    if len(cur_args.subject_id[0])>1:
        cur_args.subject_id = sorted([str(int(sbjid)) for sbjid in cur_args.subject_id[0].split(",")])

    exp_name_filled = exp_name.format("-".join(cur_args.subject_id))
    cur_args.exp_name = exp_name_filled
    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, creating...".format(cur_args.save_loc))
        os.mkdir(cur_args.save_loc)
    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir
    print("Experiment directory is {}".format(exp_dir))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    test_net(0,cur_args)