import torch
torch.backends.cudnn.benchmark = True

from encoder_dataloader import neural_loader
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
import model_vit
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from encoder_options import Options
import functools
import random
from torch import autocast
torch.backends.cudnn.benchmark=True

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def worker_init_fn(worker_id, myrank_info):
    np.random.seed(worker_id + myrank_info*100)

def shuffle_shift(input_image, extent=4):
    offset_x = random.randint(-extent, extent)
    offset_y = random.randint(-extent, extent)
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
    temp = torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), mode='replicate')
    return temp

def train_net(rank, world_size, freeport, other_args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    # other_args.subject_id = [1,2,3,4,5,6,7,8]
    dataset = neural_loader(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    neural_dataloader = torch.utils.data.DataLoader(dataset, batch_size=other_args.batch_size//world_size, shuffle=False, num_workers=4, worker_init_fn=ranked_worker_init, persistent_workers=True, sampler=train_sampler,drop_last=False)
    print(dataset.early_sizes, dataset.higher_sizes, "SIZES")
    # dist.barrier()
    # dist.destroy_process_group()
    # exit()
    feature_extractor = model_vit.feature_extractor_vit([3,5])
    feature_extractor.to(rank)
    feature_extractor.eval()
    # projector = model_vit.downproject_split(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes) # used intermediate + last CLIP layer
    projector = model_vit.downproject_CLIP_split_linear(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes)

    projector.train()
    print(projector.training, "TRAINING STATUS")
    if rank == 0:
        print("Dataloader requires {} batches".format(len(neural_dataloader)))

    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if other_args.resume:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            dist.barrier()
            dist.destroy_process_group()
            return 1
        else:
            current_files = sorted(os.listdir(other_args.exp_dir))
            if len(current_files)>0:
                latest = current_files[-1]
                start_epoch = int(latest.split(".")[0]) + 1
                if rank == 0:
                    print("Identified checkpoint {} with new starting epoch {}".format(latest, start_epoch))
                if start_epoch >= (other_args.epochs+1):
                    dist.barrier()
                    dist.destroy_process_group()
                    return 1
                # map_location = 'cuda:%d' % rank
                map_location = 'cpu'
                weight_loc = os.path.join(other_args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print("Checkpoint loaded {}".format(weight_loc))
                dist.barrier()
                projector.load_state_dict(weights["network"])
                loaded_weights = True
                if "opt" in weights:
                    load_opt = 1
                dist.barrier()
        if loaded_weights is False:
            print("Resume indicated, but no weights found!")
            dist.barrier()
            dist.destroy_process_group()
            exit()
    _ = projector.to(rank)

    ddp_projector = DDP(projector, find_unused_parameters=False, device_ids=[rank], gradient_as_bucket_view=True)
    criterion = torch.nn.MSELoss()
    decay = []
    no_decay = []
    for name, m in ddp_projector.named_parameters():
        if ("higher" in name):
            decay.append(m)
            print(name, "ADDED decay")
        else:
            no_decay.append(m)

    optimizer = torch.optim.AdamW([
        {'params': decay, 'lr': other_args.lr_init, 'weight_decay': 2e-2},
        {'params': no_decay, 'lr': other_args.lr_init, 'weight_decay': 1.5e-2}], lr=other_args.lr_init, weight_decay=1.5e-2)

    print("USING AdamW with dropout double variant, DROP CONV, big decay")

    if load_opt:
        print("loading optimizer")
        optimizer.load_state_dict(weights["opt"])
        dist.barrier()

    if rank == 0:
        old_time = time()

    for epoch in range(start_epoch, other_args.epochs+1):
        decay_rate = other_args.lr_decay
        new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        total_losses = 0
        cur_iter = 0

        train_sampler.set_epoch(epoch)
        for data_stuff in neural_dataloader:
            # with torch.no_grad():
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            #
            # start.record()

            neural_data = data_stuff["neural_data"].to(output_device, non_blocking=True) # Flat tensor already
            image_data = data_stuff["image_data"][:,0].to(output_device, non_blocking=True) # collapse along batch
            subj_order = data_stuff["subject_id"].reshape(-1).tolist() # collapse along batch
            optimizer.zero_grad(set_to_none=True)
            # if rank == 0:
            #     print(data_stuff["neural_data"].shape, data_stuff["neural_data"].reshape(-1).shape)
            #     print(data_stuff["image_data"].shape, data_stuff["image_data"].reshape(-1,3,256,256).shape)
            #     print(data_stuff["subject_id"].shape)
            #     len_neural = data_stuff["neural_data"][0].shape[0]
            #     assert torch.allclose(data_stuff["neural_data"][0], data_stuff["neural_data"].reshape(-1)[:len_neural])
            #     assert torch.allclose(data_stuff["neural_data"][2], data_stuff["neural_data"].reshape(-1)[2*len_neural:3*len_neural])
            #     assert torch.allclose(data_stuff["neural_data"][3], data_stuff["neural_data"].reshape(-1)[3*len_neural:4*len_neural])
            #
            #     assert torch.allclose(data_stuff["image_data"][0], data_stuff["image_data"].reshape(-1,3,256,256)[0])
            #     assert torch.allclose(data_stuff["image_data"][2], data_stuff["image_data"].reshape(-1,3,256,256)[2])
            #     assert torch.allclose(data_stuff["image_data"][3], data_stuff["image_data"].reshape(-1,3,256,256)[3])
            #     print("true thus far")
            # with autocast(device_type='cuda', dtype=torch.float16):
            with torch.autocast("cuda"):
                with torch.no_grad():
                    features = feature_extractor(shuffle_shift(image_data)+torch.randn_like(image_data)*0.05)
            # predicted = ddp_projector(features[0][0].float(), features[0][1].float(), features[0][2].float(), features[1].float(), subj_order)
            predicted = ddp_projector(features[0][0].float(), features[0][1].float(), features[1].float(), subj_order)

            # print(predicted.shape, "PREDICTED")
            loss = criterion(predicted, neural_data)
            if rank==0:
                total_losses += loss.detach()
                cur_iter += 1
                # if cur_iter % 50 == 0:
                #     print(loss.detach().item())
            loss.backward()
            optimizer.step()
            # end.record()
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end)/1000.0, "TIMES")


        # new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))

        if rank == 0:
            avg_loss = total_losses.item() / cur_iter
            print("{}: Ending epoch {}, loss {}, time {}, lr {}".format(other_args.exp_name, epoch, avg_loss, time() - old_time, new_lrate))
            old_time = time()
        if rank == 0 and (epoch%20==0 or epoch==1 or epoch>(other_args.epochs-3)):
            save_name = str(epoch).zfill(5)+".chkpt"
            save_dict = {}
            save_dict["network"] = ddp_projector.module.state_dict()
            torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
        dist.barrier()
    print("Wrapping up training {}".format(other_args.exp_name))
    dist.barrier()
    dist.destroy_process_group()
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
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)