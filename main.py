import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pathlib
import os
import torch.distributed as dist
import argparse

# local
from trainer import ExampleDiffusionTrainer
from data_utils import setup_data_train_and_test
from config import Config

def cleanup():
    dist.destroy_process_group()

def setup_device(config):


    # dont save any rank specific things in 
    # config so that restoring doesnt make everything
    # think it has the same rank
    if not config.use_ddp:
        config.world_size = 1
        rank = 0
        device = torch.device('cpu')
        local_seed = config.global_seed
        config.local_batch_size = config.global_batch_size_train
        return config, rank, local_seed, device

    global_bsz = config.global_batch_size_train

    if config.cpu:
        backend = 'gloo'
    else:
        backend = 'nccl'

    dist.init_process_group(backend)
    
    config.world_size = dist.get_world_size()
    
    assert global_bsz % config.world_size == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
   
    if config.cpu:
        device = 'cpu'
    else:
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)

    local_seed = config.global_seed * config.world_size + rank
    
    torch.manual_seed(local_seed)
    
     
    print(f"Starting rank={rank}, seed={local_seed}, world_size={config.world_size}.")
    
    config.local_batch_size = int(global_bsz // config.world_size)

    return config, rank, local_seed, device



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--use_wandb', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'cifar')
    parser.add_argument('--process_name', type=str, choices=['linear_vp','rf', 'cosine'])
    parser.add_argument('--time_sampler', type=str, choices=['unif','logit_normal'], default='unif')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--restore_ckpt_fname', type=str, default = None)
    parser.add_argument('--experiment_name', type = str, default = 'default_exp')

    # ask someone about these at some point
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.set_float32_matmul_precision('high')

    # ARGS are the command line args
    # CONFIG is an object with many more things in it that comes from config.py

    args = parser.parse_args()    
    config = Config(args)
    
    # DEVICE DDP STUFF
    config, rank, local_seed, device = setup_device(config)

    # CKPT DIR
    jobname = f'courant_test'

    if config.cpu:
        args.base_ckpt_dir = '~/'

    # TODO add model saving 

    base_ckpt_dir = f'./ckpts/{args.experiment_name}'


    if config.debug:
        base_ckpt_dir += '_debug'

    ckpt_dir = os.path.join(
        base_ckpt_dir, jobname
    )

    config.ckpt_dir = ckpt_dir

    print("config checkpointing directory ('config.ckpt_dir') is set to", config.ckpt_dir)

    pth = pathlib.Path(ckpt_dir)
    
    pth.mkdir(parents=True, exist_ok=True)


    # if you want to repurpose this code, you will probably want to add a
    # dataset to data_utils.setup_data_train_and_test

    train_loader, train_sampler, test_loader = setup_data_train_and_test(
        config = config,
        rank = rank,
        local_seed = local_seed,
    )

    trainer = ExampleDiffusionTrainer(
        config = config,
        rank = rank,
        local_seed = local_seed,
        device = device,
        train_loader = train_loader,
        train_sampler = train_sampler,
        test_loader = test_loader,
    )
    
    trainer.train_loop()
    print("finished training.")

    cleanup()
