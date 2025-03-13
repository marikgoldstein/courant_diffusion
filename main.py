import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pathlib
import os
import torch.distributed as dist
import argparse

# local
from trainer import DiffusionTrainer
from data_utils import setup_data_train_and_test
from config import Config

def cleanup():
    
    dist.destroy_process_group()


def setup_device(config):

    # dont save any rank specific things in config so that restoring doesnt make everything
    # think it has the same rank

    if not config.use_ddp:
        config.world_size = 1
        rank = 0
        device = torch.device('cpu')
        local_seed = config.global_seed
        if config.evaluate:
            config.local_batch_size = config.global_batch_size_evaluate
        else:
            config.local_batch_size = config.global_batch_size_train
        return config, rank, local_seed, device

    #assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if config.evaluate:
        global_bsz = config.global_batch_size_evaluate
    else:
        global_bsz = config.global_batch_size_train

    if config.cpu:
        assert False, 'should have returned earlier'
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
    parser.add_argument('--evaluate', type = int, default = 0)
    parser.add_argument('--use_wandb', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'cifar')
    parser.add_argument('--process_name', type=str, choices=['linear_vp','rf_tied','rf_const','cosine','learned_tied'], default='rf_tied')
    parser.add_argument('--time_sampler', type=str, choices=['none','importance_sample','logit_normal'], default='none')
    parser.add_argument('--base_ckpt_dir', type=str, default='/gpfs/scratch/goldsm20/ckpts_learn')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--where', type=str, default='local') # local or purple
    parser.add_argument('--loss_type', type=str, default='hybrid')

    # ask someone about these at some point
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.set_float32_matmul_precision('high')

    args = parser.parse_args()
    
    config = Config(args)
    
    # DEVICE DDP STUFF
    config, rank, local_seed, device = setup_device(config)

    # CKPT DIR
    jobname = f'courant_test'

    if config.cpu:
        args.base_ckpt_dir = '~/'

    base_ckpt_dir = args.base_ckpt_dir

    if config.debug:
        base_ckpt_dir += '_debug'

    ckpt_dir = os.path.join(
        base_ckpt_dir, jobname
    )

    print("WILL BE WRITING CKPTS IN ",ckpt_dir)
    
    config.ckpt_dir = ckpt_dir
    
    pth = pathlib.Path(ckpt_dir)
    
    pth.mkdir(parents=True, exist_ok=True)

    train_loader, train_sampler, test_loader = setup_data_train_and_test(
        config = config,
        rank = rank,
        local_seed = local_seed,
        augment = False # handle augment manually
    )

    trainer = DiffusionTrainer(
        config = config,
        rank = rank,
        local_seed = local_seed,
        device = device,
        train_loader = train_loader,
        train_sampler = train_sampler,
        test_loader = test_loader,
    )
    
    if config.evaluate:
        trainer.evaluate()
        print("finished evaluation")
    else:
        trainer.train_loop()
        print("finished training.")

    cleanup()
