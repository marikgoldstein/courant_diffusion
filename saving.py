import torch
import torch.distributed as dist 
import os, sys

def maybe_save(ckpt_dir, save_every, save_last_every, use_ddp, network, network_ema, optimizer, step, rank):

    save_args = {
        'ckpt_dir': ckpt_dir,
        'use_ddp': use_ddp,
        'network': network,
        'network_ema': network_ema,
        'optimizer': optimizer,
        'step': step
    }

    if step % save_every == 0:
        if rank == 0:
            definitely_save(**save_args)
        if use_ddp:
            dist.barrier()
    
    if step % save_last_every == 0:
        if rank == 0:
            definitely_save(**save_args, name = 'last')
        if use_ddp:
            dist.barrier()

def definitely_save(ckpt_dir, use_ddp, network, network_ema, optimizer, step, name = None):

    if use_ddp:
        net_state = network.module.state_dict()
    else:
        net_state = network.state_dict()

    checkpoint = {
        'step': step,
        'network': net_state,
        'network_ema': network_ema.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
                                                                                                                                                          
    if name is None:
        name = f'step_{step}'
    new_ckpt_path = os.path.join(ckpt_dir, name + '.pt')
    torch.save(checkpoint, new_ckpt_path)
    print("saved to ", new_ckpt_path)


def maybe_restore(restore_ckpt_fname, network, network_ema, optimizer, device):                                                                                                                                     
    if restore_ckpt_fname is not None and restore_ckpt_fname != 'none':
        path = restore_ckpt_fname
        state_dict = torch.load(path, map_location = torch.device(device))
        network.load_state_dict(state_dict['network'])
        network_ema.load_state_dict(state_dict['network_ema'])
        optimizer.load_state_dict(state_dict["optimizer"])
        optimizer.set_device(device)
        step = state_dict['step']
        del state_dict
        restored = True
        print("restored model")
    else:
        print("did not restore model")
        restored = False

    return restored
