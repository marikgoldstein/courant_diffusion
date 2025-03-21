import torch
import torch.distributed as dist 
import os, sys

def maybe_save(network, network_ema, optimizer, config, step, rank):
    c = config

    save_args = {
        'network': network,
        'network_ema': network_ema,
        'optimizer': optimizer,
        'config': config,
        'step': step
    }

    if step % c.save_every == 0:
        if rank == 0:
            def_save(**save_args)
        if config.use_ddp:
            dist.barrier()
    if step % c.save_last_every == 0:
        if rank == 0:
            def_save(**save_args, name = 'last')
        if config.use_ddp:
            dist.barrier()

def def_save(network, network_ema, optimizer, config, step, name = None):

    if config.use_ddp:
        net_state = network.module.state_dict()
    else:
        net_state = network.state_dict()

    checkpoint = {
        'step': step,
        'network': net_state,
        'network_ema': network_ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
                                                                                                                                                          
    if name is None:
        name = f'step_{step}'
    new_ckpt_path = os.path.join(config.ckpt_dir, name + '.pt')
    torch.save(checkpoint, new_ckpt_path)
    print("saved to ", new_ckpt_path)


def maybe_restore(config, network, network_ema, optimizer, device):                                                                                                                                     
    if config.restore_ckpt_fname is not None and config.restore_ckpt_fname != 'none':
        path = config.restore_ckpt_fname
        state_dict = torch.load(path, map_location = torch.device(device))
        network.load_state_dict(state_dict['network'])
        network_ema.load_state_dict(state_dict['network_ema'])
        optimizer.load_state_dict(state_dict["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        step = state_dict['step']
        del state_dict
        restored = True
        print("restored model")
    else:
        print("did not restore model")
        restored = False

    return restored
