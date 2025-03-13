import torch
import numpy as np
import wandb
from torchvision.utils import make_grid
from functools import partial
from torchvision.utils import save_image

def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False

                                                                                                                                                              
def definitely_log(logD, step, use_wandb):
 
    for k in logD:
        if 'bpd' in k or 'loss' in k:
            logD[k] = round(logD[k], 4)
    
    logD['steps'] = step
            
    for k in logD:
        if torch.is_tensor(logD[k]) and logD[k].numel() == 1:
            logD[k] = round(logD[k].item(), 4)
        elif isinstance(logD[k], float):
            logD[k] = round(logD[k], 4)

    printlogD = {k : v for k,v in logD.items() if v is not None}
    if 'base_sample_real' in printlogD:
        del printlogD['base_sample_real']
    if 'samp' in printlogD:
        del printlogD['samp']

    print(printlogD)

    if use_wandb:
        wandb.log(logD, step = step)


def wandb_fn(x, left = None, right = None, gray = False, already_img = False):

    def to_grid(_x):
        bsz = _x.shape[0]
        nrow = int(np.floor(np.sqrt(bsz)))

        if already_img:
            #print("in already img")
            return make_grid(_x, nrow = nrow, normalize = False).long()
        else:   
            #print("not in already img")
            _x = _x.clip(-1, 1)
            return make_grid(_x, nrow = nrow, normalize=True, value_range=(-1, 1))

    def to_grid_gray(_x):
        g = to_grid(_x)
        g = g[0]
        return g[None,...]

    x = x.cpu()

    if gray:
        grid_fn = to_grid_gray
    else:
        grid_fn = to_grid

    x = grid_fn(x)

    if left is not None:
        left = left.cpu()
        left = grid_fn(left)
        x = torch.cat([left, x], dim=-1)

    if right is not None:
        right = right.cpu()
        right = grid_fn(right)
        x = torch.cat([x, right], dim=-1)

    if already_img:
        #print('divving')
        x = x.float()
        x = x / 255.0

    return wandb.Image(x)



@torch.no_grad()
def plot_real_data_needs_blocking(config, batch, prepare_batch_fn):

    if not config.use_wandb:
        return

    processed_batch = prepare_batch_fn(batch)
    x0 = processed_batch.x0
    x_discrete = processed_batch.x_discrete
    x1 = torch.randn_like(x0)
    x0_aug = processed_batch.x0_aug
    
    if config.dataset in ['mnist']:
        gray = True
    else:
        gray = False
    im = wandb_fn(x0, left=x_discrete, right=x0_aug, gray=gray)
    Dplot = {'xdiscrete_x0_x0aug' : im}
    wandb.log(Dplot, step = 0)

def setup_wandb_needs_blocking(config, ckpt_dir):
                                                                                                                                                                         
    c = config
    if c.use_wandb:
        c.wandb_run = wandb.init(
            project=c.wandb_project,
            entity=c.wandb_entity,
            resume=None,
            id    =None,
            name  = c.wandb_name,
            dir   = ckpt_dir
        )
        for key in vars(c):
            item = getattr(c, key)
            if is_type_for_logging(item):
                setattr(wandb.config, key, item)

