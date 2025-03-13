import os, sys
import math
import torch
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn as nn
import time
import wandb
import random

from torchdiffeq import odeint #odeint_adjoint as odeint

# local
import utils
from logging_utils import wandb_fn
import prediction


def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

def elementwise(a, x):
    return utils.bcast_right(a, x.ndim).type_as(x) * x

@torch.no_grad()
def sample(
    apply_fn,
    config, 
    process, 
    use_ema,
    loader = None, 
    device = None, 
    logging_dict = None,
    ode = False,
):
    
    print("doing sampling with ode == ", ode)

    if logging_dict is None:
        logging_dict = {}

    N = config.batch_size_sample

    label = torch.randint(0, config.num_classes, (N,)).to(device)
    x1 = process.sample_base(N).to(device)

    apply_fn_curried = lambda xt, t: apply_fn(xt, t, label, use_ema = use_ema)

    samp = EM(
        apply_fn = apply_fn_curried,
        config = config,
        process = process,
        base = x1,
        ode = ode,
    )

    key = 'base_sample'
    key += ('_ode' if ode else '')
    key += ('_ema' if use_ema else '')
    logging_dict[key] = wandb_fn(
        samp, 
        left = x1,
        gray = (config.dataset == 'mnist')
    )
    return logging_dict


def get_ode_step_fn(config, process, apply_fn, step_size):

    model_convert_fn = prediction.get_model_out_to_pred_obj_fn(model_type = config.model_type)

    dt = step_size

    def step_fn(xt, t):
        rev_t = 1 - t
        with torch.set_grad_enabled(True):
            coefs = process(rev_t)
        model_out = apply_fn(xt, rev_t)
        

        model_obj = model_convert_fn(rev_t, xt, model_out, coefs)
        vel = getattr(model_obj, 'velocity')
        xt -= dt * vel
        mu = xt
        return xt, mu
    
    return step_fn

def get_sde_step_fn(config, process, apply_fn, step_size):

    model_convert_fn = prediction.get_model_out_to_pred_obj_fn(model_type = config.model_type)
    
    dt = step_size

    def step_fn(xt, t):
        
        rev_t = 1 - t
 
        with torch.set_grad_enabled(True):
            coefs = process(rev_t)
        
        model_out = apply_fn(xt, rev_t)

        model_obj = model_convert_fn(rev_t, xt, model_out, coefs)
       
        v = model_obj.velocity
        s = model_obj.score
        delta = utils.bcast_right(coefs.delta_t , s.ndim)
        
        drift = v - delta * s
        mu =  xt + drift * dt
        z = torch.randn_like(xt)

        xt = mu + dt.sqrt() * elementwise((2.0 * delta).sqrt(), z)
        return xt, mu

    return step_fn

@torch.no_grad()
def EM(apply_fn, config, process, base, ode=False):   
    steps = config.EM_sample_steps
    print(f"Using {steps} EM steps")
    # because times are reversed in the func
    tmin = 1. - config.T_max_sampling
    tmax = 1. - config.T_min_sampling

    print("sampling tmin and tmax is", tmin, tmax)

    ts = torch.linspace(tmin, tmax, steps).type_as(base)
    step_size = ts[1] - ts[0]
    xt = base
    ones = torch.ones(base.shape[0]).type_as(xt)


    if ode:
        step_fn = get_ode_step_fn(config, process, apply_fn, step_size)
    else:
        step_fn = get_sde_step_fn(config, process, apply_fn, step_size)

    for i, tscalar in enumerate(ts):
        xt, mu = step_fn(xt, tscalar * ones)
    return mu

