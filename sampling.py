import os, sys
import math
import torch
#from transformers import pipeline
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
#from accelerate import Accelerator
#from accelerate.utils import ProjectConfiguration  
#from transformers import get_scheduler
from tqdm.notebook import tqdm
#from huggingface_hub import Repository, get_full_repo_name
#from transformers import get_linear_schedule_with_warmup, set_seed
#from transformers import get_cosine_schedule_with_warmup
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
    ret_D = None,
    ode = False,
):
    
    print("doing sampling with ode == ", ode)

    assert not use_ema
    assert not ode

    if ret_D is None:
        ret_D = {}

    N = config.batch_size_sample

    D = {}
    D['label'] = torch.randint(0, config.num_classes, (N,))
    D['x1'] = torch.randn(N, config.C, config.H, config.W)

    if device is not None:
        D['label'] = D['label'].to(device)
        D['x1'] = D['x1'].to(device)


    apply_fn_curried = lambda xt, t: apply_fn(xt, t, D['label'], use_ema = use_ema)


    samp = EM(
        apply_fn = apply_fn_curried,
        config = config,
        process = process,
        base = D['x1'],
        ode = ode,
    )

    if 'x0' in D:
        right = D['x0']
        k = 'base_sample_real' 
    else:
        right = None
        k = 'base_sample'
    k += ('_ode' if ode else '')
    k += ('_ema' if use_ema else '')
    ret_D[k] = wandb_fn(
        samp, 
        left = D['x1'],
        right = right,
        gray = False
    )
    return ret_D


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

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

def get_sde_step_fn(config, process, apply_fn, step_size):

    model_convert_fn = prediction.get_model_out_to_pred_obj_fn(model_type = config.model_type)
    
    dt = step_size

    def step_fn(xt, t):
        

        rev_t = 1 - t

        
        with torch.set_grad_enabled(True):
            coefs = process(rev_t)
        
        model_out = apply_fn(xt, rev_t)
        model_obj = model_convert_fn(rev_t, xt, model_out, coefs)
        score = getattr(model_obj, 'score')
        g = torch.sqrt(2.0 * coefs.delta_t)
        g2 = g.pow(2)
        g2s = elementwise(g2, score)
        f = elementwise(coefs.a_dot / coefs.a_t, xt)
        z = torch.randn_like(xt)    

        if bad(g):
            print("g is bad")
            print("rev t is",  rev_t)
            print("delta is", coefs.delta_t)
            assert False, 'g is bad'
        if bad(model_out):
            assert False, 'model out is bad'
        if bad(score):
            assert False, 'score is bad'
        if bad(f):
            assert False, 'f is bad'

        mu =  xt + (g2s-f) * dt
        xt = mu + dt.sqrt() * elementwise(g, z)
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

