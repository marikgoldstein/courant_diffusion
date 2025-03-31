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
import torch.distributed as dist

from torchdiffeq import odeint #odeint_adjoint as odeint

# local
import utils
from logging_utils import wandb_fn
import prediction

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

def elementwise(a, x):
    return utils.bcast_right(a, x.ndim).type_as(x) * x


def maybe_sample(
    step,
    rank,
    sample_every,
    sample_ema_every,
    sample_after,
    apply_fn,
    batch_size_sample,
    dataset,
    num_classes,
    EM_sample_steps,
    model_type,
    T_min_sampling,
    T_max_sampling,
    process,
    device,
    loader = None,
    logging_dict = None,
    label = None,
):

    if logging_dict is None:
        logging_dict = {}
    
    for use_ema in [False, True]:
        sample_every = sample_ema_every if use_ema else sample_every
        cond1 = step % sample_every == 0
        cond2 = step >= sample_after
        if cond1 and cond2:
            if rank == 0:
                for ode in [False, True]:
                    logging_dict = sample(
                        apply_fn = apply_fn,
                        batch_size_sample = batch_size_sample,
                        dataset = dataset,
                        num_classes = num_classes,
                        EM_sample_steps = EM_sample_steps,
                        model_type = model_type,
                        T_min_sampling = T_min_sampling,
                        T_max_sampling = T_max_sampling,
                        process = process,
                        use_ema = use_ema,
                        loader = loader,
                        device = device,
                        ode = ode,
                        logging_dict = logging_dict,
                        label = label,
                    )
            dist.barrier()
        return logging_dict


@torch.no_grad()
def sample(
    apply_fn,
    batch_size_sample,
    dataset,
    num_classes,
    EM_sample_steps,
    model_type,
    ode,
    T_min_sampling,
    T_max_sampling,
    process, 
    use_ema,
    device,
    loader=None, 
    logging_dict=None,
    label=None,
):
    
    print("doing sampling with ode == ", ode)

    if logging_dict is None:
        logging_dict = {}

    N = batch_size_sample

    if label is None:
        print("using random labels for sampling")
        label = torch.randint(0, num_classes, (N,))
    else:
        print("using pre-specified labels for sampling")

    label = label.to(device)
    x1 = process.sample_base(N).to(device)
    apply_fn_curried = lambda xt, t: apply_fn(xt, t, label, use_ema = use_ema)

    samp = EM(
        apply_fn = apply_fn_curried,
        EM_sample_steps = EM_sample_steps,
        model_type = model_type,
        T_min_sampling = T_min_sampling,
        T_max_sampling = T_max_sampling,
        process = process,
        base = x1,
        ode = ode,
    )

    key = 'base_sample'
    key += ('_ode' if ode else '_sde')
    key += ('_ema' if use_ema else '_nonema')
    logging_dict[key] = wandb_fn(
        samp, 
        left = x1,
        gray = (dataset == 'mnist')
    )
    return logging_dict

@torch.no_grad()
def EM(apply_fn, EM_sample_steps, model_type, T_min_sampling, T_max_sampling, process, base, ode=False):   
    steps = EM_sample_steps
    print(f"Using {steps} EM steps")
    # because times are reversed in the func
    tmin = 1. - T_max_sampling
    tmax = 1. - T_min_sampling

    print("sampling tmin and tmax is", tmin, tmax)

    ts = torch.linspace(tmin, tmax, steps).type_as(base)
    step_size = ts[1] - ts[0]
    xt = base
    ones = torch.ones(base.shape[0]).type_as(xt)

    if ode:
        step_fn = get_ode_step_fn(model_type, process, apply_fn, step_size)
    else:
        step_fn = get_sde_step_fn(model_type, process, apply_fn, step_size)

    for i, tscalar in enumerate(ts):
        xt, mu = step_fn(xt, tscalar * ones)
    return mu


def get_ode_step_fn(model_type, process, apply_fn, step_size):

    '''
    x1 is base distribution, want to sample x0
    starting at x1, we integrate velocity backwards in an ODE
    this is the same as integrating dx = -v(x,1-t)dt in forward time
    (note both negation and 1-t)
    '''
    model_convert_fn = prediction.get_model_out_to_pred_obj_fn(model_type = model_type)

    dt = step_size

    def step_fn(xt, t):
        rev_t = 1 - t
        with torch.set_grad_enabled(True):
            coefs = process(rev_t)
        model_out = apply_fn(xt, rev_t)
        model_obj = model_convert_fn(rev_t, xt, model_out, coefs)
        vel = getattr(model_obj, 'velocity')
        xt = xt - dt * vel
        mu = xt
        return xt, mu
    
    return step_fn

def get_sde_step_fn(model_type, process, apply_fn, step_size):

    '''
    here we integrate the SDE going from x1 to x0
    In diffusion terminology, the SDE we would integrate in backward time is 
    dx = [f(x,t) - g^2(t)score(x, t) ]dt + g(t)dWt 
    
    In forward time we would do dx = (g^2(1-t)score(x,1-t) - f(x,1-t))dt + g(1-t)dWt


    In interpolant terminology, the SDE we would integrate in backward time is 

    dx = (velocity(x,t) - delta(t) * score(x,t)) dt + root(2 delta(t))dWt 
       = f_reverse(x,t) dt + root(2 delta(t)) dWt

    For some delta(t) >= 0

    These views are equivalent, where f = velocity + delta * score and 
    g = root(2 delta)
    
    It's just that for score based diffusion, f is assumed known and tractable


    Instead, for interpolant view, we may not know f even if we approximate the score, 
    but we acquire f by applying some conversions that hold between true velocities and scores 
    to our model.

    Let's stick with interpolant view. 
    
    Define f_forward(x,t) = velocity(x,t) + delta(t) * score(x,t)
    Define f_reverse(x,t) = velocity(x,t) - delta(t) * score(x,t)

    Going back to the SDE, 
    we need to integrate dx = f_reverse(x,t) dt + root(2 delta(t))dWt in reverse time
    To integrate in forward time , we thus integrate

    dx = -f_reverse(x, 1-t) dt + root(2 delta(1-t))dWt 
       = -[velocity(x, 1-t) - delta(1-t)score(x, 1-t)]dt + root(2delta(1-t)dWt

    In practice, our model outputs one of score or velocity, and we convert to get the other

    '''

    model_convert_fn = prediction.get_model_out_to_pred_obj_fn(model_type = model_type)
    
    dt = step_size

    def step_fn(xt, t):
        
        rev_t = 1 - t
 
        with torch.set_grad_enabled(True):
            coefs = process(rev_t)
        
        model_out = apply_fn(xt, rev_t)

        model_obj = model_convert_fn(rev_t, xt, model_out, coefs)
       
        velocity = model_obj.velocity
        score = model_obj.score
        delta = utils.bcast_right(coefs.delta , score.ndim)
        drift = velocity - delta * score
        mu =  xt - drift * dt
        z = torch.randn_like(xt)
        xt = mu + dt.sqrt() * elementwise((2.0 * delta).sqrt(), z)
        return xt, mu

    return step_fn


