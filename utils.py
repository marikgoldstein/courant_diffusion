import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os, sys
import math
from datetime import timedelta
import numpy as np
from torchvision import transforms
import time
from torchvision import datasets as torchvision_datasets
import wandb
import copy
import uuid
from pathlib import Path                              
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
from torchvision.utils import make_grid
from functools import partial
import torch.distributed as dist
import random
from collections import OrderedDict
from torchvision.utils import save_image

def get_value_div_fn(model_fn, create_graph=False): 
    def value_div_fn(xt, t, coefs, label):
        xt = xt.clone().detach()
        t = t.clone().detach()
        eps = torch.randn_like(xt)
        with torch.set_grad_enabled(True):
            xt.requires_grad = True
            t.requires_grad = True
            model_out = model_fn(xt, t, coefs, label)
            out = (model_out * eps).sum()
            out = torch.autograd.grad(out, xt, create_graph = create_graph)[0]
        div = (out * eps).sum(dim=[1,2,3])
        assert model_out.shape == xt.shape
        assert div.shape == (xt.shape[0],)
        return model_out, div
    return value_div_fn

def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val

def Normal(a, b):
    return torch.distributions.Normal(loc = a, scale = b)

def Unif(a, b):
    return torch.distributions.Uniform(low = a, high = b)

def flip(p):
    return torch.bernoulli(p).bool()

def wide(x):
    return x[:,None,None,None]

def image_squeeze(x):
    return x.squeeze(-1).squeeze(-1).squeeze(-1)

def image_sum(x):
    return x.sum(dim=[1,2,3])

def image_square_norm(x):
    return image_sum(x.pow(2))

def image_square_error(x,y):
    return image_square_norm(x-y)

def image_dot(x, y):
    return image_sum(x * y)

def image_std_gauss_lp(x):
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    return image_sum(Normal(z, o).log_prob(x))

def x255_to_centered(x):
    return x.float() / 127.5 - 1.0

def centered_to_x255(x):
    return (x+1.0) * 127.5

def x01_to_x255(x):
    return x.float() * 255.0

def x01_to_centered(x):
    x = x01_to_x255(x)
    x = x255_to_centered(x)
    return x

def bcast_right(value, ndim):
    if len(value.shape) > ndim:
        assert False
    if len(value.shape) < ndim:
        difference = ndim - len(value.shape)
        return value.reshape(value.shape + difference * (1,))
    else:
        return value
