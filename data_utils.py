import pathlib
import os
import pickle
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import os.path
import numpy as np
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


def setup_data_train_and_test(config, rank, local_seed):

    train_loader, train_sampler = setup_data(
        is_train = True,
        use_ddp = config.use_ddp,
        config = config,
        rank = rank,
        global_seed = config.global_seed,
        local_seed = local_seed,
    )
    # train sampler might be none if not using ddp

    test_loader, test_sampler = setup_data(
        is_train = False,
        use_ddp = False, 
        config = config, 
        rank = rank, 
        global_seed = config.global_seed,
        local_seed = local_seed,
    )
    del test_sampler

    return train_loader, train_sampler, test_loader


def setup_data(
    is_train,
    use_ddp,
    config,
    rank,
    global_seed,
    local_seed,
):
    
    path = './data/'
    transform = torchvision.transforms.ToTensor()

    if config.dataset== 'mnist':
        ds = datasets.MNIST(path, train = is_train, download = True, transform = transform)
    
    elif config.dataset == 'cifar':
        ds = datasets.CIFAR10(path, train = is_train, download = True, transform = transform)

    if use_ddp:
 
        sampler = DistributedSampler(
                ds,
                num_replicas = config.world_size,
                rank = rank,
                seed = global_seed,
                shuffle = True,
            )

        dl = DataLoader(
            ds,
            batch_size=config.local_batch_size,
            sampler= sampler,
            num_workers=config.workers,
            shuffle = False, # exclusive with sampler
            pin_memory=True,
            drop_last=True                                                                                                                                                                                          
        )
        return dl, sampler

    else:
 
        dl = DataLoader(
            ds,
            batch_size = config.local_batch_size,
            shuffle = False,
            sampler = None,
            num_workers = config.workers,
            pin_memory = True,
            drop_last = False
        )
        return dl, None

