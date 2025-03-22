import torch
import pathlib
import os
import pickle
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

#import torch.distributed as dist
#from torchvision import datasets, transforms
#import torchvision
#from torchvision.transforms import v2
#from torchvision.datasets import ImageFolder
#from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pickle

## from imagenet downsample ####

from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch                                                                                                                                                  
import random
import torchvision.transforms.functional as TF


class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles
        self.name = 'rotate'

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class Flip:

    def __init__(self,):
        self.name = 'flip'

    def __call__(self, x):
        return TF.hflip(x)

def get_augmenter(config):

    if config.augmentation:
        print("Using Actual Augmenter")
        return Augmenter(config)
    else:
        print("Using Null Augmenter")
        return NullAugmenter(config)


class NullAugmenter():
    
    def __init__(self, config):
        self.name = 'null augmenter'

    def __call__(self, x_discrete):
        x_aug = x_discrete
        aug_label = torch.zeros(x_aug.shape[0],).type_as(x_aug).bool()
        return x_aug, aug_label

class Augmenter:

    def __init__(self, config):
        self.config = config
        assert self.config.augmentation_prob == 0.5, "for now, to match the torchvision one"
        self.rotater = Rotation(angles=[-90, 90])
        self.flipper = Flip()

    def maybe(self, x, which):

        if which == 'flip':
            on = self.config.horizontal_flip
        elif which == 'rotate':
            on = self.config.rotate90
        else:
            assert False

        if on:
            probs = (torch.ones(x.shape[0],) * self.config.augmentation_prob).type_as(x)
        else:
            probs = torch.zeros(x.shape[0],).type_as(x)

        return torch.bernoulli(probs).bool()

    def maybe_flip(self, x):
        yesno = self.maybe(x, which='flip')
        x_aug = torch.where(yesno[:,None, None, None], self.flipper(x), x)
        return x_aug, yesno

    def maybe_rotate(self, x):
        yesno = self.maybe(x, which='rotate')
        x_aug = torch.where(yesno[:, None, None, None], self.rotater(x), x)
        return x_aug, yesno

    def __call__(self, x_discrete):

        x_aug = x_discrete

        # the b's here are just bools fo whether or not some augmentation was applied

        x_aug, b_flip = self.maybe_flip(x_aug)

        x_aug, b_rotate = self.maybe_rotate(x_aug)

        # update this if there are more bools
        # aug label is true if any augmentation was applied and false otherwise
        aug_label = torch.logical_or(b_flip, b_rotate)
        aug_label = None
        
        return x_aug, aug_label


'''

# old pytorch transforms code

def get_transform(augmentation_cfg):

    c = augmentation_cfg

    # these should always happen
    Crop = T.Lambda(lambda img: center_crop_arr(img, c.H))
    CenterCrop = T.CenterCrop(c.H)
    Tens = T.ToTensor()
    Norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    Resize = T.Resize(c.H)
    Compose = T.Compose

    # these should only happen if augment
    Flip = T.RandomHorizontalFlip() if augmentations else T.Lambda(lambda x: x) #torch.nn.Identity()
    Rand90 = Random90DegreeRotation(prob = 0.5) if augmentations else T.Lambda(lambda x: x)
    # use rand90 when if c.r90:

    if dset in ['cifar']:
        return Compose([Flip, Tens])
    
    elif dset in ['mnist']:
        return Compose([Tens])

    else:
        assert False

'''
