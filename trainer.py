import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import os, sys
import matplotlib.pyplot as plt
import math
from datetime import timedelta
import numpy as np
from torchvision import transforms
import time
import wandb
import copy
import uuid
import pathlib
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
from torchvision.utils import make_grid
from functools import partial
import torch.distributed as dist
import random

# LOCAL
import sampling
import processes
import utils
import optimizers
import ema
import logging_utils
import time_samplers
import prediction
import unet
import saving
import augmentations

class ProcessedBatch:

    def __init__(
        self, 
        x_original, 
        x_discrete, 
        x0, 
        label, 
        x_discrete_aug=None, 
        x0_aug=None, 
        augmentation_label=None
    ):
        
        # image discrete in [0, 1] taking 256 values
        self.x_original = x_original
       
        # preprocessed (e.g. possibly centered) in [-1,1] but still discrete
        self.x_discrete = x_discrete

        # continuous e.g. drawn from N(x_discrete, gamma^2 I) for small gamma like 0.001
        self.x0 = x0

        # image label
        self.label = label

        # augmented stuff
        self.x_discrete_aug = x_discrete_aug
        self.x0_aug = x0_aug
        self.augmentation_label = augmentation_label


class BaseDiffusionTrainer:

    def __init__(
        self, 
        config, 
        rank, 
        local_seed, 
        device, 
        train_loader, 
        train_sampler, 
        test_loader
    ):

        # device and config stuff
        self.rank = rank
        self.local_seed = local_seed
        self.config = config
        self.device = device

        # data
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.test_loader = test_loader
        self.setup_overfit()
        self.process = processes.get_process(config, device)
        self.setup_networks()
        self.definitely_save(name = 'init')
        print("---------------------")
        for k in vars(self.config):
            print(k, getattr(self.config, k))
        print("---------------------")

        # how to sample times t for training
        self.train_time_sampler = time_samplers.get_train_time_sampler(
            config, device
        )

        # for now, augmentations are handled manually 
        # rather than using torchvision and torch dataloader
        # change as needed. This is turned on/off and config.py
        self.augmenter = augmentations.get_augmenter(config)

        # This is a function that converts your model output to 
        # all possible prediction types
        self.model_convert_fn = prediction.get_model_out_to_pred_obj_fn(
            model_type = self.config.model_type
        )

        # setup logging and wandb
        self.setup_wandb()

        self.plot_real_data()

        self.plot_lr_schedule()


    def plot_lr_schedule(self,):
        
        if self.is_main() and self.config.use_wandb:
            print("Plotting LR")
            plt.clf()
            fig = plt.figure()
            # plot only ever 10 steps
            steps = torch.arange(0, self.config.num_training_steps, 100)
            lrs = []
            for step in steps:
                lrs.append(round(self.optimizer.get_lr(step).item(), 8))
            plt.plot(steps, lrs, label='learning_rate')
            plt.xlabel('step')
            plt.ylabel('lr')
            plt.legend()
            wandb.log({'learning_rate': fig}, step=0)
            print("Done plotting LR")
        self.barrier()

    def is_main(self,):
        if not self.config.use_ddp:
            return True
        return self.rank == 0

    def barrier(self,):
        if not self.config.use_ddp:
            return
        dist.barrier()

    def setup_wandb(self,):
        if self.is_main() and self.config.use_wandb:
            logging_utils.setup_wandb_needs_blocking(
                self.config, 
                self.config.ckpt_dir
            )
        self.barrier()

    def plot_real_data(self,):
        if self.is_main() and self.config.use_wandb:
            logging_utils.plot_real_data_needs_blocking(
                config = self.config, 
                batch = self.overfit_batch,
                prepare_batch_fn = self.prepare_batch_fn, 
            )
        self.barrier()


    def setup_overfit(self,):
        '''
        save one batch to be a special overfitting batch for debugging
        '''
        x, label = next(iter(self.train_loader))
        x, label = x.to(self.device), label.to(self.device)
        self.overfit_batch = (x, label)

    def setup_optimizer(self,):
        '''
        Optimizer object is a wrapper around torch.optim.AdamW
        It also handles LR scheduling, grad clipping, etc...
        '''
        self.optimizer = optimizers.Optimizer(
            network = self.network,
            grad_clip_norm = self.config.grad_clip_norm,
            weight_decay = self.config.weight_decay,
            base_lr = self.config.base_lr,
            min_lr = self.config.min_lr,
            warmup_steps = self.config.warmup_steps,
            lr_schedule_type = self.config.lr_schedule_type,
            num_training_steps = self.config.num_training_steps,
            adam_b1 = self.config.adam_b1,
            adam_b2 = self.config.adam_b2,
            adam_eps = self.config.adam_eps,
        )

    def setup_networks(self,):
        '''
        Set up neural networks and handle parallelism
        '''
        config = self.config
        
        self.network = unet.get_unet_from_config(config)
        self.network_ema = deepcopy(self.network)
        self.setup_optimizer()

        self.step = 0
        
        restored = self.maybe_restore()
        
        # do not wipe ema from ckpt if restoring
        if restored:
            self.has_updated_ema = True
        else:
            self.has_updated_ema = False

        ema.requires_grad(self.network_ema, False)
        self.network_ema.eval()

        if self.config.use_ddp:
            self.network = DDP(
                self.network.to(self.device), device_ids=[self.rank]
            )
        else:
            self.network.to(self.device)

        self.network_ema.to(self.device)

        if self.config.use_ddp:
            net = self.network.module
        else:
            net = self.network

        # set ema to initialize as model
        ema.wipe_ema(net, self.network_ema)

    def maybe_update_emas(self,):
        '''
        If first update of EMA, sets EMA to equal the current model
        else, does an exponential moving average update
        '''
        updated_ema = ema.maybe_update_emas(
            network = self.network, 
            network_ema = self.network_ema, 
            use_ddp = self.config.use_ddp,
            ema_decay = self.config.ema_decay,
            update_ema_every = self.config.update_ema_every,
            update_ema_after = self.config.update_ema_after,
            step = self.step,
            first_update = not self.has_updated_ema,
        )

        if updated_ema and (not self.has_updated_ema):
            self.has_updated_ema = True

    def do_step(self, batch, epoch_num):
        '''
        Taking one training step + any optional period logging
        '''
        config = self.config
        start_batch = time.time()
        logging_dict = {}
        loss = self.loss_fn(batch, is_train = True) 
        old_norm, old_lr, new_lr = self.optimizer.take_step(
            loss, self.step
        )
        end_batch = time.time()
        logging_dict['train_loss'] = loss.item()
        logging_dict['grad_norm_before_clip'] = old_norm.item()
        self.total_time += (end_batch - start_batch)
        self.log_steps += 1
        updated_ema = self.maybe_update_emas()
        self.maybe_save()
        logging_dict = self.maybe_sample(logging_dict)
        logging_dict['epochs'] = epoch_num
        self.maybe_log(logging_dict)
        self.step += 1

    def maybe_sample(self, logging_dict, label=None):
        return sampling.maybe_sample(
            step = self.step,
            rank = self.rank,
            sample_every = self.config.sample_every,
            sample_ema_every = self.config.sample_ema_every,
            sample_after = self.config.sample_after,
            apply_fn = self.apply_fn_eval,
            batch_size_sample = self.config.batch_size_sample,
            dataset = self.config.dataset,
            num_classes = self.config.num_classes,
            EM_sample_steps = self.config.EM_sample_steps,
            model_type = self.config.model_type,
            T_min_sampling = self.config.T_min_sampling,
            T_max_sampling = self.config.T_max_sampling,
            process = self.process,
            loader = self.train_loader,
            device = self.device,
            logging_dict = logging_dict,
            label = label,
        )

    def do_epoch(self, epoch_num):
        '''
        Each epoch is many training steps
        '''
        config = self.config

        if self.config.use_ddp:
            self.train_sampler.set_epoch(epoch_num)             

        for idx, batch in enumerate(self.train_loader):

            if self.step > config.num_training_steps:
                break
            
            self.do_step(batch, epoch_num)

    def train_loop(self,):
        '''
        Train loop is many epochs
        Epoch is many steps
        each step does train and some periodic logging/eval
        '''
        self.log_steps = 0
        self.total_time = 0.0
        epoch_num = 0
        while self.step <= self.config.num_training_steps:
            self.do_epoch(epoch_num)
            epoch_num += 1
        self.definitely_save(name = 'final')
 
    def maybe_log(self, logD):
        if self.step % self.config.log_every == 0:
            if self.is_main():
                logD['steps_per_sec'] = round(self.log_steps / self.total_time, 3)
                for i, pgroup in enumerate(self.optimizer.param_groups()):
                    logD[f'lr_{i}'] = pgroup['lr']
                logging_utils.definitely_log(
                    logD, self.step, use_wandb = self.config.use_wandb
                )
                self.log_steps = 0
                self.total_time = 0.
            self.barrier()

    def maybe_save(self,):
        # locks inside to avoid needless locking
        saving.maybe_save(
            ckpt_dir = self.config.ckpt_dir,
            save_every = self.config.save_every,
            save_last_every = self.config.save_last_every,
            use_ddp = self.config.use_ddp,
            network = self.network,
            network_ema = self.network_ema,
            optimizer = self.optimizer,
            step = self.step,
            rank = self.rank,
        )

    def definitely_save(self, name=None): 
        if self.is_main():
            saving.definitely_save(
                ckpt_dir = self.config.ckpt_dir,
                use_ddp = self.config.use_ddp,
                network = self.network,
                network_ema = self.network_ema,
                optimizer = self.optimizer,
                step = self.step,
                name = name,
            )
        self.barrier()

    def maybe_restore(self,):
        restored = saving.maybe_restore(
            restore_ckpt_fname = self.config.restore_ckpt_fname,
            network = self.network,
            network_ema = self.network_ema,
            optimizer = self.optimizer,
            device = self.device
        )
        return restored


    def apply_fn(self, xt, t, label, use_ema = False, is_train = False):
        network = self.network_ema if use_ema else self.network
        network.train() if is_train else network.eval()
        return network(xt, t, label)

    def apply_fn_train(self, xt, t, label):
        return self.apply_fn(xt, t, label, use_ema = False, is_train=True)

    def apply_fn_eval(self, xt, t, label, use_ema):
        return self.apply_fn(xt, t, label, use_ema=use_ema, is_train=False)
   
    def prepare_batch_fn(self, batch):
        raise NotImplementedError

    def loss_fn(self, batch):
        raise NotImplementedError


class ExampleDiffusionTrainer(BaseDiffusionTrainer):

    '''
    If you'd like to repurpose this code, you'll likely want to write your own
    - prepare batch function
    - loss function
    by subclassing BaseDiffusionTrainer with something like the ExampleDiffusionTrainer
    as well as add a new dataset in data_utils.setup_data_train_and_test
    '''

    def dequantize(self, x_discrete):
        x0 = torch.distributions.Normal(
            loc = x_discrete, 
            scale = self.config.gamma.type_as(x_discrete)
        ).sample()
        return x0

    def prepare_batch_fn(self, batch):
       
        config = self.config
        device = self.device

        if batch is None or config.overfit:
            batch = self.overfit_batch

        x_original, label = batch

        if device is not None:
            x_original = x_original.to(device)
            label = label.to(device)

        # for images, move from [0,1] to [-1,1] , but still discrete
        x_discrete = utils.x01_to_centered(x_original)

        # a different copy of x_discrete that has been augmented
        # augmentation_label is true if any augmentation was applied and false otherwise
        x_discrete_aug, augmentation_label = self.augmenter(x_discrete)

        # dequantization to make continuous
        # this is basically optional but some say it's the proper way
        # to apply a density model to discrete data
        x0 = self.dequantize(x_discrete)

        # dequantized version of the augmented x_discete
        x0_aug = self.dequantize(x_discrete_aug)

        return ProcessedBatch(
            x_original = x_original,
            x_discrete = x_discrete,
            x_discrete_aug = x_discrete_aug,
            x0 = x0,
            x0_aug = x0_aug,
            label = label,
            augmentation_label = augmentation_label,
        )

    def loss_fn(self, batch, is_train=False):
        '''
        This is just a squared error function
        but it looks complicated because it handles generic case
        of parameterizing one object but computing squared error on another
        e.g. your model approximates E[x0|xt] but you want to compute
        velocity squared error or score squared error
        Usually your model output and desired target type match,
        but gradients can be different for differnet parameterizations.
        '''
        processed_batch = self.prepare_batch_fn(batch)

        # get augmented data during training, regular data otherwise.
        # if you implement some evals, you probably want x0 and not x0_aug
        x0 = processed_batch.x0_aug if is_train else processed_batch.x0

        batch_size = x0.shape[0]
        t = self.train_time_sampler(batch_size)

        if is_train:
            apply_fn = lambda xt, t, label: self.apply_fn_train(xt, t, label)
        else:
            apply_fn = lambda xt, t, label: self.apply_fn_eval(xt, t, label, use_ema=False)
        
        label = processed_batch.label
        coefs = self.process(t)
        x1 = self.process.sample_base(batch_size)
        xt = self.process.compute_xt(x0, x1, coefs)
        
        # This is whatever your model approximates
        model_out = apply_fn(xt, t, label)

        # This an object containing all possible prediction tpyes
        model_obj = self.model_convert_fn(t=t, xt=xt, model_out=model_out, coefs=coefs)   

        # This is your estimate of the target (a conversion of whatever your model outputted)
        model_pred = getattr(model_obj, self.config.target_type)

        # get an object with all possible targets (score target, noise target, velocity target)
        target_fn = prediction.get_target_fn()
        target_obj = target_fn(t=t, xt=xt, x0=x0, x1=x1, coefs=coefs)
        
        # get the specific target for the loss you want to compute
        target = getattr(target_obj, self.config.target_type)
        
        # the squared loss
        sq_err = utils.image_square_error(model_pred, target)

        assert sq_err.shape == (batch_size,)
        loss = sq_err.mean()
        return loss        

