# local
import torch
import data_utils
import os
import processes
import numpy as np


class Config:
    def __init__(self, args):

        self.cpu = False
        self.use_ddp = True
        self.model_type = 'velocity'
        self.target_type = 'velocity'
        self.debug = bool(args.debug)
        

        LIGHTNING = self.debug
        self.dataset = args.dataset
        self.global_batch_size_train = 128 if LIGHTNING else 256 
        self.overfit = self.debug
        self.EM_sample_steps = 500 if LIGHTNING else 500 
        self.freq = 100 if LIGHTNING else 25_000
        self.save_every = 10_000 if LIGHTNING else self.freq
        self.save_last_every = 1000
        self.sample_every = 1000 if LIGHTNING else self.freq
        self.sample_ema_every = self.freq
        self.sample_with_ode = False if LIGHTNING else True
        self.warmup_steps = 100 if LIGHTNING else 20_000

        self.sample_after = 1
        self.batch_size_sample = 128
        self.log_every = 100
        self.use_wandb = bool(args.use_wandb)

        self.time_sampler = args.time_sampler
        self.process_name = args.process_name
       
        # some process hyperparams
        self.beta_min = 0.01 # for VP
        self.beta_max = 20.0 # for VP
        self.delta_const = 0.001 # for RF

        eps = 1e-3
        print("minimum time for", self.process_name, "is", eps)

        self.epsilon = eps

        self.T_min_training = eps
        self.T_max_training = 1 - eps

        self.T_min_sampling = eps 
        self.T_max_sampling = 1 - eps

        self.vocab_size = 256
        self.seed = 8
        self.global_seed = self.seed
        self.workers = 4 

        # std_dev of dequantized q(x0_continuous|x_discrete)
        self.gamma = torch.tensor(0.001)

        self.sample_with_ema = not self.overfit
        self.ema_decay = .9998
        self.update_ema_every = 1
        self.update_ema_after = 20000

        self.grad_clip_norm = 1.0
        self.weight_decay = 0.0
        self.base_learning_rate = 2.0e-4 
        self.min_learning_rate = 1e-6
        self.lr_schedule = 'cosine'
        self.adam_b1 = 0.9
        self.adam_b2 = 0.99
        self.adam_eps = 1e-8
        
        self.num_training_steps = 400_000

        self.max_gpu_batch_size = 128

        if self.dataset == 'mnist':
            self.C, self.H, self.W = 1, 28, 28
            self.num_classes = 10
        elif self.dataset == 'cifar':
            self.C, self.H, self.W = 3, 32, 32
            self.num_classes = 10
        else:
            assert False

        self.num_classes_for_model = self.num_classes
        self.data_dim = self.C * self.H * self.W
     
       	self.unet_use_classes = True
        self.unet_channels = 128
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 64
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

        self.wandb_entity = 'marikgoldstein'
        self.wandb_project = 'courant_diffusion'
        self.resume_wandb_id = None
        self.wandb_name = str(self.dataset) + '_' + str(self.process_name) 
      
