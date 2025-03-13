import torch
import torch.nn as nn

# local
import utils
        
def get_train_time_sampler(config):

    if config.time_sampler == 'unif':
        return UnifSampler(config)
    elif config.time_sampler == 'logit_normal':
        return LogitNormalsampler(config)
    else:
        assert False, 'unknown time sampler'

class UnifSampler:
    
    def __init__(self, config):
        self.T_min = config.T_min_training
        self.T_max = config.T_max_training
        self.time_dist = utils.Unif(self.T_min, self.T_max)
   
    def __call__(self, N, process, device):
        t = self.time_dist.sample((N,)).to(device)
        return t

class LogitNormalSampler:                                                                                                                                     

    def __init__(self, config):
        self.T_min = config.T_min_training
        self.T_max = config.T_max_training
        self.mu = 0.0

    def __call__(self, N, process, device):
        z = torch.randn(N).to(device) + self.mu
        t = z.sigmoid()
        t = torch.clip(t, self.T_min, self.T_max)    
        return t
