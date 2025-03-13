import torch
import torch.nn as nn

# local
import utils
        
def get_train_time_sampler(config, device):

    if config.time_sampler == 'unif':
        return UnifSampler(config, device)
    elif config.time_sampler == 'logit_normal':
        return LogitNormalsampler(config, device)
    else:
        assert False, 'unknown time sampler'

class UnifSampler:
    
    def __init__(self, config, device):
        self.T_min = config.T_min_training
        self.T_max = config.T_max_training
        self.time_dist = utils.Unif(self.T_min, self.T_max)
        self.device = device
    def __call__(self, N):
        return self.time_dist.sample((N,)).to(self.device)

class LogitNormalSampler:                                                                                                                                     

    def __init__(self, config, device):
        self.T_min = config.T_min_training
        self.T_max = config.T_max_training
        self.mu = 0.0
        self.device = device

    def __call__(self, N):
        z = torch.randn(N).to(self.device) + self.mu
        t = z.sigmoid()
        t = torch.clip(t, self.T_min, self.T_max)    
        return t
