import torch
import torch.nn as nn
import numpy as np
# local

def get_process(config):

    pt = config.process_name

    if pt == 'cosine':
        process = CosineScheduleProcess(config)
    elif pt == 'special':
        process = SpecialProcess(config)
    elif pt == 'rf_tied':
        process = RFScheduleProcessTiedDelta(config)
    elif pt == 'rf_const':
        process = RFScheduleProcessConstDelta(config)
    elif pt == 'linear_vp':
        process = LinearVP(config)
    elif pt == 'learned_tied':
        process = LearnedTiedProcess(config)
    else:
        assert False
    return process

class Coefs:
    def __init__(self, a_t, s_t, a_dot, s_dot, delta_t):
        self.a_t = a_t
        self.s_t = s_t
        self.a_dot = a_dot
        self.s_dot = s_dot
        self.delta_t = delta_t

class GaussianProcess:

    def __init__(self, config):
        self.config = config

    def forward(self, t):
        a_t = self.alpha(t)
        s_t = self.sigma(t)
        a_dot = self.alpha_dot(t)
        s_dot = self.sigma_dot(t)
        delta_t = self.delta(t)
        assert torch.all(
            delta_t >= 0.0
        )
        return Coefs(
            a_t = a_t, 
            s_t = s_t, 
            a_dot = a_dot,
            s_dot = s_dot, 
            delta_t = delta_t
        )

    @property
    def max_log_snr(self):
        return self.config.max_log_snr

    @property
    def final_time(self):
        return 1.0

    @property
    def final_std(self,):
        return self.sigma(torch.tensor(self.final_time))

    def alpha(self, t):
        raise NotImplementedError()
 
    def sigma(self, t):
        raise NotImplementedError()

    def alpha_dot(self, t):
        raise NotImplementedError()
 
    def sigma_dot(self, t):
        raise NotImplementedError()

    def f(self, t):
        raise NotImplementedError()

    def g(self, t):
        raise NotImplementedError()

    def delta(self ,t):
        raise NotImplementedError()

    def log_snr(self, t):
        return torch.log(
            self.alpha(t).pow(2) / self.sigma(t).pow(2)
        ).squeeze(-1).squeeze(-1).squeeze(-1)

    def inverse_log_snr(self, lam):
        raise NotImplementedError()

    def minimum_time(self,):
        max_lam = torch.tensor(self.max_log_snr)
        min_time = self.inverse_log_snr(max_lam)
        assert min_time >= 0.0
        assert min_time <= 1.0
        return min_time.item()

    def ratio_alpha(self, s, t):
        return self.alpha(t) / self.alpha(s) #? 

    def ratio_sigma(self, s, t):
        return self.sigma(s) / self.sigma(t) #? 

    def ratio(self, i, j, s, t): # ? 
        ratio_alpha_i = torch.pow(self.ratio_alpha(s,t), i)
        ratio_sigma_j = torch.pow(self.ratio_sigma(s,t), j)
        return ratio_alpha_i * ratio_sigma_j

class CosineScheduleProcess(GaussianProcess):

 
    def __init__(self, config):
        super().__init__(config)

    def alpha(self, t):
        return torch.cos(0.5 * np.pi * t)
    
    def sigma(self, t):
        return torch.sin(0.5 * np.pi * t)

    def alpha_dot(self, t):
        return -.5 * np.pi * torch.sin(0.5 * np.pi * t)

    def sigma_dot(self, t):
        return 0.5 * np.pi * torch.cos(0.5 * np.pi * t)

    def f(self, t):
        return -.5 * np.pi * torch.tan(0.5 * np.pi * t)

    def g(self, t):
        t = t.clamp(min=0.0, max=.9999)
        return torch.sqrt(np.pi * torch.tan(0.5 * np.pi * t))

    def delta(self, t):
        return self.g(t).pow(2) / 2.0

    def inverse_log_snr(self, lam):
        return (2/np.pi) * torch.atan(torch.exp(-lam/2))

class RFScheduleProcess(GaussianProcess):

    def __init__(self, config):
        super().__init__(config)

    def alpha(self, t):
        return 1 - t

    def sigma(self, t):
        return t
    
    def alpha_dot(self, t):
        return -1.0 * torch.ones_like(t)

    def sigma_dot(self, time):
        return torch.ones_like(time)

    def delta(self, time):
        raise NotImplementedError()

    def f(self, time):
        raise NotImplementedError()

    def g(self, time):
        return torch.sqrt(2.0 * self.delta(time))

    def inverse_log_snr(self, lam):
        return (-lam / 2).sigmoid()

class RFScheduleProcessTiedDelta(RFScheduleProcess):

    def delta(self, t):
        t = t.clamp(min=0.0, max=.9999)
        return t / (1-t)

class RFScheduleProcessConstDelta(RFScheduleProcess):
   
    def delta(self, time):
        return torch.ones_like(time) * self.config.delta_const

class LinearVP(GaussianProcess):

    def __init__(self, config):
        super().__init__(config)
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max

    def alpha(self, time):
        return torch.exp(-0.5 * (self.beta_min * time + 0.5 * time**2 * (self.beta_max - self.beta_min)))

    def sigma(self, time):
        return torch.sqrt(1 -  self.alpha(time)**2)

    def alpha_dot(self, time):
        r = -0.5 * (time * (self.beta_max - self.beta_min) + self.beta_min)
        return self.alpha(time) * r

    def sigma_dot(self, time):
        return (
            -self.alpha_dot(time) * self.alpha(time) / self.sigma(time)
        )
    
    def inverse_log_snr(self, lam):
        z = self.beta_min**2 + 2.0 * (self.beta_max - self.beta_min) * np.log(1 + np.exp(-lam))
        numer = -1.0 * self.beta_min + np.sqrt(z)
        denom = self.beta_max - self.beta_min
        out = numer / denom
        assert len(out.shape) <= 1
        return out

    def f(self, time):
        raise NotImplementedError()
        #return -0.5 * (self.beta_min + torch.tensor(time) * (self.beta_max - self.beta_min))
    
    def g(self, time):
        return torch.sqrt(self.beta_min + time * (self.beta_max - self.beta_min))

    def delta(self, time):
        return self.g(time).pow(2) / 2.0


