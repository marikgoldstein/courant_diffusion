import torch
import torch.nn as nn
import numpy as np

# local
import utils

def get_process(config, device):

    pt = config.process_name

    if pt == 'cosine':
        process = CosineScheduleProcess(config, device)
    elif pt == 'rf_tied':
        process = RFScheduleProcessTiedDelta(config, device)
    elif pt == 'rf_const':
        process = RFScheduleProcessConstDelta(config, device)
    elif pt == 'linear_vp':
        process = LinearVP(config, device)
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

    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, t):
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

    def sample_base(self, batch_size):
        config = self.config
        return torch.randn(batch_size, config.C, config.H, config.W).to(self.device)

    def compute_xt(self, x0, x1, coefs):                
        left = utils.bcast_right(coefs.a_t, x0.ndim) * x0
        right = utils.bcast_right(coefs.s_t, x1.ndim) * x1
        return left + right

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


class CosineScheduleProcess(GaussianProcess):

 
    def __init__(self, config, device):
        super().__init__(config, device)

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

class RFScheduleProcess(GaussianProcess):

    def __init__(self, config, device):
        super().__init__(config, device)

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

class RFScheduleProcessTiedDelta(RFScheduleProcess):

    def delta(self, t):
        t = t.clamp(min=0.0, max=.9999)
        return t / (1-t)

class RFScheduleProcessConstDelta(RFScheduleProcess):
   
    def delta(self, time):
        return torch.ones_like(time) * self.config.delta_const

class LinearVP(GaussianProcess):

    def __init__(self, config, device):
        super().__init__(config, device)
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

    def f(self, time):
        raise NotImplementedError()
        #return -0.5 * (self.beta_min + torch.tensor(time) * (self.beta_max - self.beta_min))
    
    def g(self, time):
        return torch.sqrt(self.beta_min + time * (self.beta_max - self.beta_min))

    def delta(self, time):
        return self.g(time).pow(2) / 2.0


