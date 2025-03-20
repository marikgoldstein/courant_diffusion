import torch
import torch.nn as nn
import numpy as np

# local
import utils

def get_process(config, device):

    pt = config.process_name

    if pt == 'cosine':
        process = CosineVP(config, device)
    elif pt == 'rf':
        process = RectifiedFlow(config, device)
    elif pt == 'linear_vp':
        process = LinearVP(config, device)
    else:
        assert False
    return process

class Coefs:
    def __init__(self, alpha, sigma, alpha_dot, sigma_dot, delta, g, g_squared):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_dot = alpha_dot
        self.sigma_dot = sigma_dot
        self.delta = delta
        self.g = g
        self.g_squared = g_squared

class GaussianProcess:

    def __init__(self, config, device):
        self.config = config
        self.device = device

    def __call__(self, t):
        return Coefs(
            alpha = self.alpha(t),
            sigma = self.sigma(t),
            alpha_dot = self.alpha_dot(t),
            sigma_dot = self.sigma_dot(t),
            delta = self.delta(t),
            g = self.g(t),
            g_squared = self.g_squared(t),
        )

    def sample_base(self, batch_size):
        config = self.config
        return torch.randn(batch_size, config.C, config.H, config.W).to(self.device)

    def compute_xt(self, x0, x1, coefs):                
        left = utils.bcast_right(coefs.alpha, x0.ndim) * x0
        right = utils.bcast_right(coefs.sigma, x1.ndim) * x1
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

    def g_squared(self, t):
        return 2.0 * self.delta(t)

    def delta(self ,t):
        raise NotImplementedError()


class CosineVP(GaussianProcess):
 
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

class RectifiedFlow(GaussianProcess):

    def __init__(self, config, device):
        super().__init__(config, device)

    def alpha(self, t):
        return 1 - t

    def sigma(self, t):
        return t
    
    def alpha_dot(self, t):
        return -1.0 * torch.ones_like(t)

    def sigma_dot(self, t):
        return torch.ones_like(t)

    def delta(self, t):
        left = self.sigma(t) * self.sigma_dot(t)
        right = (self.alpha_dot(t) / self.alpha(t)) * self.sigma(t).pow(2)
        return left - right
        
    def f(self, t):
        raise NotImplementedError()

    def g(self, t):
        return torch.sqrt(2.0 * self.delta(t))

class LinearVP(GaussianProcess):

    def __init__(self, config, device):
        super().__init__(config, device)
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max

    def alpha(self, t):
        return torch.exp(-0.5 * (self.beta_min * t + 0.5 * t**2 * (self.beta_max - self.beta_min)))

    def sigma(self, t):
        return torch.sqrt(1 -  self.alpha(t)**2)

    def alpha_dot(self, t):
        r = -0.5 * (t * (self.beta_max - self.beta_min) + self.beta_min)
        return self.alpha(t) * r

    def sigma_dot(self, t):
        return (
            -self.alpha_dot(t) * self.alpha(t) / self.sigma(t)
        )

    def f(self, t):
        return -0.5 * (self.beta_min + torch.tensor(t) * (self.beta_max - self.beta_min))
    
    def g(self, t):
        return torch.sqrt(self.beta_min + t * (self.beta_max - self.beta_min))

    def delta(self, t):
        return self.g(t).pow(2) / 2.0
