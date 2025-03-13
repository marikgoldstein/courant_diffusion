import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


'''
# simple net
class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim = 128, layers = 5):
        super().__init__()
        self.layers = layers
        self.in_fc = nn.Linear(in_dim + 1, hidden_dim)
        self.fclist = nn.ModuleList(
            [nn.Linear(hidden_dim + 1, hidden_dim) for i in range(layers)]
        )
        self.out_fc = nn.Linear(hidden_dim + 1, in_dim)

    def merge(self, h, t):
        return torch.cat([h, t], dim=-1)

    def forward(self, h, t):
        h = self.in_fc(self.merge(h, t)).relu()
        for i in range(self.layers):
            h = self.fclist[i](self.merge(h, t)).relu()
        return self.out_fc(self.merge(h, t))

'''



# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, rms_norm = False):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)

        if rms_norm:
            print("USING RMS NORM INSTEAD OF GROUP NORM")
            self.norm = RMSNorm(dim_out)
        else:
            print("GROUP NORM")
            self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8, num_blocks = 2, dropout = 0.0, rms_norm = False):
        super().__init__()

        if classes_emb_dim is None:
            classes_emb_dim = 0

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) 

        self.block1 = Block(dim, dim_out, groups = groups, rms_norm = rms_norm)
        self.block2 = Block(dim_out, dim_out, groups = groups, rms_norm = rms_norm)

        print("setting drop with ", dropout)
        self.drop_layer = nn.Dropout2d(dropout)


        assert num_blocks in [2, 4]
        self.num_blocks = num_blocks
        if num_blocks == 4:
            assert False
            self.block3 = Block(dim_out, dim_out, groups = groups)
            self.block4 = Block(dim_out, dim_out, groups = groups)
        else:
            self.block3 = None
            self.block4 = None

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        
        # MG: is this an okay place for this
        h = self.drop_layer(h)

        h = self.block2(h)

        if self.num_blocks > 2:
            h = self.block4(self.block3(h))

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qk_norm = False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        if qk_norm:
            self.q_norm = nn.LayerNorm(hidden_dim, elementwise_affine = False, bias = False)
            self.k_norm = nn.LayerNorm(hidden_dim, elementwise_affine = False, bias = False)
        else:
            self.q_norm = self.k_norm = nn.Identity()


        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):

        # (x,y) dims are H W	

        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        
        q, k, v = qkv

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h = self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h = self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qk_norm = False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads


        if qk_norm:
            self.q_norm = nn.LayerNorm(hidden_dim, elementwise_affine = False, bias = False)
            self.k_norm = nn.LayerNorm(hidden_dim, elementwise_affine = False, bias = False)
        else:
            self.q_norm = self.k_norm = nn.Identity()



        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
                
        q, k, v = qkv

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h = self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h = self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h = self.heads)

	#q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

def get_unet_lucid(config):

    #INPUT_HW = config.H_dgm
    kwargs = {
        'dim': config.unet_dim,
        'num_classes': config.num_classes_for_model if config.unet_use_classes else 0,
        'in_channels': config.C_dgm + config.cond_channels,
        'out_channels': config.C_dgm,
        'dim_mults': config.unet_dim_mults,
        'resnet_num_inner_blocks': config.unet_resnet_num_inner_blocks,
        'resnet_num_outer_blocks': config.unet_resnet_num_outer_blocks,
        'down_up' : config.unet_down_up,
        'dropout': config.unet_dropout,
        'rms_norm': config.unet_rms_norm,
        'qk_norm': config.unet_qk_norm,
        'full_attn': config.unet_full_attn,
    }
    
    print("Unet kwargs", kwargs)
    return Unet(**kwargs)
                                                                                                                                                                                     
class CondEmbedder(nn.Module):
    def __init__(self, cond_dim, hidden_size, num_classes):
        # [N, D] -> [N, hidden_size]

        super().__init__()

        self.num_classes = num_classes

        
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

    def forward(self, emb):

        x_mlp = self.mlp(emb)

        return x_mlp

    


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        num_classes = None,
        dim_mults=(1, 2, 4, 8),
        resnet_num_inner_blocks=2,
        resnet_num_outer_blocks=2,
        down_up = True,
        dropout = 0.0,
        rms_norm = False,
        qk_norm = False,
        full_attn = False
    ):
        
        super().__init__()
        attn_dim_head = 64
        attn_heads = 4

        #self.cond_drop_prob = cond_drop_prob
        # determine dimensions

        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding = 3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block_groups = 8
        
        assert resnet_num_inner_blocks in [2,4]
        assert resnet_num_outer_blocks in [2,4,8]
        
        self.resnet_num_outer_blocks = resnet_num_outer_blocks
        
        # time embeddings
        time_dim = dim * 4

        learned_sinusoidal_cond = True
        learned_sinusoidal_dim = 64
        random_fourier_features = False
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.use_classes = (num_classes is not None) and num_classes > 0
        if self.use_classes:
            self.classes_emb = nn.Embedding(num_classes, dim)
            #self.null_classes_emb = nn.Parameter(torch.randn(dim))
            classes_dim = dim * 4
            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim)
            )
        else:
            assert (num_classes is None) or (num_classes == 0)  
            classes_dim = None
            self.classes_mlp = None 

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)




        MaybeDown = (lambda a, b: Downsample(a, b)) if down_up else (lambda a, b: nn.Identity())
        MaybeUp = (lambda a, b: Upsample(a, b)) if down_up else (lambda a, b: nn.Identity())



        block_klass = partial(
            ResnetBlock, 
            groups = resnet_block_groups, 
            num_blocks = resnet_num_inner_blocks, 
            dropout = dropout, 
            rms_norm = rms_norm,
            time_emb_dim = time_dim,
            classes_emb_dim = classes_dim,
        )


        Attn = partial(
            Attention if full_attn else LinearAttention,
            heads = attn_heads,
            dim_head = attn_dim_head,
            qk_norm = qk_norm,
        )
        FullAttn = partial(
            Attention,
            heads = attn_heads,
            dim_head = attn_dim_head,
            qk_norm = qk_norm,
        )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)


            lst = []

            assert resnet_num_outer_blocks % 2 == 0

            for k in range(int(resnet_num_outer_blocks / 2)):

                lst += [block_klass(dim_in, dim_in), block_klass(dim_in, dim_in)]
                lst.append(Residual(PreNorm(dim_in, Attn(dim_in))))

            lst.append(
                MaybeDown(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            )


            self.downs.append(nn.ModuleList(lst))

        mid_dim = dims[-1]


        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, FullAttn(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)


            lst = []
            for k in range(int(resnet_num_outer_blocks / 2)):

                lst += [block_klass(dim_out + dim_in, dim_out), block_klass(dim_out + dim_in, dim_out)]
                lst.append(Residual(PreNorm(dim_out, Attn(dim_out))))

            lst.append(
                 MaybeUp(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            )


            self.ups.append(nn.ModuleList(lst))


        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, time, classes = None):


        batch, device = x.shape[0], x.device

        if (classes is not None) and self.use_classes: 
            classes_emb = self.classes_emb(classes)
            c = self.classes_mlp(classes_emb)
        else:   
            c = None

        # unet


        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []



        if self.resnet_num_outer_blocks == 2:


            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t, c)
                h.append(x)

                x = block2(x, t, c)
                x = attn(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t, c)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, c)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t, c)
                x = attn(x)

                x = upsample(x)

        
        elif self.resnet_num_outer_blocks == 4:


            for block1, block2, attn_A, block3, block4, attn_B, downsample in self.downs:


                x = block1(x, t, c)
                h.append(x)

                x = block2(x, t, c)
                x = attn_A(x)
                h.append(x)

                # added
                x = block3(x, t, c)
                h.append(x)

                x = block4(x, t, c)
                x = attn_B(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t, c)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, c)

            for block1, block2, attn_A, block3, block4, attn_B, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t, c)
                x = attn_A(x)

            
                # added
                x = torch.cat((x, h.pop()), dim = 1)
                x = block3(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = block4(x, t, c)
                x = attn_B(x)

                x = upsample(x)

        elif self.resnet_num_outer_blocks == 8:

            for b1, b2, attn_A, b3, b4, attn_B, b5, b6, attn_C, b7, b8, attn_D, downsample in self.downs:

                
                x = b1(x, t, c)
                h.append(x)

                x = b2(x, t, c)
                x = attn_A(x)
                h.append(x)

                x = b3(x, t, c)
                h.append(x)

                x = b4(x, t, c)
                x = attn_B(x)
                h.append(x)

                x = b5(x, t, c)
                h.append(x)

                x = b6(x, t, c)
                x = attn_C(x)
                h.append(x)

                x = b7(x, t, c)
                h.append(x)

                x = b8(x, t, c)
                x = attn_D(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t, c)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, c)


            for b1, b2, attn_A, b3, b4, attn_B, b5, b6, attn_C, b7, b8, attn_D, upsample in self.ups:

                x = torch.cat((x, h.pop()), dim = 1)
                x = b1(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b2(x, t, c)
                x = attn_A(x)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b3(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b4(x, t, c)
                x = attn_B(x)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b5(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b6(x, t, c)
                x = attn_C(x)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b7(x, t, c)

                x = torch.cat((x, h.pop()), dim = 1)
                x = b8(x, t, c)
                x = attn_D(x)
                x = upsample(x)


        else:

            assert False


        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

