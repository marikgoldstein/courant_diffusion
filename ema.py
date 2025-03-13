import torch
import torch.nn as nn
from collections import OrderedDict

# copied from https://github.com/facebookresearch/DiT
def update_ema(model, model_ema, decay):                                                                                                                                                                                                                                                                                                                                                                                
    if decay == 0.0 or decay == 0:
        print("WARNING. DOING A FULL EMA COPY")

    ema_params = OrderedDict(model_ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    # just in case it is wrapped in ddp
    model_params = {k.replace('module.','') : v for k,v in model_params.items()}

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def wipe_ema(model, model_ema):
    update_ema(
        model = model,
        model_ema = model_ema,
        decay = 0.0,
    )

def maybe_update_emas(network, network_ema, config, step, first_update):
    c = config
    cond1 = (step >= c.update_ema_after)
    cond2 = (step % c.update_ema_every == 0)

    updated = False

    if cond1 and cond2:

        if first_update:
            print("ASSUMING THIS IS FIRST UPDATE OF EMA MODEL. DOING A FULL COPY")

        update_ema(
            model = network.module,
            model_ema = network_ema,
            decay = 0.0 if first_update else c.ema_decay
        )

        updated = True

    return updated




