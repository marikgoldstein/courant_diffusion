import torch
import numpy as np

def step_optimizer(network, optimizer, loss, config, step):

    optimizer.zero_grad()

    loss.backward()
    old_norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(), 
        config.grad_clip_norm
    )
    optimizer.step()

    old_lr, new_lr = adjust_learning_rate(
	optimizer, step, config
    )
    return old_norm, old_lr, new_lr


def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):                                                    
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
    

def _cosine_decay(lr, current_step, total_steps):
    ratio = max(0.0, current_step / total_steps)
    mult = 0.5 * (1.0 + torch.cos(torch.tensor(np.pi * ratio)))
    return mult * lr

def _get_learning_rate(
    step,
    base_learning_rate,
    num_steps,
    warmup_steps,
    schedule_type
):

    assert schedule_type in ['constant', 'cosine']

    warmup = min(1.0, step / warmup_steps)

    if schedule_type == 'cosine':
        lr = _cosine_decay(base_learning_rate, step - warmup_steps, num_steps - warmup_steps)

    else:
        lr = base_learning_rate

    return lr * warmup

def adjust_learning_rate(optimizer, step, config):

    c = config
    #power = 1.0

    new_lr = _get_learning_rate(
        step=step,
        base_learning_rate= c.base_learning_rate,
        num_steps = c.num_training_steps,
        warmup_steps = c.warmup_steps,
        schedule_type = c.lr_schedule,
    )

    new_lr = max(new_lr, c.min_learning_rate)

    for pgroup in optimizer.param_groups:
        old_lr = pgroup['lr']
        pgroup["lr"] = new_lr

    return old_lr, new_lr


def setup_optimizer(model, config):
    c = config
    p = get_grouped_params(model, weight_decay=c.weight_decay)
    betas = (c.adam_b1, c.adam_b2)
    adam_args = {'betas': betas, 'eps': c.adam_eps, 'weight_decay': c.weight_decay}
    return torch.optim.AdamW(p, lr=c.base_learning_rate, **adam_args)                                                                             


