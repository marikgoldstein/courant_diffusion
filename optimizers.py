import torch
import numpy as np

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
    


class Optimizer:

    def __init__(
        self, 
        network, 
        grad_clip_norm, 
        weight_decay, 
        base_lr, 
        min_lr,
        warmup_steps, 
        lr_schedule_type, 
        num_training_steps,
        adam_b1,
        adam_b2,
        adam_eps,
    ):
        
        self.network = network
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lr_schedule_type = lr_schedule_type
        self.num_training_steps = num_training_steps

        grouped_params = get_grouped_params(network, weight_decay=weight_decay)
        betas = (adam_b1, adam_b2)
        adam_args = {'betas': betas, 'eps': adam_eps, 'weight_decay': weight_decay}
        self._optimizer = torch.optim.AdamW(
            grouped_params, lr=base_lr, **adam_args
        )

    def param_groups(self,):
        return self._optimizer.param_groups

    def state_dict(self,):
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        self._optimizer.load_state_dict(state)

    def set_device(self, device):
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def take_step(self, loss, step):
        self._optimizer.zero_grad()
        loss.backward()
        old_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 
            self.grad_clip_norm
        )
        self._optimizer.step()

        new_lr = self.get_lr(step)

        for pgroup in self.param_groups():
            old_lr = pgroup['lr']
            pgroup["lr"] = new_lr

        return old_norm, old_lr, new_lr


    def get_lr(self, current_step):

        assert self.lr_schedule_type in ['constant', 'cosine']

        # cosine schedule
        if self.lr_schedule_type == 'cosine':
            # relative to end of warmup
            relative_step = current_step - self.warmup_steps
            relative_total_steps = self.num_training_steps - self.warmup_steps
            ratio = max(0.0, relative_step / relative_total_steps)
            pi = torch.tensor(np.pi)
            mult = 0.5 * (1.0 + torch.cos(pi * ratio))
            lr = mult * self.base_lr

        # constant schedule
        else:
            lr = self.base_lr

        # just in case warmup is 0, this sets warmup_coef to 1.0
        warmup_coef = min(1.0, current_step / min(self.warmup_steps, 1))
        new_lr = warmup_coef * lr 

        # clip at min
        return new_lr.clamp(min=self.min_lr)

