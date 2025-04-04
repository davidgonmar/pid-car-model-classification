from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib.experiment import ExperimentConfig


def get_optimizer_and_scheduler(model, config: ExperimentConfig):

    for param in model.parameters():
        param.requires_grad = False

    param_groups = []
    matched_params = set()

    for group in config.differential_lr_schedule:
        layer_key = group['params']
        lr = group['lr']

        params = [param for name, param in model.named_parameters()
                  if layer_key in name and param not in matched_params]

        for param in params:
            param.requires_grad = True
            matched_params.add(param)

        if params:
            param_groups.append({'params': params, 'lr': lr})

    if not param_groups:
        raise ValueError("No parameters matched the differential LR schedule. Check your config.")

    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)

    return optimizer, scheduler

def fixed_lr(model, lr: float = 1e-3):
    return {
        'params': model.parameters(),
        'lr': lr
    }


