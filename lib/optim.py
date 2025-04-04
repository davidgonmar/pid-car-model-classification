from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

@dataclass
class ExperimentConfig:
    differential_lr_schedule: list
    weight_decay: float = 1e-4
    scheduler_t_max: int = 25
    model_name: str = 'resnet50'

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

def get_model(config: ExperimentConfig, num_classes: int = 1000, in_channels: int = 3):
    di = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152
    }

    if config.model_name not in di:
        raise ValueError(f"Model {config.model_name} not supported. Choose from {list(di.keys())}")
    
    model_class = di[config.model_name]

    ret = model_class(num_classes=num_classes, in_channels=in_channels)

    if config.pretrained:
        ret.load_pretrained()
    return ret


good = ExperimentConfig(
    differential_lr_schedule=[
            {'params': 'layer4', 'lr': 1e-3},
            {'params': 'layer3', 'lr': 1e-4},
            {'params': 'fc', 'lr': 3e-5}
        ],
    weight_decay=1e-4,
    scheduler_t_max=25,
    model_name = 'resnet50',
    pretrained = True
)