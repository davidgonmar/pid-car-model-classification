from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    differential_lr_schedule: list
    weight_decay: float = 1e-4
    scheduler_t_max: int = 25
    model_name: str = 'resnet50'
    pretrained: bool = True


_1 = ExperimentConfig(
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

_2 = ExperimentConfig(
    differential_lr_schedule=[
            {'params': 'layer4', 'lr': 1e-3},
            {'params': 'layer3', 'lr': 1e-4},
            {'params': 'fc', 'lr': 3e-5}
        ],
    weight_decay=1e-4,
    scheduler_t_max=25,
    model_name = 'resnet18',
    pretrained = True
)

configs = {
    "1": _1,
    "2": _2
}

def get_config(config_name: str) -> ExperimentConfig:
    """
    Get the experiment configuration by name.
    """
    if config_name not in configs:
        raise ValueError(f"Config {config_name} not found. Available configs: {list(configs.keys())}")
    
    return configs[config_name]