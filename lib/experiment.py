from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    differential_lr_schedule: list
    weight_decay: float = 1e-4
    scheduler_t_max: int = 25
    model_name: str = 'resnet50'


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