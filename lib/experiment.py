from dataclasses import dataclass
import torchvision
import torch

@dataclass
class ExperimentConfig:
    transforms: dict
    lr_schedule: list
    weight_decay: float
    model_name: str
    pretrained: bool
    epochs: int
    

transforms_data_augmentation = {
    "train": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "test": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
}

transforms_no_data_augmentation = {
    "train": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),

    "test": torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
}



# 
_1 = ExperimentConfig(
    transforms=transforms_data_augmentation,
    lr_schedule=[
        {"params": "layer3", "lr": 3e-5},
        {"params": "layer4", "lr": 1e-4},
        {"params": "fc", "lr": 1e-3},
    ],
    weight_decay=1e-4,
    model_name="resnet50",
    pretrained=True,
    epochs=30,
)

_2 = ExperimentConfig(
    transforms=transforms_data_augmentation,
    lr_schedule=[
        {"params": "layer3", "lr": 3e-5},
        {"params": "layer4", "lr": 1e-4},
        {"params": "fc", "lr": 1e-3},
    ],
    weight_decay=1e-4,
    model_name="resnet18",
    pretrained=True,
    epochs=30,
)

# No differential LR
_3 = ExperimentConfig(
    transforms=transforms_data_augmentation,
    lr_schedule=[{"params": "", "lr": 1e-3}],
    weight_decay=1e-4,
    model_name="resnet50",
    pretrained=True,
    epochs=30,
)

_4 = ExperimentConfig(
    transforms=transforms_data_augmentation,
    lr_schedule=[{"params": "", "lr": 1e-3}],
    weight_decay=1e-4,
    model_name="resnet50",
    pretrained=True,
    epochs=30,
)

# No data augmentation
_5 = ExperimentConfig(
    transforms=transforms_no_data_augmentation,
    lr_schedule=[{"params": "", "lr": 1e-3}],
    weight_decay=1e-4,
    model_name="resnet50",
    pretrained=True,
    epochs=30,
)

_6 = ExperimentConfig(
    transforms=transforms_no_data_augmentation,
    lr_schedule=[{"params": "", "lr": 1e-3}],
    weight_decay=1e-4,
    model_name="resnet50",
    pretrained=True,
    epochs=30,
)


configs = {"1": _1, "2": _2, "3": _3, "4": _4, "5": _5, "6": _6}


def get_config(config_name: str) -> ExperimentConfig:
    """
    Get the experiment configuration by name.
    """
    if config_name not in configs:
        raise ValueError(
            f"Config {config_name} not found. Available configs: {list(configs.keys())}"
        )

    return configs[config_name]
