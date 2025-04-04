import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Union, List
import torchvision
# -----------------------------------------------------------------------------
# Block definitions modeled after torchvision's implementation.
# -----------------------------------------------------------------------------

resnset = torchvision.models.resnet.ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# -----------------------------------------------------------------------------
# ResNet Base Class
# -----------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
                 num_classes: int = 1000, in_channels: int = 3):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int,
                    blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def load_pretrained(self):
        """
        Loads pretrained weights from torchvision's corresponding ResNet model.
        The model class name (e.g. "ResNet18") is used to select the correct weights.
        """
        import torchvision.models as models

        model_name = self.__class__.__name__.lower()  # e.g. "resnet18"
        if model_name == "resnet18":
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == "resnet34":
            pretrained_model = models.resnet34(pretrained=True)
        elif model_name == "resnet50":
            pretrained_model = models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            pretrained_model = models.resnet101(pretrained=True)
        elif model_name == "resnet152":
            pretrained_model = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Pretrained weights for {model_name} are not available.")
        
        if self.num_classes != pretrained_model.fc.out_features:
            # do not load the final fully connected layer
            del pretrained_model.fc
        self.load_state_dict(pretrained_model.state_dict(), strict=False)
        return self

    @classmethod
    def from_pretrained(cls, num_classes: int = 1000, in_channels: int = 3):
        """
        Constructs the model and loads pretrained weights.
        Note: If num_classes is different from the pretrained model (default 1000),
        the final fc layer dimensions might not match.
        """
        model = cls(num_classes=num_classes, in_channels=in_channels)
        model.load_pretrained()
        return model

    def load_from_torchvision(self, model):
        """
        Loads weights from a torchvision model.
        This is useful for transferring weights from torchvision to a custom model.
        """
        if isinstance(model, resnset):
            self.load_state_dict(model.state_dict(), strict=False)
        else:
            raise ValueError("Provided model is not a torchvision ResNet model.")


# -----------------------------------------------------------------------------
# ResNet Variants
# -----------------------------------------------------------------------------

class ResNet18(ResNet):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2],
                                       num_classes=num_classes, in_channels=in_channels)


class ResNet34(ResNet):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3],
                                       num_classes=num_classes, in_channels=in_channels)


class ResNet50(ResNet):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3],
                                       num_classes=num_classes, in_channels=in_channels)


class ResNet101(ResNet):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3],
                                        num_classes=num_classes, in_channels=in_channels)


class ResNet152(ResNet):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3],
                                        num_classes=num_classes, in_channels=in_channels)
