import torch.nn as nn
from torch import Tensor
from typing import Literal


class BasicBlock(nn.Module):
    KERNEL_SIZE = 3

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2], "Stride should be either 1 or 2"
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.downsample = None

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.KERNEL_SIZE,
            padding=1,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=self.KERNEL_SIZE,
            padding=1,
            stride=stride,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU()

    def forward(self, x: Tensor):
        identity = self.downsample(x) if self.downsample is not None else x

        x = self.activ(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += identity

        x = self.activ(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1
    ):
        super(BottleneckBlock, self).__init__()
        assert stride in [1, 2], "Stride should be either 1 or 2"

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.downsample = None

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU()

    def forward(self, x: Tensor):
        identity = self.downsample(x) if self.downsample is not None else x

        x = self.activ(self.bn1(self.conv1(x)))
        x = self.activ(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += identity

        x = self.activ(x)

        return x


class ResNet(nn.Module):
    conv2_x: nn.Module
    conv3_x: nn.Module
    conv4_x: nn.Module
    conv5_x: nn.Module

    def __init__(
        self,
        conv_out_channels: int,
        in_channels=3,
        num_classes=10,
    ):
        super(
            ResNet,
            self,
        ).__init__()
        self.conv1 = nn.Conv2d(
            kernel_size=7, in_channels=in_channels, out_channels=64, stride=2, padding=5
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=conv_out_channels, out_features=num_classes)
        self.flatten = nn.Flatten()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        type: Literal["basic", "bottleneck"] = "basic",
        n_blocks: int = 2,
        mid_channels: int = None,
    ):
        layers = []
        if type == "bottleneck":
            assert mid_channels is not None, "Should pass `mid_channels` for bottleneck"

        stride = 2

        if type == "basic":
            layers.append(
                BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )
        elif type == "bottleneck":
            layers.append(
                BottleneckBlock(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )
        else:
            raise ValueError(
                f"Unknown type of block: {type}. Should be one of 'basic' or 'bottleneck'"
            )
        for i in range(1, n_blocks):
            if type == "basic":
                layers.append(
                    BasicBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                    )
                )
            elif type == "bottleneck":
                layers.append(
                    BottleneckBlock(
                        in_channels=out_channels,
                        mid_channels=mid_channels,
                        out_channels=out_channels,
                        stride=1,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown type of block: {type}. Should be one of 'basic' or 'bottleneck'"
                )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.flatten(self.avg_pool(x))
        return self.fc(x)


class ResNet18(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet18,
            self,
        ).__init__(512, in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64, out_channels=128, n_blocks=2, type="basic"
        )
        self.conv3_x = self._make_layer(
            in_channels=128, out_channels=256, n_blocks=2, type="basic"
        )
        self.conv4_x = self._make_layer(
            in_channels=256, out_channels=512, n_blocks=2, type="basic"
        )
        self.conv5_x = self._make_layer(
            in_channels=512, out_channels=512, n_blocks=2, type="basic"
        )


class ResNet34(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet34,
            self,
        ).__init__(512, in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64, out_channels=128, n_blocks=3, type="basic"
        )
        self.conv3_x = self._make_layer(
            in_channels=128, out_channels=256, n_blocks=4, type="basic"
        )
        self.conv4_x = self._make_layer(
            in_channels=256, out_channels=512, n_blocks=6, type="basic"
        )
        self.conv5_x = self._make_layer(
            in_channels=512, out_channels=512, n_blocks=3, type="basic"
        )


class ResNet50(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet50,
            self,
        ).__init__(2048, in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64,
            mid_channels=64,
            out_channels=256,
            n_blocks=3,
            type="bottleneck",
        )
        self.conv3_x = self._make_layer(
            in_channels=256,
            mid_channels=128,
            out_channels=512,
            n_blocks=4,
            type="bottleneck",
        )
        self.conv4_x = self._make_layer(
            in_channels=512,
            mid_channels=256,
            out_channels=1024,
            n_blocks=6,
            type="bottleneck",
        )
        self.conv5_x = self._make_layer(
            in_channels=1024,
            mid_channels=512,
            out_channels=2048,
            n_blocks=3,
            type="bottleneck",
        )


class ResNet101(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet101,
            self,
        ).__init__(2048, in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64,
            mid_channels=64,
            out_channels=256,
            n_blocks=3,
            type="bottleneck",
        )
        self.conv3_x = self._make_layer(
            in_channels=256,
            mid_channels=128,
            out_channels=512,
            n_blocks=4,
            type="bottleneck",
        )
        self.conv4_x = self._make_layer(
            in_channels=512,
            mid_channels=256,
            out_channels=1024,
            n_blocks=23,
            type="bottleneck",
        )
        self.conv5_x = self._make_layer(
            in_channels=1024,
            mid_channels=512,
            out_channels=2048,
            n_blocks=3,
            type="bottleneck",
        )


class ResNet152(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet152,
            self,
        ).__init__(2048, in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64,
            mid_channels=64,
            out_channels=256,
            n_blocks=3,
            type="bottleneck",
        )
        self.conv3_x = self._make_layer(
            in_channels=256,
            mid_channels=128,
            out_channels=512,
            n_blocks=8,
            type="bottleneck",
        )
        self.conv4_x = self._make_layer(
            in_channels=512,
            mid_channels=256,
            out_channels=1024,
            n_blocks=36,
            type="bottleneck",
        )
        self.conv5_x = self._make_layer(
            in_channels=1024,
            mid_channels=512,
            out_channels=2048,
            n_blocks=3,
            type="bottleneck",
        )