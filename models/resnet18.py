import torch
import torch.nn as nn

class ResNetBase(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNetBase, self).__init__()
        
        # Initial layer: 7x7 Conv, Stride 2
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        
        # Sequential for residual blocks
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 64),
            nn.ReLU(),
            ResNetBlock(64, 64),
            nn.ReLU(),
            ResNetBlock(64, 128, stride=2),
            nn.ReLU(),
            ResNetBlock(128, 128),
            nn.ReLU(),
            ResNetBlock(128, 256, stride=2),
            nn.ReLU(),
            ResNetBlock(256, 256),
            nn.ReLU(),
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512)
        )
        
        # Capa de clasificación
        self.classification_head = ClassificationHead(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.resnet_blocks(x)
        x = self.classification_head(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_activation=True):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.use_activation = use_activation  # activate or deactivate ReLU
        
        if self.use_activation:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_activation:
            x = self.relu(x)
        x = self.conv2(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes=1000):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1 Average Pooling
        self.fc = nn.Linear(in_features, num_classes)  # Fully Connected Layer
        self.softmax = nn.Softmax(dim=1)  # Softmax layer

    def forward(self, x):
        x = self.global_avg_pool(x)  
        x = x.view(x.shape[0], -1)   
        x = self.fc(x)              
        x = self.softmax(x)          
        return x
