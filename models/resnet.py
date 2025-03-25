import torch
import torch.nn as nn

class ResNetBase(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBase, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=3)
        

        
        # 3. Global Average Pooling (GAP)
        # Reduce la dimensión espacial de las características a un solo valor por canal usando 
        # el promedio global en cada canal.
        
        # 4. Capa de Clasificación (Fully Connected Layer)
        # Se usa una capa completamente conectada para transformar las características en 
        # probabilidades sobre las clases de salida.

    def forward(self, x):
        # Aquí se definiría el flujo de datos a través de cada una de las partes mencionadas.
        pass


class ResNetBlock(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes=1000):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1 Average Pooling
        self.fc = nn.Linear(in_features, num_classes)  # Capa totalmente conectada
        self.softmax = nn.Softmax(dim=1)  # Función Softmax

    def forward(self, x):
        x = self.global_avg_pool(x)  # Reduce a (batch_size, in_features, 1, 1)
        x = x.view(x.shape[0], -1)   # Aplana a (batch_size, in_features)
        x = self.fc(x)               # Pasa por la capa totalmente conectada
        x = self.softmax(x)          # Softmax para convertir en probabilidades
        return x
