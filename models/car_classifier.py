import torch
import torch.nn as nn
import torchvision.models as models


class CarModelClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        """
        Initialize the car model classifier.
        Args:
            num_classes (int): Number of car model classes
            model_name (str): Name of the base model to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(CarModelClassifier, self).__init__()
        
        # Load the pre-trained model
        if model_name == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        
        def forward(self, x):
            features = self.base_model(x)
            features = torch.flatten(features, 1)  # Asegurarse de aplanar bien
            return self.classifier(features)
