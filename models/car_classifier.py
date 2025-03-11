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
            self.base_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        features = self.base_model(x)
        return self.classifier(features) 