from models import resnet
import torchvision

model_ours = resnet.ResNetBase()
model_torchvision = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.DEFAULT
)

model_ours.load_from_torchvision(model_torchvision.state_dict())
