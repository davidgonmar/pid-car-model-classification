import torch
import torchvision
from utils.dataset import CarsDataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.LazyLinear(196)
r = resnet.to(device)

torch.set_float32_matmul_precision('high')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),   
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
}


train_dataset = CarsDataset("data", split="train", transform=data_transforms['train'])
test_dataset = CarsDataset("data", split="test", transform=data_transforms['test'])
subset_size = 5000

indices = torch.randperm(len(test_dataset))[:subset_size]
test_dataloader = DataLoader(Subset(test_dataset, indices), batch_size=64, num_workers=4)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)


# Freeze layers except for the last block and the fully connected layer
for name, param in resnet.named_parameters():
    if not ("layer4" in name or "layer3" in name or "fc" in name):
        param.requires_grad = False

# Set up the optimizer with differential learning rates:
# - New fc layer: higher lr (e.g., 1e-3)
# - Unfrozen pretrained layers: lower lr (e.g., 1e-4)
USE_DIFFERENTIAL_LR = True

if USE_DIFFERENTIAL_LR:
    optimizer = optim.AdamW([
        {'params': resnet.fc.parameters(), 'lr': 1e-3}, # layer 3
        {'params': [param for name, param in resnet.named_parameters()
                    if param.requires_grad and "fc" not in name and "layer3" not in name], 'lr': 1e-4}, # layer 4
        {'params': [param for name, param in resnet.named_parameters()
                    if param.requires_grad and "fc" not in name and "layer4" not in name], 'lr': 3e-5} # layer 3
    ], weight_decay=1e-4)
else:
    optimizer = optim.AdamW(resnet.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing scheduler for state-of-the-art LR scheduling
scheduler = CosineAnnealingLR(optimizer, T_max=25)

criterion = nn.CrossEntropyLoss()


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    return avg_loss, accuracy

train_losses, test_losses, test_accuracies = [], [], []

resnet.to(device)

compile = True
if compile:
    resnet = torch.compile(resnet)

interval = len(train_dataloader) // 2

for ep in range(100):
    resnet.train()
    running_loss = 0.0
    scheduler.step()
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader, desc=f'Epoch {ep+1}/100'), 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = resnet(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % interval == 0:
            avg_train_loss = running_loss / interval
            train_losses.append(avg_train_loss)
            running_loss = 0.0

            test_loss, test_accuracy = evaluate_model(resnet, test_dataloader, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            PLT = False
            if PLT:
                plt.figure(figsize=(14,5))

                plt.subplot(1,2,1)
                plt.plot(train_losses, marker='o', label='Train Loss')
                plt.plot(test_losses, marker='x', label='Test Loss')
                plt.xlabel('Interval')
                plt.ylabel('Loss')
                plt.title('Train and Test Loss per Interval')
                plt.legend()
                plt.grid(True)

                plt.subplot(1,2,2)
                plt.plot(test_accuracies, marker='s', label='Test Accuracy')
                plt.xlabel('Interval')
                plt.ylabel('Accuracy (%)')
                plt.title('Test Accuracy per Interval')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.show()

            print(f'Epoch [{ep+1}/100], Interval [{batch_idx}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')