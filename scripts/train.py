import torch
from utils.dataset import CarsDataset
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from lib.optim import get_optimizer_and_scheduler
from lib.resnet import get_model
from lib.experiment import get_config
import os
import argparse


torch.set_float32_matmul_precision("high")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


parser = argparse.ArgumentParser(description="Train ResNet on Stanford Cars dataset")
parser.add_argument(
    "--plot", action="store_true", help="Plot training and test loss/accuracy"
)
parser.add_argument("--experiment", type=str, default="1", help="Experiment name")


args = parser.parse_args()

config = get_config(args.experiment)


args = parser.parse_args()

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    ),
}


train_dataset = CarsDataset("data", split="train", transform=data_transforms["train"])
test_dataset = CarsDataset("data", split="test", transform=data_transforms["test"])
subset_size = 5000

indices = torch.randperm(len(test_dataset))[:subset_size]
test_dataloader = DataLoader(Subset(test_dataset, indices), batch_size=64)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

resnet = get_model(config, num_classes=train_dataset.num_classes).to(device)

optimizer, scheduler = get_optimizer_and_scheduler(resnet, config)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += torch.nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return avg_loss, accuracy


train_losses, test_losses, test_accuracies = [], [], []


def plot_results(train_losses, test_losses, test_accuracies):
    """
    Plots training and test losses alongside test accuracy per interval.

    Parameters:
    - train_losses (list or array-like): Training loss values.
    - test_losses (list or array-like): Test loss values.
    - test_accuracies (list or array-like): Test accuracy values.
    """
    plt.figure(figsize=(14, 5))

    # Plotting Train and Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker="o", label="Train Loss")
    plt.plot(test_losses, marker="x", label="Test Loss")
    plt.xlabel("Interval")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss per Interval")
    plt.legend()
    plt.grid(True)

    # Plotting Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, marker="s", label="Test Accuracy")
    plt.xlabel("Interval")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy per Interval")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


resnet.to(device)

compile = True
# do not compile on windows
if compile and os.name != "nt":
    resnet = torch.compile(resnet)

interval = len(train_dataloader) // 2


if __name__ == "__main__":
    for ep in range(100):
        resnet.train()
        running_loss = 0.0
        scheduler.step()
        for batch_idx, (data, target) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {ep+1}/100"), 1
        ):
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

                test_loss, test_accuracy = evaluate_model(
                    resnet, test_dataloader, device
                )
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

                if args.plot:
                    plot_results(train_losses, test_losses, test_accuracies)

                print(
                    f"Epoch [{ep+1}/100], Interval [{batch_idx}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
                )
                torch.save(resnet, f"checkpoints/experiment_{args.experiment}.pth")
