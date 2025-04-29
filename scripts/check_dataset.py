from utils.dataset import CarsDataset
from torch.utils.data import DataLoader
from torchvision import transforms


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train = CarsDataset(root="data", split="train", transform=transform)
test = CarsDataset(root="data", split="test", transform=transform)

print(f"Number of classes: {train.num_classes}")
print(f"Number of training samples: {train.class_meta}")

train_loader = DataLoader(train, batch_size=1024, shuffle=True)
test_loader = DataLoader(test, batch_size=1024, shuffle=False)


# some code to test the dataset
for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    pass

for i, (images, labels) in enumerate(test_loader):
    print(images.shape)
    pass

print("✅ Dataset and DataLoader are working!")
