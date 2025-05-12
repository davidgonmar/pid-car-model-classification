import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lib.dataset import CarsDataset
from lib.optim import get_optimizer_and_scheduler
from lib.resnet import get_model
from lib.experiment import get_config
import time

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return avg_loss, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on Cars Dataset")
    parser.add_argument("--experiment", type=str, default="1")
    parser.add_argument("--train_bs", type=int, default=128)
    parser.add_argument("--test_bs", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=40)
    args = parser.parse_args()

    config = get_config(args.experiment)
    data_transforms = config.transforms
    train_dataset = CarsDataset("data", split="train", transform=data_transforms["train"])
    test_dataset = CarsDataset("data", split="test", transform=data_transforms["test"])

    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, num_classes=train_dataset.num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    writer = SummaryWriter(log_dir=f"logs/exp_{args.experiment}_{time.strftime('%Y%m%d_%H%M%S')}")
    global_step = 0

    n_global_steps = len(train_loader) * config.epochs
    for epoch in range(1, config.epochs + 1):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"), 1):
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            global_step += 1

            if batch_idx % args.log_every == 0 or global_step == n_global_steps:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
                for name, param in model.named_parameters():
                    writer.add_scalar(f"WeightNorm/{name}", param.norm().item(), global_step)
                    if param.grad is not None:
                        writer.add_scalar(f"GradNorm/{name}", param.grad.norm().item(), global_step)

            if batch_idx % args.eval_every == 0 or global_step == n_global_steps:
                test_loss, test_acc = evaluate_model(model, test_loader, device)
                writer.add_scalar("Eval/Loss", test_loss, global_step)
                writer.add_scalar("Eval/Accuracy", test_acc, global_step)
                print(f"Epoch {epoch}, Step {batch_idx}, Train Loss: {loss.item():.4f}, Eval Loss: {test_loss:.4f}, Eval Acc: {test_acc:.2f}%")
                torch.save(model.state_dict(), f"checkpoints/exp_{args.experiment}.pth")

        scheduler.step()

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    writer.add_scalar("Final/Eval/Loss", test_loss, global_step)
    writer.add_scalar("Final/Eval/Accuracy", test_acc, global_step)

    writer.close()
