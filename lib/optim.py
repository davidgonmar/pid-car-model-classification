from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer_and_scheduler(model, config):
    param_groups = []
    for group in config.lr_schedule:
        params = getattr(model, group["params"]).parameters() if group["params"] else model.parameters()
        keys = list(map(lambda x: (group["params"] + "." if group["params"] else "") + x[0], getattr(model, group["params"]).named_parameters() if group["params"] else model.named_parameters()))
        param_groups.append({"params": params, "lr": group["lr"]})
        print("Group:", keys, "LR:", group["lr"])

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = config.epochs
    warmup_steps = int(0.1 * total_steps)
    beta = 0.95

    def lr_lambda(current_epoch):
        if current_epoch < warmup_steps:
            return float(current_epoch) / float(max(1, warmup_steps))
        else:
            decay_epochs = current_epoch - warmup_steps
            return beta ** decay_epochs

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler