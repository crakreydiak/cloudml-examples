#
# Script adapted from: 
# https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/pipelines/2b_train_cifar_10_with_pytorch/train-model/main.py
# ==============================================================================


import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import save_on_master


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_epochs", default=1, type=int)
    parser.add_argument("--path_to_data", default="./data", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    return parser.parse_args()

class ConvNet(nn.Module):

    def __init__(self, in_chan: int, n_classes: int):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, X: torch.Tensor):
        out = F.relu(self.conv1(X))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = out.view(-1, 128 * 6 * 6)
        out = self.dropout(F.relu(self.fc1(out)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def train(
    train_ldr: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    criterion: nn.Module,
    rank: int,
    epoch: int 
):
    cur_loss = 0.

    for X, y in train_ldr:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        
        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        cur_loss = loss.item()

    print(
        "Rank %d: [%d] loss: %.3f" % (rank, epoch + 1, cur_loss)
    )


def main(args):
    # get PyTorch environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"""
    world_size: {world_size}
    rank: {rank}
    local_rank: {local_rank}
    """)

    distributed = world_size > 1
    
    # set device
    if distributed:
        device = torch.device("cuda", local_rank)
        print("using device %s (distributed)" % device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device %s (not distributed)" % device)

    # initialize distributed process group using default env:// method
    if distributed:
        print("using nccl backend")
        torch.distributed.init_process_group(backend="nccl")
    
    # define train and dataset DataLoaders
    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # load training data
    train_set = torchvision.datasets.CIFAR10(
        root=args.path_to_data, train=True, download=False, transform=transform
    )

    if distributed:
        print("using distributed sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_ldr = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    sample = next(iter(train_ldr))[0]

    model = ConvNet(
        in_chan=sample.shape[1],
        n_classes=10
    ).to(device)

    # wrap model with DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum
    )

    # train the model
    for epoch in range(args.max_epochs):
        print("Rank %d: Starting epoch %d" % (rank, epoch + 1))
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        train(
            train_ldr,
            device,
            optimizer,
            model,
            criterion,
            rank,
            epoch
        )

    print("Rank %d: Finished Training" % (rank))

    # save model
    os.makedirs(args.model_dir, exist_ok=True)
    if distributed:
        save_on_master(model, args.model_dir + "/model.pt")
    else:
        model_scripted = torch.jit.script(model)
        model_scripted.save(args.model_dir + "/model.pt")


if __name__ == "__main__":
    main(
        parse_args()
    )