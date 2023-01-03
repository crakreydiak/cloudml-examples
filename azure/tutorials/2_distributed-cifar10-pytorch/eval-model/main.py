#
# Script adapted from:
# https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/pipelines/2b_train_cifar_10_with_pytorch/eval-model/main.py
# ==============================================================================

# imports
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--path_to_model", type=str, default="./models/model.pt")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # parse args
    return parser.parse_args()


# define functions
def evaluate(test_ldr: DataLoader, model: nn.Module, device: str):
    n_classes = 10
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    model.eval()

    correct, total = 0, 0
    class_correct = list(0.0 for _ in range(10))
    class_total = list(0.0 for _ in range(10))
    with torch.no_grad():
        for X, y in test_ldr:
            X, y = X.float().to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            c = (predicted == y).squeeze()
            for i in range(n_classes):
                label = y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # print total test set accuracy
    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )

    # print test accuracy for each of the classes
    for i in range(n_classes):
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i])
        )


def main(args):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define test dataset DataLoaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_set = torchvision.datasets.CIFAR10(
        root=args.path_to_data, train=False, download=False, transform=transform
    )
    test_ldr = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # load model
    model = torch.jit.load(args.path_to_model)
    model = model.to(device)

    evaluate(test_ldr, model, device)


# run script
if __name__ == "__main__":
    # call main function
    main(
        parse_args()
    )
