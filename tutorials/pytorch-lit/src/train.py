import torch
import pytorch_lightning as pl
import argparse
import os
from models import LitAutoEncoder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", default="./data", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


def main(
   args
):
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # logger
    logger = TensorBoardLogger(save_dir=args.log_dir)

    # data
    dataset = MNIST(
        os.path.join(args.path_to_data, "train"),
        train=True, download=False, transform=transforms.ToTensor()
    )
    mnist_test = MNIST(
        os.path.join(args.path_to_data, "test"),
        train=False, download=False, transform=transforms.ToTensor()
    )
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # loaders
    train_ldr = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_ldr = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_ldr = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # get img size
    data_sample = next(iter(train_ldr))
    in_features = data_sample[0].shape[-1]

    # model
    model = LitAutoEncoder(
        in_layers=[in_features * in_features, 64, 3],
        device=device
    )

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator=device,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, train_ldr, val_ldr)

    # test
    res = trainer.test(test_ldr)
    print(res)

    # save model
    torch.onnx.export(
        model,
        data_sample,
        "model_torch_export.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        })



if __name__ == "__main__":
    main(
        parse_args()
    )