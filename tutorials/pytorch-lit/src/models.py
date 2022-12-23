import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchmetrics import Accuracy


def create_layers(targets: List[int]) -> List[int]:
    in_layer = targets[0]
    layers = []
    for i in range(1, len(targets) - 1):
        layers.extend([
            nn.Linear(in_layer, targets[i]),
            nn.ReLU()
        ])
        in_layer = targets[i]
    layers.append(
        nn.Linear(in_layer, targets[-1])
    )
    return layers


class LitMNISTClassifier(pl.LightningModule):

    def __init__(self, in_layers: List[int]):
        super(LitMNISTClassifier, self).__init__()
        assert len(in_layers) >= 3, "minimum number of 3 layers" 
        self.net = nn.Sequential(
            *create_layers(in_layers)
        )
        self.criterion = nn.NLLLoss()
        self.accuracy = Accuracy("multiclass", num_classes=in_layers[-1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emb = self.net(X)
        logits = F.log_softmax(emb, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.view(X.size(0), -1)
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        X, y = batch
        X = X.view(X.size(0), -1)
        logits = self.forward(X)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return {
            "val_loss": loss,
            "val_accuracy": acc
        }
    
    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx) -> dict:
        X, y = batch
        X = X.view(X.size(0), -1)
        logits = self.forward(X)
        loss = self.criterion(X, y)
        acc = self.accuracy(logits, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return {
            "test_loss": loss,
            "test_accuracy": acc
        }

    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.log("test_loss", avg_loss)
        self.log("test_accuracy", avg_acc)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
