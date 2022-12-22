import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import List


def create_layers(targets: List[int]) -> List[int]:
    layers = [
        nn.Linear(targets[0], targets[1]),
        nn.ReLU()
    ]
    for i in range(2, len(targets) - 1):
        dim = targets[i]
        layers.extend([
            nn.Linear(targets[i-1], dim),
            nn.ReLU()
        ])
    layers.append(
        nn.Linear(targets[-2], targets[-1])
    )
    return layers


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, in_layers: List[int], device: str):
        super(LitAutoEncoder, self).__init__()
        assert len(in_layers) >= 3, "minimum number of 3 layers" 
        self.enc = nn.Sequential(
            *create_layers(in_layers)
        )
        self.dec = nn.Sequential(
            *create_layers(list(reversed(in_layers)))
        )
        self.criterion = nn.MSELoss().to(device)

    def forward(self, X: torch.Tensor):
        emb = self.enc(X)
        return emb
    
    def score_sample(self, X: torch.Tensor):
        emb = self.enc(X)
        X_hat = self.dec(emb)
        score = self.criterion(X, X_hat)
        return score

    def training_step(self, batch, batch_idx):
        X, _ = batch
        X = X.view(X.size(0), -1)
        loss = self.score_sample(X)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, _ = batch
        X = X.view(X.size(0), -1)
        loss = self.score_sample(X)
        self.log("val_loss", loss)

        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
