import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from typing import Union, Dict, List, Tuple

class LitRNN(pl.LightningModule):
    def __init__(self,
                    model,
                    loss,
                    optimizer_klass,
                    optimizer_args: Dict) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_klass = optimizer_klass
        self.optimizer_args = optimizer_args

        return

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer_klass(self.model.parameters(), **self.optimizer_args)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.model(x)
        loss = self.loss(out, y)

        return loss

    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch
        out = self.model(x)
        loss = self.loss(out, y)

        if batch_idx % 10 == 0:
            print(f'{batch_idx}) Validation loss: {loss}')
        return

    def backward(self, loss, optimizer, optimizer_idx) -> None:
        loss.backward()
        return