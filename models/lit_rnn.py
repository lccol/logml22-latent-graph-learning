import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

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

        self.metrics = [
            Accuracy(num_classes=self.model.nclasses, average='micro', multiclass=True),
            Precision(num_classes=self.model.nclasses, average='micro', multiclass=True),
            Recall(num_classes=self.model.nclasses, average='micro', multiclass=True),
            F1Score(num_classes=self.model.nclasses, average='micro', multiclass=True)
        ]

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

        for m in self.metrics:
            m.update(out, y)

        return loss

    def training_epoch_end(self, outputs) -> None:
        for m in self.metrics:
            val = m.compute()
            print(f'{type(m).__name__}: {val}')
        return

    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch
        out = self.model(x)
        loss = self.loss(out, y)

        if batch_idx % 10 == 0:
            print(f'{batch_idx}) Validation loss: {loss}')

        for m in self.metrics:
            m.update(out, y)
        return

    def validation_epoch_end(self, outputs) -> None:
        for m in self.metrics:
            val = m.compute()
            print(f'{type(m).__name__}: {val}')
        return

    def backward(self, loss, optimizer, optimizer_idx) -> None:
        loss.backward()
        return