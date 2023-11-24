import pytorch_lightning as pl
import torch

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from datetime import datetime

import wandb

import numpy as np

common_dict_counter = {}


def save_model(epoch, model, optimizer):
    common_dict_counter[epoch] = common_dict_counter.get(epoch, -1) + 1

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    },
        "./models/model_{}_v{}_{}.ckpt"
        .format(
            epoch,
            common_dict_counter[epoch],
            datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        )
    )


class UpscalerModule(pl.LightningModule):
    def __init__(self, model, loss, lr=0.005):
        super().__init__()
        self.model = model
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.loss = loss
        self.lr = lr

        self.loss_values_train = []
        self.metrics_values_train = []
        self.metrics_values_valid = []

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)

        loss_value = self.loss(y / 255, y_)
        metric_value = self.lpips(y_, y / 255)

        self.loss_values_train.append(loss_value.item())
        self.metrics_values_train.append(metric_value.item())

        return loss_value

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)

        metric_value = self.lpips(y / 255, y_)
        self.metrics_values_valid.append(metric_value.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_end(self):
        save_model(self.current_epoch, self.model, None)
        wandb.log({
            "loss": np.array(self.loss_values_train).mean(),
            "metrics": {
                "train": np.array(self.metrics_values_train).mean(),
                "valid": np.array(self.metrics_values_valid).mean(),
            }
        })

        self.loss_values_train = []
        self.metrics_values_train = []
        self.metrics_values_valid = []


def train_model(model_version, train_loader, valid_loader, loss, optimizer_type, accelerator, devices, lr=0.005, max_epochs=1000,
                log_every_n_steps=5):
    model = UpscalerModule(model_version, loss, lr)

    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs,
                         log_every_n_steps=log_every_n_steps, default_root_dir="./models/")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
