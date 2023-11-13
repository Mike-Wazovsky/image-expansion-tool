import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .upscaler_v1 import Upscaler

class UpscalerModule(pl.LightningModule):
    def __init__(self, model, loss, lr=0.005):
        super().__init__()
        self.model = model
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.loss = loss
        self.lr = lr

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)

        loss_value = self.loss(y / 255, y_)
        metric_value = self.lpips(y_, y / 255)

        self.log("train_loss", loss_value, on_epoch=True)
        self.log("train_metric", metric_value, on_epoch=True)

        return loss_value

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)

        metric_value = self.lpips(y / 255, y_)
        self.log("valid_metric", metric_value, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_model(train_loader):
    model_version = Upscaler()
    model = UpscalerModule(model_version, nn.MSELoss())

    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
