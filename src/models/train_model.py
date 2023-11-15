import pytorch_lightning as pl

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class UpscalerModule(pl.LightningModule):
    def __init__(self, model, loss, lr, optimizer_type):
        super().__init__()
        self.model = model
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.loss = loss
        self.lr = lr
        self.optimizer_type = optimizer_type

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
        optimizer = self.optimizer_type(self.parameters(), lr=self.lr)
        return optimizer


def train_model(model_version, train_loader, loss, optimizer_type, accelerator, devices, lr=0.005, max_epochs=1000):
    model = UpscalerModule(model_version, loss, lr, optimizer_type)

    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)
