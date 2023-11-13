import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np

import os
from PIL import Image
from PIL.Image import Resampling

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tqdm
from time import sleep

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pytorch_lightning.loggers import CSVLogger

class UpscalerDataset(Dataset):
    def __init__(self, path):
        self.filenames = []
        self.root = path
        for (_, _, files) in os.walk(path):
            for file in files:
                # TODO: Добавить аугментаций
                self.filenames.append(file)
            break

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(self.root + filename)

        image_width = image.size[0]
        image_height = image.size[1]
        image_preprocess = lambda lenght: lenght if lenght % 2 == 0 else lenght - 1

        image_preprocessed = image.crop((0, 0, image_preprocess(image_width), image_preprocess(image_height)))

        image_rescaled = image_preprocessed.copy()
        image_rescaled.thumbnail((image_rescaled.size[0] // 2, image_rescaled.size[1] // 2),
                                 resample=Resampling.BICUBIC)

        X = torch.tensor(np.array(image_rescaled, dtype=np.float32)).transpose(1, 2).transpose(0, 1)
        Y = torch.tensor(np.array(image_preprocessed, dtype=np.float32)).transpose(1, 2).transpose(0, 1)

        return X, Y

batch_size = 8

train_dataset = UpscalerDataset("./train/train/images/")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = UpscalerDataset("./train/valid/images/")
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

test_dataset = UpscalerDataset("./test/images/")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Upscaler(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.BatchNorm2d(3)
        self.l2 = nn.Conv2d(3, 24, (3, 3), padding=1)
        self.l3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.l4 = nn.Sigmoid()
        self.l5 = nn.BatchNorm2d(24)
        self.l6 = nn.Upsample(scale_factor=2)
        self.l7 = nn.Conv2d(24, 3, (3, 3), padding=1)
        self.l8 = nn.Sigmoid()
        self.l9 = nn.BatchNorm2d(3)
        self.l10 = nn.Upsample(scale_factor=2)
        self.l11 = nn.Sigmoid()

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        y = self.l5(y)
        y = self.l6(y)
        y = self.l7(y)
        y = self.l8(y)
        y = self.l9(y)
        y = self.l10(y)

        return self.l11(y)


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
        y_ = model(x)

        metric_value = self.lpips(y / 255, y_)
        self.log("valid_metric", metric_value, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    model = UpscalerModule(Upscaler(), nn.MSELoss())

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, logger=CSVLogger(save_dir="logs/"))
