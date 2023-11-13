import subprocess
from torch.utils.data import Dataset, DataLoader
import os

import torch

from PIL import Image
from PIL.Image import Resampling

import numpy as np


def download_data():
    subprocess.run(["chmod", "+x", "./src/data/download_data.sh"])
    subprocess.run(["./src/data/download_data.sh"])
    subprocess.run(["unzip", "./data/raw/dataset.zip", "-d", "./data/processed/"])


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


def get_dataset(batch_size=8):
    download_data()

    train_dataset = UpscalerDataset("./data/processed/train/train/images/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = UpscalerDataset("./data/processed/train/valid/images/")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = UpscalerDataset("./data/processed/test/images/")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return [train_loader, valid_loader, test_loader]
