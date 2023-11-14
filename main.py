from omegaconf import DictConfig
from torch import nn
import hydra
import torch

from src.data import get_dataset
from src.data import download_and_process_data

from src.data import SeagullDataset

from src.models.train_model import train_model

from src.models.upscaler_v1 import Upscaler


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.model.version == "upscaler_v1":
        model_version = Upscaler()
    else:
        raise Exception("No correct value for model.version in config is declared")

    if cfg.dataset.version == "seagull_dataset":
        download_and_process_data()

        train_dataset = SeagullDataset("./data/processed/train/train/images/")
        val_dataset = SeagullDataset("./data/processed/train/valid/images/")
        test_dataset = SeagullDataset("./data/processed/test/images/")
    else:
        raise Exception("No correct value for dataset.version in config is declared")

    if cfg.learning.loss == "mse":
        loss = nn.MSELoss()
    else:
        raise Exception("No correct value for learning.loss in config is declared")

    if cfg.learning.optimizer_type == "Adam":
        optimizer_type = torch.optim.Adam
    else:
        raise Exception("No correct value for learning.optimizer_type in config is declared")

    accelerator = cfg.server.accelerator.type if cfg.server.get("accelerator", None) is not None else "cpu"
    devices = cfg.server.accelerator.devices if cfg.server.get("accelerator", None) is not None else 1

    train_loader, valid_loader, test_loader = get_dataset(train_dataset=train_dataset, val_dataset=val_dataset,
                                                          test_dataset=test_dataset, batch_size=cfg.learning.batch_size)

    train_model(model_version=model_version, train_loader=train_loader, loss=loss, lr=0.005,
                optimizer_type=optimizer_type, accelerator=accelerator,
                devices=devices)


if __name__ == "__main__":
    main()
