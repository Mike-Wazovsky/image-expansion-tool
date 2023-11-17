from omegaconf import DictConfig
from torch import nn
import hydra
import torch

from src.data import get_dataloaders
from src.data import download_and_process_data

from src.data import SeagullDataset

from src.models.train_model import train_model

from src.models.upscaler_v1 import UpscalerV1
from src.models.upscaler_v2 import UpscalerV2


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.model.version == "upscaler_v1":
        model_version = UpscalerV1()
    elif cfg.model.version == "upscaler_v2":
        model_version = UpscalerV2()
    else:
        raise Exception("No correct value for model.version in config is declared")

    if cfg.dataset.version == "seagull_dataset":
        download_and_process_data("./src/data/download_seagull_data.sh")

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
    elif cfg.learning.optimizer_type == "SGD":
        optimizer_type = torch.optim.SGD
    else:
        raise Exception("No correct value for learning.optimizer_type in config is declared")

    accelerator = cfg.server.accelerator.type if cfg.server.get("accelerator", None) is not None else "cpu"
    devices = cfg.server.accelerator.devices if cfg.server.get("accelerator", None) is not None else 1

    train_loader, valid_loader, test_loader = get_dataloaders(train_dataset=train_dataset, val_dataset=val_dataset,
                                                              test_dataset=test_dataset,
                                                              batch_size=cfg.learning.batch_size)

    train_model(model_version=model_version, train_loader=train_loader, loss=loss, lr=cfg.learning.lr,
                optimizer_type=optimizer_type, accelerator=accelerator,
                devices=devices, max_epochs=cfg.learning.epoch_amount, log_every_n_steps=cfg.learning.log_every_n_steps)


if __name__ == "__main__":
    main()
