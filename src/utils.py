import wandb
from omegaconf import DictConfig

from datetime import datetime

import subprocess


def init_logger(cfg: DictConfig, login_id, experiment_notes=None):
    subprocess.run(["wandb", "--relogin", "login", str(login_id)])
    wandb.init(project="image-expansion-tool",
               name="Exp {}".format(str(int(datetime.utcnow().timestamp()))[2:]),
               notes=experiment_notes if experiment_notes is not None else str(cfg),
               tags=[cfg.model.version, cfg.dataset.version, cfg.learning.loss, cfg.learning.optimizer_type],
               config=cfg)


def finish_logger():
    wandb.finish()
