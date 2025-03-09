import wandb
from train import config

entity=config.wandb_entity
project=config.wandb_project

run = wandb.init(project=project, entity=entity)
