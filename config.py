import wandb

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="da24m007-iit-madras",
    # Set the wandb project where this run will be logged.
    project="DL-A1",
    # Track hyperparameters and run metadata.
    # config={
    #     "learning_rate": 0.02,
    #     "architecture": "CNN",
    #     "dataset": "CIFAR-100",
    #     "epochs": 10,
    # },
)