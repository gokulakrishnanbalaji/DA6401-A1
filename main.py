from question_3 import AdvancedFFNN
from question_1 import X_test, y_test, sample_images,sample_labels,X_train as X_train_full, y_train as y_train_full
import wandb
from question_4 import train, sweep_config
from config import run


# Reporting one image per class in wandb
run.log({
    "Fashion-MNIST Samples": [
        wandb.Image(img, caption=lbl) for img, lbl in zip(sample_images, sample_labels)
    ]
})

run.finish()


# Create the sweep
sweep_id = wandb.sweep(sweep_config, entity="da24m007-iit-madras", project="DL-A1")

# Define a wrapper function to pass the arguments to train
def run_sweep():
    train(X_train_full, y_train_full, X_test, y_test, AdvancedFFNN)

# Run the sweep
wandb.agent(sweep_id, function=run_sweep, count=50)