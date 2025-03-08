from question_3 import AdvancedFFNN
from question_1 import X_test, y_test
from question_1 import X_train as X_train_full, y_train as y_train_full
import wandb
from question_4 import train, sweep_config



# Create the sweep
sweep_id = wandb.sweep(sweep_config, entity="da24m007-iit-madras", project="DL-A1")

# Define a wrapper function to pass the arguments to train
def run_sweep():
    train(X_train_full, y_train_full, X_test, y_test, AdvancedFFNN)

# Run the sweep
wandb.agent(sweep_id, function=run_sweep)