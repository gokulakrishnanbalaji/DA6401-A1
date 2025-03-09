import wandb
import numpy as np
import os
from config import project, entity

# Define the train function
def train(X_train, y_train, X_val, y_val, X_test, y_test, AdvancedFFNN):
    # Configuration will be injected by wandb sweep
    run = wandb.init(project=project, entity=entity)  
    config = wandb.config  
    
    # Initialize network with sweep parameters
    nn = AdvancedFFNN(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        weight_init=config.weight_init,
        activation=config.activation
    )
    
    # Set run name based on config
    run.name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}"
    
    # Train the network
    nn.train(X_train, y_train, X_val, y_val, X_test, y_test, config.epochs)
    W,b = nn.return_weights_and_bias()

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{run.name}.npz"
    np.savez(model_path,
            weights=W,
            biases=b)
            
    # Create and log the Artifact
    artifact = wandb.Artifact(name=f"{run.name}", type="model", description=f"model for run {run.name}")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


sweep_config = {
    'method': 'bayes',  # Bayesian Optimization
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [32, 64, 128]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    },
    'early_terminate': {
        'type': 'hyperband',  # Stops bad runs early
        'min_iter': 3
    }
}



