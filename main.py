from question_3 import AdvancedFFNN
from question_1 import X_test, y_test
from question_1 import X_train as X_train_full, y_train as y_train_full
from config import run
import wandb


def train():
    run = wandb.init(
    entity="da24m007-iit-madras",
    project="DL-A1")
    
    config = wandb.config
    
    # Simulated Fashion MNIST data (replace with actual loading)

    val_size = int(0.1 * len(X_train_full))
    X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
    
    # Initialize network
    nn = AdvancedFFNN(
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        weight_init=config.weight_init,
        activation=config.activation
    )
    
    # Set run name
    run.name = f"hl_{len(config.hidden_layers)}_bs_{config.batch_size}_ac_{config.activation}"
    
    # Train
    nn.train(X_train, y_train, X_val, y_val, X_test, y_test, config.epochs)

if __name__ == "__main__":
    sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {
            'values': [[128, 64, 32], [128, 64, 32, 16], [128, 64, 32, 16, 8]]
        },
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}
    # Create sweep (input sweep_config here)
    sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_sweep")
    # Run agent to execute the sweep
    wandb.agent(sweep_id, function=train)