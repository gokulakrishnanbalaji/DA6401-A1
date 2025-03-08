import wandb

# Define the train function
def train(X_train_full, y_train_full, X_test, y_test, AdvancedFFNN):

    # Initialize wandb run
    run = wandb.init(
        entity="da24m007-iit-madras",
        project="DL-A1"
    )
    
    # Configuration will be injected by wandb sweep
    config = wandb.config  

    # Simulated Fashion MNIST data split
    val_size = int(0.1 * len(X_train_full))
    X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
    
    # Initialize network with sweep parameters
    nn = AdvancedFFNN(
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        weight_init=config.weight_init,
        activation=config.activation
    )
    
    # Set run name based on config
    run.name = f"hl_{len(config.hidden_layers)}_bs_{config.batch_size}_ac_{config.activation}"
    
    # Train the network
    nn.train(X_train, y_train, X_val, y_val, X_test, y_test, config.epochs)

sweep_config = {
    'method': 'bayes',  # Bayesian Optimization
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
    },
    'early_terminate': {
        'type': 'hyperband',  # Stops bad runs early
        'min_iter': 3
    }
}



