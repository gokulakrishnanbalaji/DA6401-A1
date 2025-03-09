import wandb
from question_3 import AdvancedFFNN
from question_1 import X_test, y_test, X_train, y_train
from config import entity, project

def compare_loss():
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    best_run = max(runs, key=lambda run: run.summary.get("val_accuracy", 0))

    # Sort by best validation accuracy (or lowest loss)
    # Get the best run's hyperparameters
    best_hyperparams = best_run.config

    # Print results
    print("Best Run URL:", best_run.url)  

    print("Best Run ID:", best_run.id)
    print("Best Validation Accuracy:", best_run.summary.get("val_accuracy", "N/A"))
    print("Best Hyperparameters:", best_hyperparams)


    model = AdvancedFFNN(
        hidden_layers=best_hyperparams.get('hidden_layers'),
        learning_rate=best_hyperparams.get('learning_rate'),
        optimizer=best_hyperparams.get('optimizer'),
        weight_decay=best_hyperparams.get('weight_decay'),
        batch_size=best_hyperparams.get('batch_size'),
        weight_init=best_hyperparams.get('weight_init'),
        activation=best_hyperparams.get('activation'),mse=True

    )

    model.train(X_train, y_train, X_train, y_train,X_test, y_test, 20)

    mse_loss , cross_loss = model.mse_loss , model.cross_loss

    return mse_loss, cross_loss

if __name__ == "__main__":
    mse_loss, cross_loss = compare_loss()  
    epochs = list(range(len(mse_loss)))  # Generate x-axis (epochs)

    wandb.init(project=project, entity=entity)

    # Create a line plot with both loss curves
    wandb.log({
        "Loss Curves": wandb.plot.line_series(
            xs=epochs,
            ys=[mse_loss, cross_loss],
            keys=["MSE Loss", "Cross Entropy Loss"],
            title="Loss Curves",
            xname="Epochs"
        )
    })

    wandb.finish()

