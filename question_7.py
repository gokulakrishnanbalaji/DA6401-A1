import wandb
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from config import entity, project


# Forward pass function
def forward(X, W, b):
    h = X
    if isinstance(W, np.ndarray):
        W = W.item()
    if isinstance(b, np.ndarray):
        b = b.item()
    
    num_layers = len(W)
    for i in range(num_layers):
        W_i = W[f'W_{i}']
        b_i = b[f'b_{i}']
        h = np.dot(h, W_i) + b_i
        if i < num_layers - 1:
            h = np.maximum(0, h)
    return h

# Plot and log confusion matrix
def plot_confusion_matrix( X_test, y_test):
    class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    best_run = max(runs, key=lambda run: run.summary.get("val_accuracy", 0))
    
    artifact = api.artifact(f"{entity}/{project}/{best_run.name}:latest")
    artifact_dir = artifact.download()
    
    artifact_files = os.listdir(artifact_dir)
    model_file = next(file for file in artifact_files if file.endswith(".npz"))
    
    model_data = np.load(os.path.join(artifact_dir, model_file), allow_pickle=True)
    W = model_data["weights"]
    b = model_data["biases"]
    
    # Compute predictions
    logits = forward(X_test, W, b)
    y_pred = np.argmax(logits, axis=1)
    
    # Log confusion matrix to W&B
    run = wandb.init(entity=entity, project=project)
    run.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,  # Use None since we have predicted labels, not probabilities
            y_true=y_test,
            preds=y_pred,
            class_names=class_names
        )
    })
    run.finish()

