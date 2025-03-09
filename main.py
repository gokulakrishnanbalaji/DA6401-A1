from question_3 import AdvancedFFNN
from question_1 import X_train, y_train, X_test, y_test, X_val, y_val, sample_images, sample_labels
import wandb
from question_4 import train, sweep_config
from config import entity, project
from train import config



def main():

    if config.dataset == "fashion_mnist":
        run = wandb.init(project=project, entity=entity)
        run.log({
            "Fashion-MNIST Samples": [
                wandb.Image(img, caption=lbl) for img, lbl in zip(sample_images, sample_labels)
            ]
        })

        run.finish()

        # Create the sweep
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)

        # Define a wrapper function to pass the arguments to train
        def run_sweep():
            train(X_train, y_train, X_val,y_val, X_test, y_test, AdvancedFFNN)

        # Run the sweep
        wandb.agent(sweep_id, function=run_sweep, count=5)

        wandb.finish()

    else:
        from question_10 import compare_models
        compare_models()



if __name__ == "__main__":
    main()