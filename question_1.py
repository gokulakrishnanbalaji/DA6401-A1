# importing keras to download dataset
from keras.datasets import fashion_mnist
import wandb

# importing numpy
import numpy as np

# importing run object of wandb class to commit
from config import run

# Getting training and testing data from fashion_mnist dataset
(x_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# storing the class names in a list
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Select one sample per class
sample_images = []
sample_labels = []

# Get first index of each class
for class_id in range(10):
    idx = np.where(y_train == class_id)[0][0]  
    sample_images.append(x_train[idx])
    sample_labels.append(class_names[class_id])


# Reporting one image per class in wandb
run.log({
    "Fashion-MNIST Samples": [
        wandb.Image(img, caption=lbl) for img, lbl in zip(sample_images, sample_labels)
    ]
})

run.finish()