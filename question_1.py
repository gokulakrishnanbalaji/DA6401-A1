# importing keras to download dataset
from keras.datasets import fashion_mnist
# importing numpy
import numpy as np

# Getting training and testing data from fashion_mnist dataset

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

val_size = int(0.1 * len(X_train_full))
X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
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
    sample_images.append(X_train[idx])
    sample_labels.append(class_names[class_id])


