# importing keras to download dataset
from keras.datasets import fashion_mnist
# importing numpy
import numpy as np

# Getting training and testing data from fashion_mnist dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

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


