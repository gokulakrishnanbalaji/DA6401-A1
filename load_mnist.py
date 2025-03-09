from keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

val_size = int(0.1 * len(X_train_full))
X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]