from keras.datasets import mnist
from question_3 import AdvancedFFNN
import wandb
from config import entity, project

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

val_size = int(0.1 * len(X_train_full))
X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]

# my hyperparameters will be 
# activation function: relu, tanh
# batch_size: 32, 64
# hl: 5
# weight:decayed: 0
# weight_init: xavier

# (relu, momentum ) ,(tanh, nadam), (tanh,momentum)

model_1 = AdvancedFFNN(hidden_layers=[128, 64, 32, 16, 8], learning_rate=1e-3, optimizer='momentum', weight_decay=0, batch_size=64, weight_init='xavier', activation='relu')
model_2 = AdvancedFFNN(hidden_layers=[128, 64, 32, 16, 8], learning_rate=1e-3, optimizer='nadam', weight_decay=0, batch_size=64, weight_init='xavier', activation='tanh')
model_3 = AdvancedFFNN(hidden_layers=[128, 64, 32, 16, 8], learning_rate=1e-3, optimizer='momentum', weight_decay=0, batch_size=64, weight_init='xavier', activation='tanh')

# running each of the model for 10 epochs
model_1.train(X_train, y_train,X_val,y_val,X_test, y_test, 10)
model_2.train(X_train, y_train,X_val,y_val, X_test, y_test, 10) 
model_3.train(X_train, y_train, X_val,y_val,X_test, y_test, 10)

accuracy=[]
accuracy.append(model_1.test_accuracy[-1])
accuracy.append(model_3.test_accuracy[-1])
accuracy.append(model_2.test_accuracy[-1])

table = wandb.Table(columns=["Model", "Test Accuracy"])

# Append accuracy values for each model
table.add_data("ReLU + Momentum", accuracy[0])
table.add_data("Tanh + Nadam", accuracy[1])
table.add_data("Tanh + Momentum", accuracy[2])

# Log the table
wandb.log({"Test Accuracy Comparison": table})

wandb.finish()