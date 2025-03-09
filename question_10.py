from load_mnist import X_train, y_train, X_test, y_test, X_val, y_val
from question_3 import AdvancedFFNN
import wandb
from config import entity, project




# (relu, momentum ) ,(tanh, nadam), (tanh,momentum)
def compare_models(epochs,batch_size):
    wandb.init(project=project, entity=entity)
    model_1 = AdvancedFFNN(num_layers=5,hidden_size=128, learning_rate=1e-3, optimizer='momentum', weight_decay=0, batch_size=batch_size, weight_init='xavier', activation='relu')
    model_2 = AdvancedFFNN(num_layers=5,hidden_size=128, learning_rate=1e-3, optimizer='nadam', weight_decay=0, batch_size=batch_size, weight_init='xavier', activation='tanh')
    model_3 = AdvancedFFNN(num_layers=5,hidden_size=128, learning_rate=1e-3, optimizer='momentum', weight_decay=0, batch_size=batch_size, weight_init='xavier', activation='tanh')

    # running each of the model for 10 epochs
    model_1.train(X_train, y_train,X_val,y_val,X_test, y_test, epochs)
    model_2.train(X_train, y_train,X_val,y_val, X_test, y_test, epochs) 
    model_3.train(X_train, y_train, X_val,y_val,X_test, y_test, epochs)

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

if __name__ == "__main__":
    compare_models(epochs=20,batch_size=64) 