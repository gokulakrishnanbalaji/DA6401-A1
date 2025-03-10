from load_mnist import X_train, y_train, X_test, y_test, X_val, y_val
from question_3 import AdvancedFFNN
import wandb
from config import entity, project





def compare_models():
    wandb.init(project=project, entity=entity)
    model_1 = AdvancedFFNN(num_layers=4,hidden_size=128, learning_rate=1e-3, optimizer='adam', weight_decay=0.0005, batch_size=64, weight_init='xavier', activation='sigmoid')
    model_2 = AdvancedFFNN(num_layers=4,hidden_size=128, learning_rate=1e-3, optimizer='momentum', weight_decay=0.0005, batch_size=64, weight_init='xavier', activation='ReLU')
    model_3 = AdvancedFFNN(num_layers=4,hidden_size=128, learning_rate=1e-3, optimizer='rmsprop', weight_decay=0.0005, batch_size=64, weight_init='xavier', activation='ReLU')

    # running each of the model for 10 epochs
    model_1.train(X_train, y_train,X_val,y_val,X_test, y_test, 20)
    model_2.train(X_train, y_train,X_val,y_val, X_test, y_test, 20) 
    model_3.train(X_train, y_train, X_val,y_val,X_test, y_test, 20)

    accuracy=[]
    accuracy.append(model_1.test_accuracy[-1])
    accuracy.append(model_3.test_accuracy[-1])
    accuracy.append(model_2.test_accuracy[-1])

    table = wandb.Table(columns=["Model", "Test Accuracy"])

    # Append accuracy values for each model
    table.add_data("adam + sigmoid", accuracy[0])
    table.add_data("momentum + ReLU", accuracy[1])
    table.add_data("rmsprop + ReLU", accuracy[2])

    # Log the table
    wandb.log({"Test Accuracy Comparison": table})

    wandb.finish()

if __name__ == "__main__":
    compare_models() 