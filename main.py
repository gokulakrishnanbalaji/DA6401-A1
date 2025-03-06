from question_2 import FFNN
from question_1 import X_test, y_test

nn= FFNN()
x = X_test[:5].reshape(-1, 784)
print(x.shape)
print(y_test[:5].shape)
predictions = nn.forward(x)
print(predictions)
print('-'*50)
print(y_test[:5])