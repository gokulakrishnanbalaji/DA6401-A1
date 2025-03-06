import numpy as np


# Defining class for feed forward neural network
class FFNN:
    def __init__(self, input_size = 784, hidden_layers=[128,64], output_size=10, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers=[]
        layer_size = [input_size]+hidden_layers+[output_size]

        for i in range(len(layer_size) - 1):
            weight=np.random.randn(layer_size[i] , layer_size[i+1]) * 0.01
            bias = np.zeros((1,layer_size[i+1]))
            self.layers.append(
                {
                    'W':weight,
                    'b':bias
                }
            )

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self,x):
        for layer in self.layers:
            x = np.dot(x, layer['W']) + layer['b']
            x = self.sigmoid(x)
        
        return self.softmax(x)

 
