# importing numpy
import numpy as np

# importing from other modules
from question_2 import FFNN
from config import run

#importing wandb
import wandb

# defining AdvancedFFNN class (inherits FFNN) that supports backpropagation, with different optimisers
class AdvancedFFNN(FFNN):
    # constructor, that takes in required hyperparameters
    def __init__(self, input_size=784, hidden_layers=[128, 64], output_size=10, 
                 learning_rate=0.01, optimizer='sgd', weight_decay=0, 
                 activation='sigmoid', batch_size=32, weight_init='random'):
        
        # Calling constructor of FFNN class
        super().__init__(input_size, hidden_layers, output_size, learning_rate)

        # storing the hyperparameters in class variables
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.activation = activation
        self.weight_init = weight_init

        # for returing the final set of weights and biases
        self.weights = {}
        self.biases = {}

        # initialise a list of dictionaries to track the velocity or momentum of weights and biases
        self.velocity = [{} for _ in self.layers]

        # initialise a list of dictionaries to track the first and second moments of weights and biases
        self.m = [{} for _ in self.layers]
        self.v = [{} for _ in self.layers]

        # variable to keep track of the number of iterations
        self.t = 0

        # epsilon value for numerical stability
        self.epsilon = 1e-8

        # hyperparameters for optimisers
        self.beta1, self.beta2 = 0.9, 0.999
        self.rho = 0.9
        
        # initialise weights using xavier initialisation
        if weight_init == 'xavier':
            self.initialize_xavier()

    # function to initialise weights using xavier initialisation
    def initialize_xavier(self):

        # iterate over all layers
        for i, layer in enumerate(self.layers):

            # calculate the limit for xavier initialisation
            fan_in = layer['W'].shape[0]
            fan_out = layer['W'].shape[1]
            limit = np.sqrt(6 / (fan_in + fan_out))

            # initialise weights using uniform distribution
            layer['W'] = np.random.uniform(-limit, limit, layer['W'].shape)

    # function that calculates the output of activation function
    def activation_func(self, x):

        # if sigmoid activation function is used
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        
        # if tanh activation function is used
        elif self.activation == 'tanh':
            return np.tanh(x)
        
        # if relu activation function is used
        elif self.activation == 'relu':
            return np.maximum(0, x)
        
        # if none of the above, use sigmoid activation function
        return self.sigmoid(x)

    # function that calculates the derivative of activation function
    def activation_deriv(self, x):

        # if sigmoid activation function is used, return derivative of sigmoid function
        if self.activation == 'sigmoid':
            s = self.sigmoid(x)
            return s * (1 - s)
        
        # if tanh activation function is used, return derivative of tanh function
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        
        # if relu activation function is used, return derivative of relu function
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        
        # if none of the above, return derivative of sigmoid function
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # forwrd pass of the network
    def forward(self, x):

        # if input is 3D, reshape it to 2D
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # store the input in activations list
        self.activations = [x]

        # store the z values in z_values list
        self.z_values = []
        
        # iterte for all layers except the last one
        for layer in self.layers[:-1]:

            # calculate z value ( z = Wx + b )
            z = np.dot(x, layer['W']) + layer['b']
            self.z_values.append(z)

            # calculate the output of activation function
            x = self.activation_func(z)

            # store the output in activations list
            self.activations.append(x)
        
        # calculate z value for last layer
        z = np.dot(x, self.layers[-1]['W']) + self.layers[-1]['b']
        self.z_values.append(z)

        # calculate the output of softmax function
        output = self.softmax(z)

        # store the output in activations list
        self.activations.append(output)

        return output

    # backward pass of the network
    def backward(self, x, y, output):

        # if input is 3D, reshape it to 2D
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # number of samples
        m = x.shape[0]

        # list of dictionaries to store gradients
        grads = [{'W': 0, 'b': 0} for _ in self.layers]
        
        # y must be one-hot encoded here
        delta = output - y  

        # calculate gradients for last layer
        grads[-1]['W'] = np.dot(self.activations[-2].T, delta) / m
        grads[-1]['b'] = np.sum(delta, axis=0, keepdims=True) / m
        
        # iterate for all layers except the last one
        for i in range(len(self.layers)-2, -1, -1):

            # calculate delta value (delta = W.T * delta) * derivative of activation function
            delta = np.dot(delta, self.layers[i+1]['W'].T) * self.activation_deriv(self.z_values[i])

            # calculate gradients (dW = x.T * delta, db = sum(delta))
            grads[i]['W'] = np.dot(self.activations[i].T, delta) / m
            grads[i]['b'] = np.sum(delta, axis=0, keepdims=True) / m

        # add weight decay to gradients    
        for i in range(len(self.layers)):
            grads[i]['W'] += self.weight_decay * self.layers[i]['W']
            
        return grads

    # function to update parameters using gradients
    def update_parameters(self, grads):

        # increment the iteration count
        self.t += 1

        # iterate for all layers
        for i, layer in enumerate(self.layers):

            # if sgd optimiser is used
            if self.optimizer == 'sgd':
                
                # simply subtract the learning rate * gradient from weights and biases
                layer['W'] -= self.learning_rate * grads[i]['W']
                layer['b'] -= self.learning_rate * grads[i]['b']

            # if momentum optimiser is used
            elif self.optimizer == 'momentum':

                # if velocity is not already stored, initialise it
                if i not in self.velocity:
                    self.velocity[i] = {'W': 0, 'b': 0}
                
                # calculate the new velocity (v = 0.9 * v - learning_rate * gradient)
                self.velocity[i]['W'] = 0.9 * self.velocity[i]['W'] - self.learning_rate * grads[i]['W']
                self.velocity[i]['b'] = 0.9 * self.velocity[i]['b'] - self.learning_rate * grads[i]['b']
                
                # update weghts and biases (W = W + v, b = b + v)
                layer['W'] += self.velocity[i]['W']
                layer['b'] += self.velocity[i]['b']

            # if nesterov optimiser is used
            elif self.optimizer == 'nesterov':

                # if velocity is not already stored, initialise it
                if i not in self.velocity:
                    self.velocity[i] = {'W': 0, 'b': 0}

                # calculate the new velocity (v = 0.9 * v - learning_rate * gradient)
                temp_W = layer['W'] + 0.9 * self.velocity[i]['W']
                temp_b = layer['b'] + 0.9 * self.velocity[i]['b']

                self.velocity[i]['W'] = 0.9 * self.velocity[i]['W'] - self.learning_rate * grads[i]['W']
                self.velocity[i]['b'] = 0.9 * self.velocity[i]['b'] - self.learning_rate * grads[i]['b']
                
                # update weghts and biases (W = W + v, b = b + v)
                layer['W'] = temp_W + self.velocity[i]['W']
                layer['b'] = temp_b + self.velocity[i]['b']

            # if rmsprop optimiser is used
            elif self.optimizer == 'rmsprop':

                # if velocity is not already stored, initialise it
                if i not in self.v:
                    self.v[i] = {'W': 0, 'b': 0}

                # calculate the new velocity (v = rho * v + (1-rho)*learning_rate * gradient)
                self.v[i]['W'] = self.rho * self.v[i]['W'] + (1-self.rho) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.rho * self.v[i]['b'] + (1-self.rho) * np.square(grads[i]['b'])
                
                # updating weights and biases 
                layer['W'] -= self.learning_rate * grads[i]['W'] / (np.sqrt(self.v[i]['W']) + self.epsilon)
                layer['b'] -= self.learning_rate * grads[i]['b'] / (np.sqrt(self.v[i]['b']) + self.epsilon)
            
            # if adam optimiser is used
            elif self.optimizer == 'adam':

                # if m and v are not already stored, initialise them
                if i not in self.m:
                    self.m[i] = {'W': 0, 'b': 0}
                    self.v[i] = {'W': 0, 'b': 0}

                # calculating m and v values (m = beta * m + (1-beta) * gradient, v = 0.(beta) * v + (1-beta) * gradient^2)
                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1-self.beta1) * grads[i]['W']
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1-self.beta1) * grads[i]['b']
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1-self.beta2) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1-self.beta2) * np.square(grads[i]['b'])

                # calculating m_hat and v_hat values (m_hat = m / (1-beta^t), v_hat = v / (1-beta^t))
                m_hat_w = self.m[i]['W'] / (1 - self.beta1**self.t)
                m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
                v_hat_w = self.v[i]['W'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)

                # updating weights and biases
                layer['W'] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer['b'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            
            # if nadam optimiser is used
            elif self.optimizer == 'nadam':

                # if m and v are not already stored, initialise them
                if i not in self.m:
                    self.m[i] = {'W': 0, 'b': 0}
                    self.v[i] = {'W': 0, 'b': 0}

                # callculating m values (m = beta * m + (1-beta) * gradient)
                m_w = self.beta1 * self.m[i]['W'] + (1-self.beta1) * grads[i]['W']
                m_b = self.beta1 * self.m[i]['b'] + (1-self.beta1) * grads[i]['b']

                # calculating v values (v = 0.(beta) * v + (1-beta) * gradient^2)
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1-self.beta2) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1-self.beta2) * np.square(grads[i]['b'])
                
                # calculating m_hat values (m_hat = m / (1-beta^t))
                m_hat_w = m_w / (1 - self.beta1**self.t)
                m_hat_b = m_b / (1 - self.beta1**self.t)

                # calculating v_hat values (v_hat = v / (1-beta^t))
                v_hat_w = self.v[i]['W'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)

                # updating weights and biases
                # (W = W - learning_rate / (sqrt(v_hat) + epsilon) * (beta1 * m_hat + (1-beta1) * gradient/(1-beta1^t)))
                layer['W'] -= self.learning_rate / (np.sqrt(v_hat_w) + self.epsilon) * \
                             (self.beta1 * m_hat_w + (1-self.beta1) * grads[i]['W']/(1-self.beta1**self.t))
                layer['b'] -= self.learning_rate / (np.sqrt(v_hat_b) + self.epsilon) * \
                             (self.beta1 * m_hat_b + (1-self.beta1) * grads[i]['b']/(1-self.beta1**self.t))

    # function to calculate loss
    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + self.epsilon), axis=1))

    # function to train the network
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs):

        # Convert integer labels to one-hot if they aren’t already
        if len(y_train.shape) == 1:
            y_train = np.eye(10)[y_train]
            y_val = np.eye(10)[y_val]
            y_test = np.eye(10)[y_test]
        
        # number of samples in training data
        n_train = len(X_train)

        # iterate for all epochs
        for epoch in range(epochs):

            # shuffle the training data
            perm = np.random.permutation(n_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # iterate over all batches
            for i in range(0, n_train, self.batch_size):

                # get the batch data
                batch_X = X_train_shuffled[i:i+self.batch_size]
                batch_y = y_train_shuffled[i:i+self.batch_size]
                
                # forward pass
                output = self.forward(batch_X)

                # backward pass
                grads = self.backward(batch_X, batch_y, output)

                # update parameters
                self.update_parameters(grads)
            
            # calculate predictions for training, validation and test data
            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)
            test_pred = self.forward(X_test)
            
            # calculate loss for training, validation and test data
            train_loss = self.compute_loss(train_pred, y_train)
            val_loss = self.compute_loss(val_pred, y_val)
            test_loss = self.compute_loss(test_pred, y_test)
            
            # calculate accuracy for training, validation and test data
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
            
            
            for i, layer in enumerate(self.layers):
                self.weights[f'W_{i}'] = layer['W'].copy()
                self.biases[f'b_{i}'] = layer['b'].copy()

            # log the metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc
            })


            # print the metrics
            print(f"Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

        
    def return_weights_and_bias(self):
        return self.weights, self.biases
        