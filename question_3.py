import numpy as np

from question_2 import FFNN
from config import run
import wandb
class AdvancedFFNN(FFNN):
    def __init__(self, input_size=784, hidden_layers=[128, 64], output_size=10, 
                 learning_rate=0.01, optimizer='sgd', weight_decay=0, 
                 activation='sigmoid', batch_size=32, weight_init='random'):
        super().__init__(input_size, hidden_layers, output_size, learning_rate)
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.activation = activation
        self.weight_init = weight_init
        
        self.velocity = [{} for _ in self.layers]
        self.m = [{} for _ in self.layers]
        self.v = [{} for _ in self.layers]
        self.t = 0
        self.epsilon = 1e-8
        self.beta1, self.beta2 = 0.9, 0.999
        self.rho = 0.9
        
        if weight_init == 'xavier':
            self.initialize_xavier()

    def initialize_xavier(self):
        for i, layer in enumerate(self.layers):
            fan_in = layer['W'].shape[0]
            fan_out = layer['W'].shape[1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            layer['W'] = np.random.uniform(-limit, limit, layer['W'].shape)

    def activation_func(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return self.sigmoid(x)

    def activation_deriv(self, x):
        if self.activation == 'sigmoid':
            s = self.sigmoid(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        self.activations = [x]
        self.z_values = []
        
        for layer in self.layers[:-1]:
            z = np.dot(x, layer['W']) + layer['b']
            self.z_values.append(z)
            x = self.activation_func(z)
            self.activations.append(x)
        
        z = np.dot(x, self.layers[-1]['W']) + self.layers[-1]['b']
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        return output

    def backward(self, x, y, output):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        m = x.shape[0]
        grads = [{'W': 0, 'b': 0} for _ in self.layers]
        
        delta = output - y  # y must be one-hot encoded here
        grads[-1]['W'] = np.dot(self.activations[-2].T, delta) / m
        grads[-1]['b'] = np.sum(delta, axis=0, keepdims=True) / m
        
        for i in range(len(self.layers)-2, -1, -1):
            delta = np.dot(delta, self.layers[i+1]['W'].T) * self.activation_deriv(self.z_values[i])
            grads[i]['W'] = np.dot(self.activations[i].T, delta) / m
            grads[i]['b'] = np.sum(delta, axis=0, keepdims=True) / m
            
        for i in range(len(self.layers)):
            grads[i]['W'] += self.weight_decay * self.layers[i]['W']
            
        return grads

    def update_parameters(self, grads):
        self.t += 1
        for i, layer in enumerate(self.layers):
            if self.optimizer == 'sgd':
                layer['W'] -= self.learning_rate * grads[i]['W']
                layer['b'] -= self.learning_rate * grads[i]['b']
            elif self.optimizer == 'momentum':
                if i not in self.velocity:
                    self.velocity[i] = {'W': 0, 'b': 0}
                self.velocity[i]['W'] = 0.9 * self.velocity[i]['W'] - self.learning_rate * grads[i]['W']
                self.velocity[i]['b'] = 0.9 * self.velocity[i]['b'] - self.learning_rate * grads[i]['b']
                layer['W'] += self.velocity[i]['W']
                layer['b'] += self.velocity[i]['b']
            elif self.optimizer == 'nesterov':
                if i not in self.velocity:
                    self.velocity[i] = {'W': 0, 'b': 0}
                temp_W = layer['W'] + 0.9 * self.velocity[i]['W']
                temp_b = layer['b'] + 0.9 * self.velocity[i]['b']
                self.velocity[i]['W'] = 0.9 * self.velocity[i]['W'] - self.learning_rate * grads[i]['W']
                self.velocity[i]['b'] = 0.9 * self.velocity[i]['b'] - self.learning_rate * grads[i]['b']
                layer['W'] = temp_W + self.velocity[i]['W']
                layer['b'] = temp_b + self.velocity[i]['b']
            elif self.optimizer == 'rmsprop':
                if i not in self.v:
                    self.v[i] = {'W': 0, 'b': 0}
                self.v[i]['W'] = self.rho * self.v[i]['W'] + (1-self.rho) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.rho * self.v[i]['b'] + (1-self.rho) * np.square(grads[i]['b'])
                layer['W'] -= self.learning_rate * grads[i]['W'] / (np.sqrt(self.v[i]['W']) + self.epsilon)
                layer['b'] -= self.learning_rate * grads[i]['b'] / (np.sqrt(self.v[i]['b']) + self.epsilon)
            elif self.optimizer == 'adam':
                if i not in self.m:
                    self.m[i] = {'W': 0, 'b': 0}
                    self.v[i] = {'W': 0, 'b': 0}
                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1-self.beta1) * grads[i]['W']
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1-self.beta1) * grads[i]['b']
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1-self.beta2) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1-self.beta2) * np.square(grads[i]['b'])
                m_hat_w = self.m[i]['W'] / (1 - self.beta1**self.t)
                m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
                v_hat_w = self.v[i]['W'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)
                layer['W'] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer['b'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            elif self.optimizer == 'nadam':
                if i not in self.m:
                    self.m[i] = {'W': 0, 'b': 0}
                    self.v[i] = {'W': 0, 'b': 0}
                m_w = self.beta1 * self.m[i]['W'] + (1-self.beta1) * grads[i]['W']
                m_b = self.beta1 * self.m[i]['b'] + (1-self.beta1) * grads[i]['b']
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1-self.beta2) * np.square(grads[i]['W'])
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1-self.beta2) * np.square(grads[i]['b'])
                m_hat_w = m_w / (1 - self.beta1**self.t)
                m_hat_b = m_b / (1 - self.beta1**self.t)
                v_hat_w = self.v[i]['W'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)
                layer['W'] -= self.learning_rate / (np.sqrt(v_hat_w) + self.epsilon) * \
                             (self.beta1 * m_hat_w + (1-self.beta1) * grads[i]['W']/(1-self.beta1**self.t))
                layer['b'] -= self.learning_rate / (np.sqrt(v_hat_b) + self.epsilon) * \
                             (self.beta1 * m_hat_b + (1-self.beta1) * grads[i]['b']/(1-self.beta1**self.t))

    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + self.epsilon), axis=1))

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs):
        # Convert integer labels to one-hot if they aren’t already
        if len(y_train.shape) == 1:
            y_train = np.eye(10)[y_train]
            y_val = np.eye(10)[y_val]
            y_test = np.eye(10)[y_test]
        
        n_train = len(X_train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for i in range(0, n_train, self.batch_size):
                batch_X = X_train_shuffled[i:i+self.batch_size]
                batch_y = y_train_shuffled[i:i+self.batch_size]
                
                output = self.forward(batch_X)
                grads = self.backward(batch_X, batch_y, output)
                self.update_parameters(grads)
            
            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)
            test_pred = self.forward(X_test)
            
            train_loss = self.compute_loss(train_pred, y_train)
            val_loss = self.compute_loss(val_pred, y_val)
            test_loss = self.compute_loss(test_pred, y_test)
            
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
            
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc
            })
            print(f"Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")