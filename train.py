import argparse
import wandb
from question_3 import AdvancedFFNN

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration')
    
    # Weights & Biases arguments
    parser.add_argument('-wp', '--wandb_project', 
                       default='myprojectname',
                       type=str,
                       help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-we', '--wandb_entity',
                       default='myname',
                       type=str,
                       help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset and training parameters
    parser.add_argument('-d', '--dataset',
                       default='fashion_mnist',
                       choices=['mnist', 'fashion_mnist'],
                       help='Dataset to use for training')
    
    parser.add_argument('-e', '--epochs',
                       default=1,
                       type=int,
                       help='Number of epochs to train neural network')
    
    parser.add_argument('-b', '--batch_size',
                       default=4,
                       type=int,
                       help='Batch size used to train neural network')
    
    # Loss and optimizer parameters
    parser.add_argument('-l', '--loss',
                       default='cross_entropy',
                       choices=['mean_squared_error', 'cross_entropy'],
                       help='Loss function to use')
    
    parser.add_argument('-o', '--optimizer',
                       default='sgd',
                       choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                       help='Optimizer to use')
    
    parser.add_argument('-lr', '--learning_rate',
                       default=0.1,
                       type=float,
                       help='Learning rate used to optimize model parameters')
    
    # Optimizer-specific parameters
    parser.add_argument('-m', '--momentum',
                       default=0.5,
                       type=float,
                       help='Momentum used by momentum and nag optimizers')
    
    parser.add_argument('-beta', '--beta',
                       default=0.5,
                       type=float,
                       help='Beta used by rmsprop optimizer')
    
    parser.add_argument('-beta1', '--beta1',
                       default=0.5,
                       type=float,
                       help='Beta1 used by adam and nadam optimizers')
    
    parser.add_argument('-beta2', '--beta2',
                       default=0.5,
                       type=float,
                       help='Beta2 used by adam and nadam optimizers')
    
    parser.add_argument('-eps', '--epsilon',
                       default=0.000001,
                       type=float,
                       help='Epsilon used by optimizers')
    
    parser.add_argument('-w_d', '--weight_decay',
                       default=0.0,
                       type=float,
                       help='Weight decay used by optimizers')
    
    # Network architecture parameters
    parser.add_argument('-w_i', '--weight_init',
                       default='random',
                       choices=['random', 'Xavier'],
                       help='Weight initialization method')
    
    parser.add_argument('-nhl', '--num_layers',
                       default=1,
                       type=int,
                       help='Number of hidden layers used in feedforward neural network')
    
    parser.add_argument('-sz', '--hidden_size',
                       default=4,
                       type=int,
                       help='Number of hidden neurons in a feedforward layer')
    
    parser.add_argument('-a', '--activation',
                       default='sigmoid',
                       choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                       help='Activation function to use')

    # Parse arguments and return
    args = parser.parse_args()
    return args


config = parse_arguments()
wandb.init(project=config.wandb_project, entity=config.wandb_entity)

if config.dataset == 'mnist':
    from question_10 import X_train, y_train, X_test, y_test, X_val, y_val
else:
    from question_1 import X_train, y_train, X_test, y_test, X_val, y_val

epochs = config.epochs
batch_size = config.batch_size
loss = config.loss
optimizer = config.optimizer
lr = config.learning_rate
momentum = config.momentum
beta = config.beta
beta1 = config.beta1
beta2 = config.beta2
epsion = config.epsilon
weight_decay = config.weight_decay
weight_init = config.weight_init
num_layers = config.num_layers
hidden_size = config.hidden_size
activation = config.activation

model = AdvancedFFNN()