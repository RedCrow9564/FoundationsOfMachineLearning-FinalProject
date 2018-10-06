"""
optimizers.py - Contains all supported optimization methods. create_optimizer is the factory for optimizers creation.
"""

from keras import optimizers


def create_optimizer(optimizer_config):
    optimizer_name = optimizer_config['Name']  # The name of the chosen model.
    learning_rate = optimizer_config['Learning rate']  # Initial learning rate.
    if optimizer_name == 'Gradient Descent':
        decay = optimizer_config['Decay']  # Decay in learning rate between epochs.
        momentum = optimizer_config['Momentum']  # Initial momentum. Default is 0.
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=True)

    elif optimizer_name == 'AdaGrad':
        decay = optimizer_config['Decay']  # Decay in learning rate between epochs.
        epsilon = optimizer_config['Epsilon']  # Initial learning rate normalization.
        optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=epsilon, decay=decay)

    elif optimizer_name == 'AdaDelta':
        decay = optimizer_config['Decay']  # Decay in learning rate between epochs.
        optimizer = optimizers.Adadelta(lr=learning_rate, decay=decay)

    elif optimizer_name == 'Adam':
        decay = optimizer_config['Decay']  # Decay in learning rate between epochs.
        optimizer = optimizers.Adadelta(lr=learning_rate, decay=decay)

    elif optimizer_name == 'Nadam':
        optimizer = optimizers.Nadam(lr=learning_rate)

    else:
        raise IOError('Unknown optimizer type {0}'.format(optimizer_name))

    return optimizer
