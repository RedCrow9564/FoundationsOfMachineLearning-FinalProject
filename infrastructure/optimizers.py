# TODO: Import keras.optimizers and remove the comments signs (they replace the None values).


def create_optimizer(optimizer_config):
    optimizer_name = optimizer_config['Name']
    learning_rate = optimizer_config['Learning rate']  # Default is 0.01
    decay = optimizer_config['Decay']  # Decay in learning rate between epochs. Default is 0.
    if optimizer_name == 'Gradient Descent':
        momentum = optimizer_config['Momentum']  # Initial momentum. Default is 0.
        optimizer = None  # keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay)

    elif optimizer_name == 'AdaGrad':
        epsilon = optimizer_config['Momentum']  # Initial learning rate normalization. Default is keras' epsilon..
        optimizer = None  # keras.optimizers.Adagrad(lr=learning_rate, epsilon=epsilon, decay=0.0)

    else:
        raise IOError('Unknown optimizer type {0}'.format(optimizer_name))

    return optimizer
