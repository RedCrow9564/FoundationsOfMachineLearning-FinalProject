from tensorflow.python.keras.models import Sequential
from infrastructure.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LocalResponseNormalization
from infrastructure.optimizers import create_optimizer

_loss_names_to_funcs = {
    'Multi-class cross-entropy': 'categorical_crossentropy'
}


# TODO: Complete base classifier model.
class _CNNClassifier(Sequential):
    def __init__(self, layers, classes_num):
        super(_CNNClassifier, self).__init__(layers=layers)
        self._classes_num = classes_num
        pass

    def train(self, data, labels, epochs, batch_size, optimizer_config, loss_func, metrics, log_training):
        optimizer = create_optimizer(optimizer_config)
        self.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

        verbose = 0
        if log_training:
            verbose = 1

        super(_CNNClassifier, self).fit(data, labels, batch_size, epochs, verbose)


def create_model(model_name):

    class _TensorFlowMNISTNet(_CNNClassifier):
        def __init__(self):
            mnist_classes = 10
            layers = [
                Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=()),
                Activation(activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(filters=64, kernel_size=(5, 5), padding='same'),
                Activation(activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Flatten(),
                Dense(units=1024),
                Activation(activation='relu'),
                Dropout(rate=0.4),
                Dense(units=mnist_classes),
                Activation(activation='softmax')
            ]
            super(_TensorFlowMNISTNet, self).__init__(layers, mnist_classes)

    class _TensorFlowCIFAR10Net(_CNNClassifier):
        def __init__(self):
            # TODO: Replace with the real layers of this network.

            cifar10_classes = 10
            layers = [
                Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),  # TODO: Correct this red signs...
                Activation('relu'),
                Conv2D(32, (3, 3)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Conv2D(64, (3, 3), padding='same'),
                Activation('relu'),
                Conv2D(64, (3, 3)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                Flatten(),
                Dense(512),
                Activation('relu'),
                Dropout(0.5),
                Dense(cifar10_classes),
                Activation('softmax')
            ]
            super(_TensorFlowCIFAR10Net, self).__init__(layers, cifar10_classes)

    # Add new specific networks here.
    _models_names_to_classes = {
        'TensorFlow MNIST Net': _TensorFlowMNISTNet,
        'TensorFlow CIFAR10 Net': _TensorFlowCIFAR10Net
    }

    selected_model = _models_names_to_classes[model_name]
    return selected_model()
