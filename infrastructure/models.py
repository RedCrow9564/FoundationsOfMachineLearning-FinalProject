from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
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
        labels = to_categorical(labels, self._classes_num)

        verbose = 0
        if log_training:
            verbose = 1

        super(_CNNClassifier, self).fit(data, labels, batch_size, epochs, verbose)

# TODO: Consider making the factory a class, to hold the names mapper as a static variable.


def create_model(model_name, classes_num):

    class _TemsorFlowMNISTNet(_CNNClassifier):
        def __init__(self, classes_num):
            layers = [
                Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Flatten(),
                Dense(units=1024, activation='relu'),
                Dropout(rate=0.4),
                Dense(units=10, activation='softmax')
            ]
            super(_TemsorFlowMNISTNet, self).__init__(layers, classes_num)

    # TODO: Implement specific networks here.

    # Add new specific networks here.
    _models_names_to_classes = {
        'TensorFlow MNIST Net': _TemsorFlowMNISTNet  # Example
    }

    selected_model = _models_names_to_classes[model_name]
    return selected_model(classes_num)
