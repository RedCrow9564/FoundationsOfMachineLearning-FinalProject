from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras import backend as K
from infrastructure.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LocalResponseNormalization
from infrastructure.optimizers import create_optimizer


# A shared basis for all linear models. All these models inherit this class.
class _CNNClassifier(Sequential):
    def __init__(self, layers, classes_num, weights_file):
        super(_CNNClassifier, self).__init__(layers=layers)
        self._classes_num = classes_num
        self._callbacks = []
        if weights_file is not None:
            self.load_weights(weights_file)

    def train(self, data, labels, epochs, batch_size, optimizer_config, loss_func, metrics, x_val, y_val,
              log_training, log_tensorboard):
        optimizer = create_optimizer(optimizer_config)
        self.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

        verbose = 0
        if log_training:
            verbose = 1

        if log_tensorboard:
            tensorboard_callback = TensorBoard(log_dir='./logs/TensorBoard')
            tensorboard_callback.set_model(self)
            self._callbacks.append(tensorboard_callback)

        self.fit(data, labels, batch_size, epochs, verbose, validation_data=(x_val, y_val), callbacks=None)


def create_model(model_name, weights_file=None):
    """
    A factory for CNN models creation. Models are declared only is this function, so they can NOT be
    created in any other way (such as import to another file).

    :param model_name: The name of the requested model.
    :param weights_file: Optional. The path to the file which contains previously computed weights for the model.
    :return: A CNN model.
    """

    class _TensorFlowMNISTNet(_CNNClassifier):
        """
        A model based on an example from TensorFlow: https://www.tensorflow.org/tutorials/layers
        """
        def __init__(self, initial_weights_file):
            mnist_classes = 10
            layers = [
                Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), name="First_conv"),
                Activation(activation='relu', name='First_Relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2, name='First_MaxPool'),
                Conv2D(filters=64, kernel_size=(5, 5), padding='same', name='Second_conv'),
                Activation(activation='relu', name='Second_Relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2, name='Second_MaxPool'),
                Flatten(name='Flatten_image_to_vectors_layer'),
                Dense(units=1024, name='First_fully_connected'),
                Activation(activation='relu', name='Third_Relu'),
                Dropout(rate=0.4, name='Dropout_layer'),
                Dense(units=mnist_classes, name='Second_fully_connected'),
                Activation(activation='softmax', name='Softmax')
            ]
            super(_TensorFlowMNISTNet, self).__init__(layers, mnist_classes, initial_weights_file)

    class _TensorFlowCIFAR10Net(_CNNClassifier):
        """
        A model based on Keras example:
        """
        def __init__(self, initial_weights_file):

            cifar10_classes = 10
            layers = [
                Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
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
            super(_TensorFlowCIFAR10Net, self).__init__(layers, cifar10_classes, initial_weights_file)

    # A dictionary which matches models names to their matching classes.
    # To add new models, add their name and class here.
    _models_names_to_classes = {
        'TensorFlow MNIST Net': _TensorFlowMNISTNet,
        'TensorFlow CIFAR10 Net': _TensorFlowCIFAR10Net
    }

    selected_model = _models_names_to_classes[model_name]
    model = selected_model(initial_weights_file=weights_file)

    layers_output_func = K.function([model.layers[0].input, K.learning_phase()],
                                    [layer.output for layer in model.layers])

    return selected_model(initial_weights_file=weights_file), layers_output_func
