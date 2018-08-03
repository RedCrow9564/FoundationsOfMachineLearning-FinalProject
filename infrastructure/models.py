from keras.models import Sequential
from keras.utils import to_categorical
from keras import backend as K
import numpy as np

from infrastructure.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from infrastructure.optimizers import create_optimizer
from infrastructure.loss import create_loss_func
from infrastructure.metrics import create_metrics
from utils import tensorboard_logs_path, TrainValTensorBoard


def _from_categorical(y):
    """
    Function used for de-coding labels, which were encoded in 'one-hot' form,
    to numerical index in the labels list, i.e [0, 0, 1, 0] --> 2.
    :param y: Numpy array of one-hot vectors.
    :return: Integers list of label indices.
    """
    return np.argmax(y, axis=1).astype(int)


def _get_all_layers_output(model, input_data, learning_phase='Testing'):
    """
    Function for retrieving a model's inner layers output.

    :param model: The network object.
    :param input_data: The data which is given as input to the network.
    :param learning_phase: The 'phase' of the network. This is relevant for networks which have layers
     that respond differently when training or testing. I.e Dropout layers. Default is Testing
    :return: A list of lists, which contain each layer's outputs.
    """

    learning_phase_value = 1  # Default for when learning phase is testing
    if learning_phase == 'Training':
        learning_phase_value = 0

    layers_output_func = K.function([model.layers[0].input, K.learning_phase()],
                                    [layer.output for layer in model.layers])

    layers_output = layers_output_func([input_data, learning_phase_value])
    return layers_output


class _CNNClassifier(Sequential):
    """
    A shared basis for all linear models. All CNN models inherit this class.
    """
    def __init__(self, layers, classes_num, labels_list, weights_file):
        """
        Shared constructor for CNN models.

        :param layers: A list of Keras Layers which is the architecture.
        :param classes_num: The number of classes this model can predict.
        :param labels_list: A list of possible labels. Can be numeric or strings. Its length must be classes_num
        :param weights_file: A file of initial weights to be used in the model.
            Can be None to use Keras's initialization.
        """
        super(_CNNClassifier, self).__init__(layers=layers)
        self._classes_num = classes_num
        self._labels_list = labels_list
        self._callbacks = []
        self._predictions_to_labels = np.vectorize(lambda pred: self._labels_list[pred])
        if weights_file is not None:
            self.load_weights(weights_file)

    def train(self, data, labels, epochs, batch_size, optimizer_config, loss_func_name, metrics_names, x_val, y_val,
              log_training, log_tensorboard):
        """
        A method for training a model.

        :param data: The input data on which the model is trained.
        :param labels: The labels of the given input data.
        :param epochs: Number of epochs to perform in the training.
        :param batch_size: Number of samples in each batch of the training.
        :param optimizer_config: A dictionary which contains all the data required to create the optimizer for learning.
        :param loss_func_name: The loss function on which optimization is performed.
        :param metrics_names: The statistics which are measured on each epoch of the training.
        :param x_val: The validation data. Used for plotting TensorBoard statistics, NOT for training.
        :param y_val: The labels for the validation data.
        :param log_training: The flag which decides whether to plot the the terminal the training process.
        :param log_tensorboard: The flag which decides whether to plot to TensorBoard the training process.
        :return: None.
        """
        optimizer = create_optimizer(optimizer_config)
        loss_func = create_loss_func(loss_func_name)
        metrics = create_metrics(metrics_names)
        self.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

        verbose = 0
        if log_training:
            verbose = 1

        if log_tensorboard:
            tensorboard_callback = TrainValTensorBoard(log_dir=tensorboard_logs_path, write_graph=False)
            tensorboard_callback.set_model(self)
            self._callbacks.append(tensorboard_callback)

        labels = to_categorical(labels, self._classes_num)
        y_val = to_categorical(y_val, self._classes_num)

        self.fit(data, labels, batch_size, epochs, verbose, validation_data=(x_val, y_val), callbacks=self._callbacks)

    def get_layers_output(self, input_data, learning_phase='Testing'):
        """
        A method for extracting the output of the inner layers of the model, for a given input.
        
        :param input_data: The data which is given as input to the network.
        :param learning_phase: The 'phase' of the network. This is relevant for networks which have layers
         that respond differently when training or testing. I.e Dropout layers. Default is Testing
        :return: A list of lists, which contain each layer's outputs.
        """
        return _get_all_layers_output(self, input_data, learning_phase)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        """
        An expansion of Keras's predict method- It predicts the 'one-hot' labels for given input samples,
        and then de-codes it back to the models possible labels.
        :param x: The input to predict on it.
        :param batch_size: The size of the batches for each prediction.
        :param verbose: A flag- 1 for
        :param steps:
        :return: List of predictions for each input sample.
        """
        y_pred = super(_CNNClassifier, self).predict(x, batch_size, verbose, steps)
        y_pred = self._predictions_to_labels(_from_categorical(y_pred))
        return y_pred

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        """
        An expansion of Keras's evaluate method- It converts the given labels to one-hot notation,
        and then it performs Keras's evaluate method, for getting the requested metrics for this model.
        :param x: The input to predict on it.
        :param y: The known labels of the given input.
        :param batch_size: The size of the batches for each prediction.
        :param verbose: A flag- 1 for
        :param sample_weight:
        :param steps:
        :return: A list of performance metrics requested (in the constructor) for this model.
        """
        y = to_categorical(y)
        return super(_CNNClassifier, self).evaluate(x, y, batch_size, verbose, sample_weight, steps)


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
            mnist_labels = list(range(mnist_classes))
            layers = [
                Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1),
                       name="First_conv", activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2, name='First_MaxPool'),
                Conv2D(filters=64, kernel_size=(5, 5), padding='same', name='Second_conv', activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2, name='Second_MaxPool'),
                Flatten(name='Flatten_image_to_vectors_layer'),
                Dense(units=1024, name='First_fully_connected', activation='relu'),
                Dropout(rate=0.4, name='Dropout_layer'),
                Dense(units=mnist_classes, name='Second_fully_connected', activation='softmax')
            ]
            super(_TensorFlowMNISTNet, self).__init__(layers, mnist_classes, mnist_labels, initial_weights_file)

    class _TensorFlowCIFAR10Net(_CNNClassifier):
        """
        A model based on Keras example:
        TODO: Complete this comment.
        """
        def __init__(self, initial_weights_file):

            cifar10_classes = 10
            cifar10_labels = list(range(cifar10_classes))
            layers = [
                Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
                Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(rate=0.25),

                Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(rate=0.25),

                Flatten(),
                Dense(units=512, activation='relu'),
                Dropout(rate=0.5),
                Dense(cifar10_classes, activation='softmax')
            ]
            super(_TensorFlowCIFAR10Net, self).__init__(layers, cifar10_classes, cifar10_labels, initial_weights_file)

    # A dictionary which matches models names to their matching classes.
    # To add new models, add their name and class here.
    _models_names_to_classes = {
        'TensorFlow MNIST Net': _TensorFlowMNISTNet,
        'TensorFlow CIFAR10 Net': _TensorFlowCIFAR10Net
    }

    selected_model = _models_names_to_classes[model_name]
    model = selected_model(initial_weights_file=weights_file)
    return model
