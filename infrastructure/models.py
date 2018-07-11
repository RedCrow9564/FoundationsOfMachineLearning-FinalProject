from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import tensorflow as tf
from os import path
from infrastructure.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LocalResponseNormalization
from infrastructure.optimizers import create_optimizer


def _from_categorical(y):
    return np.argmax(y, axis=1).astype(int)


def get_all_layers_output(model, test_data, learning_phase='Testing'):

    learning_phase_value = 1  # Default for when learning phase is testing
    if learning_phase == 'Training':
        learning_phase_value = 0

    layers_output_func = K.function([model.layers[0].input, K.learning_phase()],
                                    [layer.output for layer in model.layers])

    layers_output = layers_output_func([test_data, learning_phase_value])
    return layers_output

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class _CNNClassifier(Sequential):
    """
    A shared basis for all linear models. All these models inherit this class.
    """
    def __init__(self, layers, classes_num, weights_file, labels_list):
        super(_CNNClassifier, self).__init__(layers=layers)
        self._classes_num = classes_num
        self._labels_list = labels_list
        self._callbacks = []
        self._predictions_to_labels = np.vectorize(lambda pred: self._labels_list[pred])
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
            tensorboard_callback = TrainValTensorBoard(log_dir='./logs/TensorBoard')
            tensorboard_callback.set_model(self)
            self._callbacks.append(tensorboard_callback)

        labels = to_categorical(labels, self._classes_num)
        y_val = to_categorical(y_val, self._classes_num)

        self.fit(data, labels, batch_size, epochs, verbose, validation_data=(x_val, y_val), callbacks=self._callbacks)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        y_pred = super(_CNNClassifier, self).predict(x, batch_size, verbose, steps)
        y_pred = self._predictions_to_labels(_from_categorical(y_pred))
        return y_pred

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
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
            super(_TensorFlowMNISTNet, self).__init__(layers, mnist_classes, initial_weights_file, mnist_labels)

    class _TensorFlowCIFAR10Net(_CNNClassifier):
        """
        A model based on Keras example:
        """
        def __init__(self, initial_weights_file):

            cifar10_classes = 10
            cifar10_labels = list(range(cifar10_classes))
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
            super(_TensorFlowCIFAR10Net, self).__init__(layers, cifar10_classes, initial_weights_file, cifar10_labels)

    # A dictionary which matches models names to their matching classes.
    # To add new models, add their name and class here.
    _models_names_to_classes = {
        'TensorFlow MNIST Net': _TensorFlowMNISTNet,
        'TensorFlow CIFAR10 Net': _TensorFlowCIFAR10Net
    }

    selected_model = _models_names_to_classes[model_name]
    model = selected_model(initial_weights_file=weights_file)
    return model
