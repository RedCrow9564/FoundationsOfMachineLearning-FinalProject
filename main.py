#!/usr/bin/env python  #  Shebang line, needed for running as a script from a unix terminal.

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import os
import time
import datetime

from infrastructure.metrics import create_metrics
from infrastructure.loss import create_loss_func
from infrastructure.models import create_model, get_all_layers_output
from infrastructure.datasets import create_dataset

__author__ = "Elad Eatah"
__copyright__ = "Copyright 2018"
__license__ = "MIT"
__version__ = "1.0.1"
__email__ = "eladeatah@mail.tau.ac.il"
__status__ = "Development"

_logs_dir = os.path.join('logs')


def main():
    all_experiments = {
        'Test MNIST': {
            'Model name': 'TensorFlow CIFAR10 Net',
            'Optimizer': {
                'Name': 'AdaGrad',
                'Learning rate': 0.02,
                'Decay': 0,
                'Epsilon': 1e-7,  # Relevant only for Adagrad.
                'Momentum': 0.9  # Relevant only for Gradient Descent.
            },
            'Dataset': 'CIFAR10',
            'Loss func': 'Multi-class cross-entropy',
            'Epochs num': 1,
            'Batch size': 128
        }
    }

    for experiment_name, experiment_config in all_experiments.items():
        results, model = perform_experiment(experiment_config)

        weights_file_name = experiment_name + '_' + str(datetime.datetime.today()) + '.h5'
        weights_file_name.replace(' ', '_')
        weights_file_path = os.path.join(_logs_dir, 'ModelsWeights', weights_file_name)
        model.save_weights(filepath=weights_file_path)


def perform_experiment(experiment_config):
    initial_seed = 5
    seed(initial_seed)  # Initializing numpy seed.
    set_random_seed(initial_seed)  # Initializing TensorFlow seed.

    optimizer = experiment_config['Optimizer']
    loss = create_loss_func(experiment_config['Loss func'])
    epochs = experiment_config['Epochs num']
    batch_size = experiment_config['Batch size']

    x_train, y_train, x_test, y_test = create_dataset(experiment_config['Dataset'])
    model = create_model(experiment_config['Model name'])
    sampled_metrics = create_metrics(['Total Accuracy', 'Average precision', 'Average recall'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        time.time()
        model.train(x_train, y_train, epochs, batch_size, optimizer, loss, sampled_metrics, x_test, y_test,
                    log_training=True, log_tensorboard=True)
        learning_time = time.time()

        score = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)

        results = {
            'Training Time [sec]': learning_time,
            'Test Loss': score[0],
            'Test Accuracy': score[1],
            'Test Mean Precision': score[2],
            'Test Mean Recall': score[3],
            'Precision per class': precision_score(y_test, y_pred, average=None),
            'Recall per class': recall_score(y_test, y_pred, average=None),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(results['Confusion Matrix'])
        #output = get_all_layers_output(model, x_test, learning_phase='Testing')
        #print(output[0])
        return results, model


if __name__ == '__main__':
    main()
