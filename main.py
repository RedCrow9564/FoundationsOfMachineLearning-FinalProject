#!/usr/bin/env python  #  Shebang line, needed for running as a script from a unix terminal.

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tensorflow import set_random_seed
import tensorflow as tf
import time
import argparse
import numpy as np
from os import path

from infrastructure.models import create_model
from infrastructure.datasets import create_dataset
from utils import read_experiments_config, save_model_weights, save_layers_logs, save_experiment_log, weights_files_path

__author__ = "Elad Eatah"
__copyright__ = "Copyright 2018"
__license__ = "MIT"
__version__ = "1.0.1"
__email__ = "eladeatah@mail.tau.ac.il"
__status__ = "Development"


def _parse_input():
    """
    A function for handling terminal commands.

    :return: The path to the experiment configuration file.
    """
    parser = argparse.ArgumentParser(description='Performs CNN analysis according to the input config.')
    parser.add_argument('-i', '--experiments_file', default='experiments_config.json', type=str,
                        help='A path to the experiments config file.')
    args = parser.parse_args()
    experiments_config_path = args.experiments_file
    return experiments_config_path


def main():
    """
    The main function of the project.
    It iterates over all experiments in the config file, performs the experiment
     and saves its results to external files.
    :return: None.
    """

    experiment_config_path = _parse_input()
    all_experiments = read_experiments_config(experiment_config_path)

    for experiment_name, experiment_config in all_experiments.items():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            results, model = perform_experiment(experiment_config)
            save_model_weights(experiment_name, model)
            save_experiment_log(results, experiment_name)
            save_layers_logs(results['Layers Testing Output'], 'Testing')
            save_layers_logs(results['Layers Training Output'], 'Training')


# TODO: Add inner layers results, only when run by the GPU VM.
def perform_experiment(experiment_config):
    """
    The main function which performs the requested experiment.

    :param experiment_config: A relative path to the experiment configuration file.
    :return: A dictionary of results.
    """

    # Initializing seeds for the random numbers generators.
    if 'Numpy seed' in experiment_config:
        initial_seed = experiment_config['Numpy seed']
    else:
        initial_seed = 5

    np.random.seed(initial_seed)  # Initializing numpy seed.

    if 'TensorFlow seed' in experiment_config:
        initial_seed = experiment_config['TensorFlow seed']
    else:
        initial_seed = 5

    set_random_seed(initial_seed)  # Initializing TensorFlow seed.

    # Reading the config data from the file.
    optimizer = experiment_config['Optimizer']
    loss = experiment_config['Loss func']
    epochs = experiment_config['Epochs num']
    batch_size = experiment_config['Batch size']

    x_train, y_train, x_test, y_test = create_dataset(experiment_config['Dataset'])
    initial_weights_file = None
    if 'Initial weights file' in experiment_config:
        initial_weights_file = path.join(weights_files_path, experiment_config['Initial weights file'])
    sampled_metrics = experiment_config['Performance metrics']

    # Creating the model the experiment will be performed on.
    model = create_model(experiment_config['Model name'], initial_weights_file)

    # Train the new model. Training time is measured.
    start_time = time.time()
    model.train(x_train, y_train, epochs, batch_size, optimizer, loss, sampled_metrics, x_test, y_test,
                log_training=True, log_tensorboard=True)
    learning_time = time.time() - start_time

    # Reading all the performance details of this model, including inner layers outputs.
    score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    confusion_mat = confusion_matrix(y_test, y_pred)
    layers_training_output = model.get_layers_output(x_train, learning_phase='Testing')
    layers_testing_output = model.get_layers_output(x_test, learning_phase='Testing')

    # Saving all results to a single dictionary. Later it will be saved to external files.
    results = {
        'Training Time [sec]': learning_time,
        'Test Loss': score[0],
        'Test Accuracy': score[1],
        'Test Mean Precision': score[2],
        'Test Mean Recall': score[3],
        'Precision per class': np.array2string(precision_score(y_test, y_pred, average=None)),
        'Recall per class': np.array2string(recall_score(y_test, y_pred, average=None)),
        'Confusion Matrix': str(confusion_mat.tolist()),
        'Layers Training Output': layers_training_output,
        'Layers Testing Output': layers_testing_output
    }

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(np.array2string(confusion_mat))

    return results, model


# When running this file, activate main() function.
if __name__ == '__main__':
    main()
