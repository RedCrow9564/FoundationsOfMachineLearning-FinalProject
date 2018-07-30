#!/usr/bin/env python  #  Shebang line, needed for running as a script from a unix terminal.

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import time
import argparse
import pandas as pd

from infrastructure.metrics import create_metrics
from infrastructure.loss import create_loss_func
from infrastructure.models import create_model, get_all_layers_output
from infrastructure.datasets import create_dataset
from utils import read_experiments_config, save_model_weights, save_layers_logs

__author__ = "Elad Eatah"
__copyright__ = "Copyright 2018"
__license__ = "MIT"
__version__ = "1.0.1"
__email__ = "eladeatah@mail.tau.ac.il"
__status__ = "Development"


def _parse_input():
    parser = argparse.ArgumentParser(description='Performs CNN analysis according to the input config.')
    parser.add_argument('-i', '--experiments_file', default='experiments_config.json', type=str,
                        help='A path to the experiments config file.')
    args = parser.parse_args()
    experiments_config_path = args.experiments_file
    return experiments_config_path


def main():

    experiment_config_path = _parse_input()
    all_experiments = read_experiments_config(experiment_config_path)

    for experiment_name, experiment_config in all_experiments.items():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            results, model = perform_experiment(experiment_config)
            weights_file_path = save_model_weights(experiment_name, model)
            save_layers_logs(results['Layers Output'])


# TODO: Add inner layers results, only when run by the GPU VM.
def perform_experiment(experiment_config):
    initial_seed = 5
    seed(initial_seed)  # Initializing numpy seed.
    set_random_seed(initial_seed)  # Initializing TensorFlow seed.

    optimizer = experiment_config['Optimizer']
    loss = create_loss_func(experiment_config['Loss func'])
    epochs = experiment_config['Epochs num']
    batch_size = experiment_config['Batch size']

    x_train, y_train, x_test, y_test = create_dataset(experiment_config['Dataset'])
    initial_weights_file = None
    if 'Initial weights file' in experiment_config:
        initial_weights_file = experiment_config['Initial weights file']
    model = create_model(experiment_config['Model name'], initial_weights_file)
    sampled_metrics = create_metrics(['Total Accuracy', 'Average precision', 'Average recall'])

    time.time()
    model.train(x_train, y_train, epochs, batch_size, optimizer, loss, sampled_metrics, x_test, y_test,
                log_training=True, log_tensorboard=True)
    learning_time = time.time()

    score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    layers_output = get_all_layers_output(model, x_test, learning_phase='Testing')

    results = {
        'Training Time [sec]': learning_time,
        'Test Loss': score[0],
        'Test Accuracy': score[1],
        'Test Mean Precision': score[2],
        'Test Mean Recall': score[3],
        'Precision per class': precision_score(y_test, y_pred, average=None),
        'Recall per class': recall_score(y_test, y_pred, average=None),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Layers Output': layers_output
    }

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(results['Confusion Matrix'])

    return results, model


if __name__ == '__main__':
    #main()
    pd.DataFrame([1, 2, 3, 4]).to_csv(r'logs/out.csv', index=False)
