import unittest
import logging
from infrastructure.optimizers import create_optimizer


def _test_creation(optimizer_config):
    exception_flag = False
    try:
        create_optimizer(optimizer_config)
    except Exception as e:
        logging.exception(e)
        exception_flag = True
    return exception_flag


class OptimizersTests(unittest.TestCase):
    def test_sgd_creation(self):
        """
        Test creation of Gradient Descent optimizer.
        The test is successful if and only if no exception occurs.
        :return: None.
        """
        sgd_config = {
            'Name': 'Gradient Descent',
            'Learning rate': 0.1,
            'Decay': 0,
            'Momentum': 0
        }
        self.assertFalse(_test_creation(sgd_config))

    def test_adagrad_creation(self):
        """

        :return:
        """
        adagrad_config = {
            'Name': 'AdaGrad',
            'Learning rate': 0.01,
            'Decay': 0,
            'Epsilon': 1e-10
        }
        self.assertFalse(_test_creation(adagrad_config))


if __name__ == '__main__':
    unittest.main()
