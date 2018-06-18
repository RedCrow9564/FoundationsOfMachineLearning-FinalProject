# TODO: Complete base classifier model.
class _CNNClassifier(object):
    def __init__(self, layers, classes_num):
        pass

# TODO: Consider making the factory a class, to hold the names mapper as a static variable.


def create_model(model_name):

    class _LeNet(object):
        pass  # TODO: Implement specific networks here.

    # Add new specific networks here.
    _models_names_to_classes = {
        'LeNet': _LeNet  # Example
    }

    selected_model = _models_names_to_classes[model_name]
    return selected_model()
