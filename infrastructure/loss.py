_loss_names_to_funcs = {
    'multi-class cross-entropy': 'categorical_crossentropy'
}


def create_loss_func(func_name):
    name_as_lower = func_name.lower()
    if name_as_lower in _loss_names_to_funcs:
        return _loss_names_to_funcs[name_as_lower]
    else:
        raise IOError('Loss func {0} is NOT supported'.format(func_name))
