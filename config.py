import argparse

config_args = {
    'training_config': {
        'dataset': ('lastfm', 'Name of dataset'),
        'cuda': (0, 'Which cuda device to use (-1 for CPU training)'),
        'manifold': ('unite', 'Which manifold to use, can be any of [unite, Euclidean, ...]'),
        'lr': (0.001, 'Learning rate'),
        'rg': (0.0, 'L2 regularization'),
        'batch': (512, 'Training batch size'),
        'seed': (1, 'Random seed'),
        'epoch': (1000, 'Number of training epochs'),
        'dropout': (0.5, 'Dropout probability'),
        'beta': (0.2, 'Strength of disentanglement, in (0, oo)'),
        'tau': (0.1, 'Temperature of sigmoid/softmax, in (0, 1)'),
        'std': (0.075, 'Standard deviation of the Gaussian prior'),
        'dim': (100, 'Dimension of each facet'),
        'k': (7, 'Number of facets (macro concepts)'),
        'nogb': (1, 'Disable Gumbel-Softmax sampling'),
        'tensorboard': (False, 'Use tensorboard or not'),
        'component': ('h6', 'List of manifold to be used. e: Euclidean, s: hypersphere, h: hyperboloid')
    }
}

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """
    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)
        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
