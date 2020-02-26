import yaml

# TODO: function 'add' needs modification
class HParams(object):
    # Hyperparameter class using yaml
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def add(self, **kwargs):
        # Needs modification - if key exists, do not update.
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
        return self

    def __repr__(self):
        return '\nHyperparameters:\n' + '\n'.join([' {}={}'.format(k, v) for k, v in self.__dict__.items()])

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.load(f))
