from abc import ABCMeta, abstractclassmethod


class Trainer(metaclass=ABCMeta):
    def __init__(self, model, args, margs, dataset) -> None:
        self.args = args
        self.margs = margs
        self.model = model
        self.dataset = dataset

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def evaluate(self):
        pass
