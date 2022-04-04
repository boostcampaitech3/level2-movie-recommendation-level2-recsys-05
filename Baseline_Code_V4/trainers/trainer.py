from abc import ABCMeta, abstractclassmethod
import torch


class Trainer(metaclass=ABCMeta):
    def __init__(self, model, args, margs, train_method) -> None:
        self.args = args
        self.margs = margs
        self.model = model
        self.train_method = train_method
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def evalutate(self):
        pass
