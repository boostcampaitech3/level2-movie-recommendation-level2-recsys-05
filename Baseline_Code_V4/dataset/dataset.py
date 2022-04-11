from abc import ABCMeta, abstractclassmethod


class Dataset(metaclass=ABCMeta):
    @abstractclassmethod
    def get_train_valid_data(self):
        pass
