from importlib import import_module
from dataset.dataset import Dataset

# from trainers.trainer import Trainer

import torch.nn as nn


class Experiment:
    def __init__(self, args, margs) -> None:
        self.model: nn.Module = self.init_model(args, margs)
        self.dataset: Dataset = self.init_dataset(args)
        # self.trainer: Trainer = self.init_trainer(self.model, args, margs) # 트레이터 인스턴스 init

    def run(self) -> None:
        print("run experiment")
        print(self.model)
        # TODO: train.py 의 main 함수를 구현
        # self.trainer.train()

        pass

    def init_model(self, args, margs):
        model_module = getattr(import_module(f"models.{args.model}"), args.model)
        return model_module(margs)

    def init_dataset(self, args):
        pass
        # dataset_module = getattr(import_module(f"models.{args.dataset}"), args.dataset)

    def init_trainer(self, args, margs):
        trainer_module = getattr(
            import_module(f"trainers.{args.model}_trainer"), args.model
        )
        return trainer_module(args)
