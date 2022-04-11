from importlib import import_module
from dataset.dataset import Dataset
from trainers.trainer import Trainer
import os
import torch
import torch.nn as nn

class Inference:
    def __init__(self, args, margs) -> None:
        self.args = args
        self.margs = margs
        self.dataset: Dataset = self.init_dataset(args, margs)
        self.model: nn.Module = self.init_model(args, margs)
        self.trainer: Trainer = self.init_trainer(
            self.model, args, margs, self.dataset
        )  # 트레이터 인스턴스 init
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = os.path.join(args.model_dir, args.name)
        self.save_dir = os.path.join(args.submission_path, args.submission_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self) -> None:
        print("run inference")
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.args.name)+'.pt'))
        except:
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.args.name)+'.pth'))

        submision = self.trainer.predict()
        submision.to_csv(os.path.join(self.save_dir, self.args.submission_name)+'.csv', index=False)
        
    def init_model(self, args, margs):
        model_module = getattr(import_module(f"models.{args.model}"), args.model)
        # 모델에서 데이터셋에 의존된 부분이 존재 함으로 모델에 데이터셋 인스턴스를 넘겨줌
        margs["data_instance"] = self.dataset
        return model_module(margs)

    def init_dataset(self, args, margs):
        dataset_module = getattr(import_module(f"dataset.{args.dataset}"), args.dataset)
        return dataset_module(args, margs)

    def init_trainer(self, model, args, margs, dataset):
        trainer_module = getattr(
            import_module(f"trainers.{margs.trainer}"), margs.trainer
        )
        user_train, user_valid = self.dataset.get_train_valid_data()
        margs["user_train"] = user_train
        margs["user_valid"] = user_valid
        return trainer_module(model, args, margs, dataset)
