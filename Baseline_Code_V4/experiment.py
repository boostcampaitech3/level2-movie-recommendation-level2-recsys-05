from importlib import import_module
from dataset.dataset import Dataset
from trainers.trainer import Trainer
import os
import torch
import torch.nn as nn
from utils import increment_path

import mlflow
import mlflow.pytorch

class Experiment:
    def __init__(self, args, margs) -> None:
        self.args = args
        self.margs = margs
        self.dataset: Dataset = self.init_dataset(args, margs)
        self.model: nn.Module = self.init_model(args, margs)
        self.trainer: Trainer = self.init_trainer(
            self.model, args, margs, self.dataset
        )  # 트레이터 인스턴스 init
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = increment_path(os.path.join(args.model_dir, args.name))
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self) -> None:
        print("run experiment")
        self.model.to(self.device)

        best_hit = 0
        for epoch in range(1, self.margs.num_epochs + 1):

            train_loss = self.trainer.train()
            ndcg, hit = self.trainer.evaluate()

            if best_hit < hit:
                best_hit = hit
                print(
                    f"New best model for best hit! HIT@10: {best_hit:.5f} saving the best model.."
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"{self.args.name}_best.pth"),
                )
                mlflow.pytorch.log_model(self.model, f"{self.args.name}_bestModel") 

            print(
                f"Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}"
            )

            mlflow.log_metric("Train_loss", float(train_loss), epoch) 
            mlflow.log_metric("NDCG-10", float(ndcg), epoch) 
            mlflow.log_metric("HIT-10", float(hit), epoch) 
            
            
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
