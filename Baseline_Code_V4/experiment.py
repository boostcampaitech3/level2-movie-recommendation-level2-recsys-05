from importlib import import_module
from dataset.dataset import Dataset
from trainers.trainer import Trainer
import torch
import torch.nn as nn


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

    def run(self) -> None:
        print("run experiment")
        print(self.model)

        self.model.to(self.device)

        # TODO: train.py 의 main 함수를 구현

        # best_hit = 0
        for epoch in range(1, self.margs.num_epochs + 1):
            self.trainer.train()

        #     ndcg, hit = matrix_evaluate(
        #         model=model,
        #         data_loader=data_loader,
        #         user_valid=user_valid,
        #         make_data_set=make_matrix_data_set,
        #     )

        #     if best_hit < hit:
        #         best_hit = hit
        #         torch.save(
        #             model.state_dict(), os.path.join(config.model_path, config.model_name)
        #         )

        # print(
        #     f"Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}"
        # )

        pass

    def init_model(self, args, margs):
        model_module = getattr(import_module(f"models.{args.model}"), args.model)
        margs["data_instance"] = self.dataset
        return model_module(margs)

    def init_dataset(self, args, margs):
        dataset_module = getattr(import_module(f"dataset.{args.dataset}"), args.dataset)
        return dataset_module(args, margs)

    def init_trainer(self, model, args, margs, dataset):
        trainer_module = getattr(
            import_module(f"trainers.{margs.trainer}"), margs.trainer
        )
        return trainer_module(model, args, margs, dataset)
