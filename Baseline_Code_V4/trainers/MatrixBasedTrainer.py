import torch
from trainers.trainer import Trainer
from torch.utils.data import DataLoader
from trainers.Optimizer import Optimizer
from importlib import import_module


class MatrixBasedTrainer(Trainer):
    def __init__(self, model, args, margs, dataset) -> None:
        super().__init__(model, args, margs, dataset)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 여기서는 AEDataset이라는 MatrixDataset에서 한번 더 preprocessed dataset을 이용하므로
        if margs.preprocessor is not None:
            self.preprocessed_dataset = self.preprocess_dataset(margs)

        self.dataloader = DataLoader(
            self.preprocessed_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=margs.num_workers,
        )
        # Optimizer 클래스 인스턴스 생성
        optimizer_instance = Optimizer()
        # margs에 optimizer라고 명시된 optimizer attribute 이용
        optimizer_func = getattr(optimizer_instance, margs.optimizer)
        self.optimizer = optimizer_func(
            self.model.parameters(), lr=margs.lr, weight_decay=margs.weight_decay
        )

    def train(self):
        self.model.train()
        loss_val = 0
        for users in self.dataloader:
            mat = self.dataset.make_matrix(users)
            mat = mat.to(self.device)
            loss = self.model(mat)

            self.optimizer.zero_grad()

            loss_val += loss.item()

            loss.backward()
            self.optimizer.step()

        loss_val /= len(self.dataloader)

        return loss_val

    def evaluate(self):
        print("not implemented yet")
        pass

    def preprocess_dataset(self, margs):
        preprocess_module = getattr(
            import_module(f"dataset.{margs.preprocessor}"), margs.preprocessor
        )
        return preprocess_module(self.dataset.num_user)
