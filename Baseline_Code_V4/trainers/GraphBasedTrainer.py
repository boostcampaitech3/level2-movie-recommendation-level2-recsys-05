import torch
from trainers.trainer import Trainer
from torch.utils.data import DataLoader
from trainers.Optimizer import Optimizer
from importlib import import_module


class GraphBasedTrainer(Trainer):
    def __init__(self, model, args, margs, dataset) -> None:
        super().__init__(model, args, margs, dataset)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimizer 클래스 인스턴스 생성
        optimizer_instance = Optimizer()
        # margs에 optimizer라고 명시된 optimizer attribute 이용
        optimizer_func = getattr(optimizer_instance, margs.optimizer)
        self.optimizer = optimizer_func(
            self.model.parameters(), lr=margs.lr) #weight_decay=margs.weight_decay
        self.n_batch = args.n_batch
        
    def train(self):
        self.model.train()
        loss_val = 0
        for _ in range(1, self.n_batch + 1):
            user, pos, neg = self.dataset.sampling()
            
            self.optimizer.zero_grad()
            
            loss = self.model(user, pos, neg)
            loss_val += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
        loss_val /= self.n_batch
        
        return loss_val


    def evaluate(self):
        print("not implemented yet")
        pass

    def preprocess_dataset(self, margs):
        preprocess_module = getattr(
            import_module(f"dataset.{margs.preprocessor}"), margs.preprocessor
        )
        return preprocess_module(self.dataset.num_user)