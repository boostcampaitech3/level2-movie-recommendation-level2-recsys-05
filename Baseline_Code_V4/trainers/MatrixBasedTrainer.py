import torch
import numpy as np
import tqdm
from trainers.trainer import Trainer
from torch.utils.data import DataLoader
from trainers.Optimizer import Optimizer
from importlib import import_module
import trainers.metric as metric


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

        rec_data_iter = tqdm.tqdm(
            self.dataloader,
            desc="Recommendation EP_%s train" % (self.args.model),
            total=len(self.dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        loss_val = 0
        for users in rec_data_iter:
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
        self.model.eval()

        NDCG = 0.0  # NDCG@10
        HIT = 0.0  # HIT@10

        rec_data_iter = tqdm.tqdm(
            self.dataloader,
            desc="Recommendation EP_%s evaluate" % (self.args.model),
            total=len(self.dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        with torch.no_grad():
            for users in rec_data_iter:
                mat = self.dataset.make_matrix(users)
                mat = mat.to(self.device)

                recon_mat = self.model(mat, calculate_loss=False)
                recon_mat[mat == 1] = -np.inf
                rec_list = recon_mat.argsort(dim=1)

                for user, rec in zip(users, rec_list):
                    uv = self.margs.user_valid[user.item()]
                    up = rec[-10:].cpu().numpy().tolist()
                    NDCG += metric.get_ndcg(pred_list=up, true_list=uv)
                    HIT += metric.get_hit(pred_list=up, true_list=uv)

        NDCG /= len(self.dataloader.dataset)
        HIT /= len(self.dataloader.dataset)

        return NDCG, HIT

    def preprocess_dataset(self, margs):
        preprocess_module = getattr(
            import_module(f"dataset.{margs.preprocessor}"), margs.preprocessor
        )
        return preprocess_module(self.dataset.num_user)
