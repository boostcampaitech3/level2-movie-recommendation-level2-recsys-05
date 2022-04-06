import torch
from trainers.trainer import Trainer
from trainers.Optimizer import Optimizer
from importlib import import_module
import trainers.metric as metric

class GraphBasedTrainer(Trainer):
    def __init__(self, model, args, margs, dataset): # -> None:
        super().__init__(model, args, margs, dataset)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimizer 클래스 인스턴스 생성
        optimizer_instance = Optimizer()
        # margs에 optimizer라고 명시된 optimizer attribute 이용
        optimizer_func = getattr(optimizer_instance, margs.optimizer)
        self.optimizer = optimizer_func(
            self.model.parameters(), lr=margs.lr, weight_decay=margs.weight_decay
        )
        
        self.n_batch = margs.n_batch
        self.Rtr, self.Rte, self.R_total = self.dataset.get_R_data()
        self.k = margs.k
        
        
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
        self.model.eval()
        self.u_emb, self.i_emb = self.model.get_f_embeddings()

        # split matrices
        ue_splits = self.dataset.split_matrix(self.u_emb)
        tr_splits = self.dataset.split_matrix(self.Rtr)
        te_splits = self.dataset.split_matrix(self.Rte)

        recall_k, ndcg_k = [], []
        ## compute results for split matrices
        for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

            scores = torch.mm(ue_f, self.i_emb.t())

            test_items = torch.from_numpy(te_f.todense()).float().to(self.device)
            non_train_items = torch.from_numpy(1 - (tr_f.todense())).float().to(self.device)
            scores = scores * non_train_items

            _, test_indices = torch.topk(scores, dim=1, k=self.k)

            pred_items = torch.zeros_like(scores).float()
            pred_items.scatter_(
                dim=1,
                index=test_indices,
                src=torch.ones_like(test_indices).float().to(self.device),
            )

            topk_preds = torch.zeros_like(scores).float()
            topk_preds.scatter_(
                dim=1, index=test_indices[:, :self.k], src=torch.ones_like(test_indices).float()
            )

            TP = (test_items * topk_preds).sum(1)
            rec = TP / test_items.sum(1)

            ndcg = metric.get_graph_ndcg(pred_items, test_items, test_indices, self.k)

            recall_k.append(rec)
            ndcg_k.append(ndcg)

        return torch.cat(ndcg_k).mean(), torch.cat(recall_k).mean()
            
    def preprocess_dataset(self, margs):
        preprocess_module = getattr(
            import_module(f"dataset.{margs.preprocessor}"), margs.preprocessor
        )
        return preprocess_module(self.dataset.num_user)