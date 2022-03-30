import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, make_graph_data_set, optimizer, n_batch):
    model.train()
    loss_val = 0
    for step in range(1, n_batch + 1):
        user, pos, neg = make_graph_data_set.sampling()
        optimizer.zero_grad()
        loss = model(user, pos, neg)
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
    loss_val /= n_batch
    return loss_val


def split_matrix(X, n_splits=10):
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits


def compute_ndcg_k(pred_items, test_items, test_indices, k):

    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k + 2))).float().to(device)

    dcg = (r[:, :k] / f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1)
    ndcg = dcg / dcg_max

    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def evaluate(u_emb, i_emb, Rtr, Rte, k=10):

    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k = [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().to(device)
        non_train_items = torch.from_numpy(1 - (tr_f.todense())).float().to(device)
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)

        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(
            dim=1,
            index=test_indices,
            src=torch.ones_like(test_indices).float().to(device),
        )

        topk_preds = torch.zeros_like(scores).float()
        topk_preds.scatter_(
            dim=1, index=test_indices[:, :k], src=torch.ones_like(test_indices).float()
        )

        TP = (test_items * topk_preds).sum(1)
        rec = TP / test_items.sum(1)

        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(ndcg_k).mean(), torch.cat(recall_k).mean()
