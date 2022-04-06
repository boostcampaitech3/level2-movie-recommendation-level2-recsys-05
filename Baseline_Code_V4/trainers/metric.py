import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_idcg(pred_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    return idcg


def get_dcg(pred_list, true_list):
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    return dcg


def get_ndcg(pred_list, true_list):
    ndcg = get_dcg(pred_list, true_list) / get_idcg(pred_list)
    return ndcg


def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def get_graph_ndcg(pred_items, test_items, test_indices, k):
    """그래프는 배치단위로 계산해야하기 때문에 따로 함수 지정
    """    
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k + 2))).float().to(device)

    dcg = (r[:, :k] / f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1)
    ndcg = dcg / dcg_max

    ndcg[torch.isnan(ndcg)] = 0
    return ndcg
