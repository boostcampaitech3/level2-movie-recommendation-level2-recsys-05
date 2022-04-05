import numpy as np


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
