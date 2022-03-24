import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    """
    negative sampling을 수행

    Args:
        item_set (set): 아이템 집합
        item_size (int): 아이템 사이즈

    Returns:
        int: 랜덤으로 추출한 아이템
    """    
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    """
    valid_rating_matrix 생성
    
    --------------------------------------------
    item_list가 [0, 1, 2, 3, 4, 5, 6]일 때,
    valid가 사용하는 item_list -> [0, 1, 2, 3, 4]
    --------------------------------------------
    
    Args:
        user_seq (list): 유저 시퀀스 (2차원 리스트 형태)
        num_users (int): 유저 수
        num_items (int): 아이템 수

    Returns:
        scipy.sparse._csr.csr_matrix: rating_matrix 생성
    """        
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):   # user_seq의 인덱스는 0~유저수인데, 그대로 쓰나봄
        for item in item_list[:-2]:  # 이건 왜 또 -3까지만 가져오는가,,? 
            row.append(user_id)      # row - 유저
            col.append(item)         # col - 아이템
            data.append(1)           # 있음을 체크

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 유저-아이템이 있는 곳에 평점 매트릭스를 만들어줌 row와 col에 해당하는 셀에 1을 넣는 방식
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    
    """
    text_rating_matrix 생성
    
    --------------------------------------------
    item_list가 [0, 1, 2, 3, 4, 5, 6]일 때,
    test가 사용하는 item_list -> [0, 1, 2, 3, 4, 5]
    --------------------------------------------
    
    Args:
        user_seq (list): 유저 시퀀스 (2차원 리스트 형태)
        num_users (int): 유저 수
        num_items (int): 아이템 수

    Returns:
        scipy.sparse._csr.csr_matrix: rating_matrix 생성
    """    
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):  # user_seq의 인덱스는 0~유저수인데, 그대로 쓰나봄
        for item in item_list[:-1]:  # 이건 왜 또 -2까지만 가져오는가,,? 
            row.append(user_id)      # row - 유저
            col.append(item)         # col - 아이템
            data.append(1)           # 있음을 체크

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 유저-아이템이 있는 곳에 평점 매트릭스를 만들어줌 row와 col에 해당하는 셀에 1을 넣는 방식
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    """
    submission_rating_matrix 생성

    --------------------------------------------
    item_list가 [0, 1, 2, 3, 4, 5, 6] 일 때,
    submission가 사용하는 item_list -> [0, 1, 2, 3, 4, 5, 6]   
    ---------------------------------------------

    Args:
        user_seq (list): 유저 시퀀스 (2차원 리스트 형태)
        num_users (int): 유저 수
        num_items (int): 아이템 수

    Returns:
        scipy.sparse._csr.csr_matrix: rating_matrix 생성
    """    
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )


def get_user_seqs(data_file):
    """
    데이터 파일을 받아 user_seq, max_item을 생성하고 valid, test, submission의 rating matrix 생성 및 반환

    Args:
        data_file (str): args.data_file -> train_rating.csv

    Returns:
        (user_seq : 2차원 리스트 형태 -> [[유저1의 아이템들][...][유저n의 아이템들]]
         max_item : item수 + 2
         valid_rating_matrix : (item[:-2])로 만든 rating matrix
         test_rating_matrix : (item[:-1])로 만든 rating matrix
         submission_rating_matrix : (item[:])로 만든 rating matrix (submission)
        )
    """    
    rating_df = pd.read_csv(data_file)                    # 파일을 열고
    lines = rating_df.groupby("user")["item"].apply(list) # user별 아이템을 리스트 형태 그룹핑
    user_seq = []
    item_set = set()
    for line in lines:

        items = line
        user_seq.append(items)             # 2차원 리스트로, 유저별 아이템 리스트
        item_set = item_set | set(items)   # 아이템들을 넣어줌
    max_item = max(item_set) 

    num_users = len(lines)   
    num_items = max_item + 2 

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)  # valid_matrix (item[:-2])
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)    # test_matrix  (item[:-1])
    submission_rating_matrix = generate_rating_matrix_submission(                       # submission_matrix (item[:])
        user_seq, num_users, num_items
    )
    return (
        user_seq,                   
        max_item,                   
        valid_rating_matrix,        
        test_rating_matrix,         
        submission_rating_matrix,   
    )


def get_user_seqs_long(data_file):
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        items = line
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
