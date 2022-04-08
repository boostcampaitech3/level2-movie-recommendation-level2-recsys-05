import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg
    return ndcg

# hit == recall == precision
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def evaluate(model1, model2, RecVAE, AutoRec, MultiDAE, MultiVAE, X, user_train, user_valid, candidate_cnt):
    RecVAE.eval()
    AutoRec.eval()
    MultiDAE.eval()
    MultiVAE.eval()

    mat = torch.from_numpy(X)

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    recon_mat1 = model1.pred.cpu()
    recon_mat1[mat == 1] = -np.inf
    rec_list1 = recon_mat1.argsort(dim = 1)

    recon_mat2 = model2.pred.T.cpu()
    recon_mat2[mat == 1] = -np.inf
    rec_list2 = recon_mat2.argsort(dim = 1)

    recon_mat3 = RecVAE(mat.to(device), calculate_loss = False).cpu().detach()
    recon_mat3[mat == 1] = -np.inf
    rec_list3 = recon_mat3.argsort(dim = 1)

    recon_mat4 = AutoRec(mat.to(device)).cpu().detach()
    recon_mat4[mat == 1] = -np.inf
    rec_list4 = recon_mat4.argsort(dim = 1)

    recon_mat5 = MultiDAE(mat.to(device)).cpu().detach()
    recon_mat5[mat == 1] = -np.inf
    rec_list5 = recon_mat5.argsort(dim = 1)

    recon_mat6, mu, logvar = MultiVAE(mat.to(device))
    recon_mat6 = recon_mat6.cpu().detach()
    recon_mat6[mat == 1] = -np.inf
    rec_list6 = recon_mat6.argsort(dim = 1)

    score_li = np.array([1/np.log2(rank + 2) for rank in range(0, candidate_cnt)])

    for user, (rec1, rec2, rec3, rec4, rec5, rec6) in tqdm(enumerate(zip(rec_list1, rec_list2, rec_list3, rec_list4, rec_list5, rec_list6))):
        uv = user_valid[user]

        # ranking
        rec1 = rec1[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec2 = rec2[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec3 = rec3[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec4 = rec4[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec5 = rec5[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec6 = rec6[-candidate_cnt:].cpu().numpy().tolist()[::-1]

        items = list(set(rec1 + rec2 + rec3 + rec4 + rec5 + rec6))

        movie_df = pd.DataFrame(index = items)
        movie_df.loc[rec1, 'rec1_score'] = score_li * 0.25
        movie_df.loc[rec2, 'rec2_score'] = score_li * 0.25
        movie_df.loc[rec3, 'rec3_score'] = score_li * 0.2
        movie_df.loc[rec4, 'rec4_score'] = score_li * 0.1
        movie_df.loc[rec5, 'rec5_score'] = score_li * 0.1
        movie_df.loc[rec6, 'rec6_score'] = score_li * 0.1
        movie_df = movie_df.fillna(min(score_li) * 0.1)
        movie_df['total_score'] = movie_df['rec1_score'] + movie_df['rec2_score'] + movie_df['rec3_score'] + movie_df['rec4_score'] + movie_df['rec5_score'] + movie_df['rec6_score']
        movie_df = movie_df.sort_values('total_score', ascending = False)
        up = movie_df.index.tolist()[:10]

        NDCG += get_ndcg(pred_list = up, true_list = uv)
        HIT += get_hit(pred_list = up, true_list = uv)

    NDCG /= len(user_train)
    HIT /= len(user_train)

    return NDCG, HIT

def predict(model1, model2, RecVAE, AutoRec, MultiDAE, MultiVAE, X, candidate_cnt):
    user2rec = {}

    RecVAE.eval()
    AutoRec.eval()
    MultiDAE.eval()
    MultiVAE.eval()

    mat = torch.from_numpy(X)

    recon_mat1 = model1.pred.cpu()
    recon_mat1[mat == 1] = -np.inf
    rec_list1 = recon_mat1.argsort(dim = 1)

    recon_mat2 = model2.pred.T.cpu()
    recon_mat2[mat == 1] = -np.inf
    rec_list2 = recon_mat2.argsort(dim = 1)

    recon_mat3 = RecVAE(mat.to(device), calculate_loss = False).cpu().detach()
    recon_mat3[mat == 1] = -np.inf
    rec_list3 = recon_mat3.argsort(dim = 1)

    recon_mat4 = AutoRec(mat.to(device)).cpu().detach()
    recon_mat4[mat == 1] = -np.inf
    rec_list4 = recon_mat4.argsort(dim = 1)

    recon_mat5 = MultiDAE(mat.to(device)).cpu().detach()
    recon_mat5[mat == 1] = -np.inf
    rec_list5 = recon_mat5.argsort(dim = 1)

    recon_mat6, mu, logvar = MultiVAE(mat.to(device))
    recon_mat6 = recon_mat6.cpu().detach()
    recon_mat6[mat == 1] = -np.inf
    rec_list6 = recon_mat6.argsort(dim = 1)

    score_li = np.array([1/np.log2(rank + 2) for rank in range(0, candidate_cnt)])

    for user, (rec1, rec2, rec3, rec4, rec5, rec6) in tqdm(enumerate(zip(rec_list1, rec_list2, rec_list3, rec_list4, rec_list5, rec_list6))):
        
        # ranking
        rec1 = rec1[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec2 = rec2[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec3 = rec3[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec4 = rec4[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec5 = rec5[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec6 = rec6[-candidate_cnt:].cpu().numpy().tolist()[::-1]

        items = list(set(rec1 + rec2 + rec3 + rec4 + rec5 + rec6))

        movie_df = pd.DataFrame(index = items)
        movie_df.loc[rec1, 'rec1_score'] = score_li * 0.25
        movie_df.loc[rec2, 'rec2_score'] = score_li * 0.25
        movie_df.loc[rec3, 'rec3_score'] = score_li * 0.2
        movie_df.loc[rec4, 'rec4_score'] = score_li * 0.1
        movie_df.loc[rec5, 'rec5_score'] = score_li * 0.1
        movie_df.loc[rec6, 'rec6_score'] = score_li * 0.1
        movie_df = movie_df.fillna(min(score_li) * 0.1)
        movie_df['total_score'] = movie_df['rec1_score'] + movie_df['rec2_score'] + movie_df['rec3_score'] + movie_df['rec4_score'] + movie_df['rec5_score'] + movie_df['rec6_score']
        movie_df = movie_df.sort_values('total_score', ascending = False)
        up = movie_df.index.tolist()[:10]

        user2rec[user] = up

    return user2rec


def total_evaluate(model1, model2, RecVAE, AutoRec, MultiDAE, MultiVAE, X, user_train, user_valid, candidate_cnt):
    RecVAE.eval()
    AutoRec.eval()
    MultiDAE.eval()
    MultiVAE.eval()

    df = []

    mat = torch.from_numpy(X)

    recon_mat1 = model1.pred.cpu()
    recon_mat1[mat == 1] = -np.inf
    rec_list1 = recon_mat1.argsort(dim = 1)

    recon_mat2 = model2.pred.T.cpu()
    recon_mat2[mat == 1] = -np.inf
    rec_list2 = recon_mat2.argsort(dim = 1)

    recon_mat3 = RecVAE(mat.to(device), calculate_loss = False).cpu().detach()
    recon_mat3[mat == 1] = -np.inf
    rec_list3 = recon_mat3.argsort(dim = 1)

    recon_mat4 = AutoRec(mat.to(device)).cpu().detach()
    recon_mat4[mat == 1] = -np.inf
    rec_list4 = recon_mat4.argsort(dim = 1)

    recon_mat5 = MultiDAE(mat.to(device)).cpu().detach()
    recon_mat5[mat == 1] = -np.inf
    rec_list5 = recon_mat5.argsort(dim = 1)

    recon_mat6, mu, logvar = MultiVAE(mat.to(device))
    recon_mat6 = recon_mat6.cpu().detach()
    recon_mat6[mat == 1] = -np.inf
    rec_list6 = recon_mat6.argsort(dim = 1)

    for user, (rec1, rec2, rec3, rec4, rec5, rec6) in tqdm(enumerate(zip(rec_list1, rec_list2, rec_list3, rec_list4, rec_list5, rec_list6))):
        uv = user_valid[user]

        # ranking
        rec1 = rec1[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec2 = rec2[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec3 = rec3[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec4 = rec4[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec5 = rec5[-candidate_cnt:].cpu().numpy().tolist()[::-1]
        rec6 = rec6[-candidate_cnt:].cpu().numpy().tolist()[::-1]

        rec123456 = list(set(rec1 + rec2 + rec3 + rec4 + rec5 + rec6))

        df.append(
            {
               'user' : user,
               'len' : len(rec123456),

               'rec1' : get_hit(pred_list = rec1, true_list = uv),
               'rec2' : get_hit(pred_list = rec2, true_list = uv),
               'rec3' : get_hit(pred_list = rec3, true_list = uv),
               'rec4' : get_hit(pred_list = rec4, true_list = uv),
               'rec5' : get_hit(pred_list = rec5, true_list = uv),
               'rec6' : get_hit(pred_list = rec6, true_list = uv),

               'rec123456' : get_hit(pred_list = rec123456, true_list = uv),
            }
        )

    return df