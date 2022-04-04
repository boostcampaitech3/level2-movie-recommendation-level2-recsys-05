import numpy as np
import torch
from .metric import get_ndcg, get_hit

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, data_loader, make_data_set):
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_data_set.make_matrix(users)
        mat = mat.to(device)
        loss = model(mat)

        optimizer.zero_grad()

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def evaluate(model, data_loader, user_valid, make_data_set):
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    with torch.no_grad():
        for users in data_loader:
            mat = make_data_set.make_matrix(users)
            mat = mat.to(device)

            recon_mat = model(mat, calculate_loss=False)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-10:].cpu().numpy().tolist()
                NDCG += get_ndcg(pred_list=up, true_list=uv)
                HIT += get_hit(pred_list=up, true_list=uv)

    NDCG /= len(data_loader.dataset)
    HIT /= len(data_loader.dataset)

    return NDCG, HIT


def predict(model, data_loader, make_data_set):
    model.eval()

    user2rec_list = {}
    with torch.no_grad():
        for users in data_loader:
            mat = make_data_set.make_matrix(users, train=False)
            mat = mat.to(device)

            recon_mat = model(mat, calculate_loss=False)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                up = rec[-10:].cpu().numpy().tolist()
                user2rec_list[user.item()] = up

    return user2rec_list
