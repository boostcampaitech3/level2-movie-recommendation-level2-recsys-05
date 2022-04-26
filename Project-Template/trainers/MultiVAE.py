import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
update_count = 1


def train(model, criterion, optimizer, data_loader, make_matrix_data_set, config):
    global update_count
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(device)

        anneal = min(config.anneal_cap, 1.0 * update_count / config.total_anneal_steps)

        recon_mat, mu, logvar = model(mat)

        optimizer.zero_grad()
        loss = criterion(recon_mat, mat, mu, logvar, anneal)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

        update_count += 1

    loss_val /= len(data_loader)

    return loss_val


def get_ndcg(pred_list, true_list):
    ndcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            ndcg += 1 / np.log2(rank + 2)
    return ndcg


# 대회 메트릭인 recall과 동일
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def evaluate(model, data_loader, user_train, user_valid, make_matrix_data_set):
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(device)

            recon_mat, mu, logvar = model(mat)
            recon_mat = recon_mat.softmax(dim=1)
            recon_mat[mat == 1] = -1.0
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-10:].cpu().numpy().tolist()
                NDCG += get_ndcg(pred_list=up, true_list=uv)
                HIT += get_hit(pred_list=up, true_list=uv)

    NDCG /= len(data_loader.dataset)
    HIT /= len(data_loader.dataset)

    return NDCG, HIT


def predict(model, data_loader, user_train, user_valid, make_matrix_data_set):
    model.eval()

    user2rec_list = {}
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users, train=False)
            mat = mat.to(device)

            recon_mat, mu, logvar = model(mat)
            recon_mat = recon_mat.softmax(dim=1)
            recon_mat[mat == 1] = -1.0
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                up = rec[-10:].cpu().numpy().tolist()
                user2rec_list[user.item()] = up

    return user2rec_list


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD
