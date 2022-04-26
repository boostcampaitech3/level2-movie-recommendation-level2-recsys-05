import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, criterion, optimizer, data_loader):
    model.train()
    loss_val = 0
    for seq, labels in data_loader:
        logits = model(seq)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).to(device)

        optimizer.zero_grad()
        loss = criterion(logits, labels)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def evaluate(model, user_train, user_valid, max_len, dataset, make_sequence_dataset):
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    num_item_sample = 100
    num_user_sample = 1000

    users = np.random.randint(
        0, make_sequence_dataset.num_user, num_user_sample
    )  # 1000개만 sampling 하여 evaluation
    for user in users:
        seq = (user_train[user] + [make_sequence_dataset.num_item + 1])[-max_len:]
        rated = set(user_train[user] + user_valid[user])
        item_idx = user_valid[user] + [
            dataset.random_neg_sampling(rated) for _ in range(num_item_sample)
        ]

        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][item_idx]  # sampling
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10:  # @10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    NDCG /= num_user_sample
    HIT /= num_user_sample

    return NDCG, HIT
