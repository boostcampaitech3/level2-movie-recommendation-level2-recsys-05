import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, criterion, optimizer, data_loader):
    model.train()
    loss_val = 0
    for seq, pos, neg in data_loader:
        pos_logits, neg_logits = model(
            seq.cpu().numpy(), pos.cpu().numpy(), neg.cpu().numpy()
        )
        pos_labels, neg_labels = torch.ones(
            pos_logits.shape, device=device
        ), torch.zeros(neg_logits.shape, device=device)

        optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def evaluate(
    model, user_train, user_valid, max_len, sasrec_dataset, make_sequence_dataset
):
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    num_item_sample = 100
    num_user_sample = 1000

    users = np.random.randint(
        0, make_sequence_dataset.num_user, num_user_sample
    )  # 1000개만 sampling 하여 evaluation
    for user in users:
        seq = user_train[user][-max_len:]
        rated = set(user_train[user] + user_valid[user])
        item_idx = user_valid[user] + [
            sasrec_dataset.random_neg_sampling(rated) for _ in range(num_item_sample)
        ]
        with torch.no_grad():
            predictions = -model.predict(np.array([seq]), np.array(item_idx))
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10:  # @10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    NDCG /= num_user_sample
    HIT /= num_user_sample

    return NDCG, HIT
