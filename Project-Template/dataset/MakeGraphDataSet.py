import os
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import scipy.sparse as sp


class MakeGraphDataSet:
    """
    GraphDataSet 생성
    """

    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, "train_ratings.csv"))

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder("item")
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder("user")
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df["item_idx"] = self.df["item"].apply(lambda x: self.item_encoder[x])
        self.df["user_idx"] = self.df["user"].apply(lambda x: self.user_encoder[x])

        self.exist_users = [i for i in range(self.num_user)]
        self.exist_items = [i for i in range(self.num_item)]
        self.user_train, self.user_valid = self.generate_sequence_data()
        self.R_train, self.R_valid, self.R_total = self.generate_dok_matrix()
        self.ngcf_adj_matrix = self.generate_ngcf_adj_matrix()
        self.n_train = len(self.R_train)
        self.batch_size = self.config.batch_size

    def generate_encoder_decoder(self, col: str) -> dict:
        """
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item, time in zip(
            self.df["user_idx"], self.df["item_idx"], self.df["time"]
        ):
            users[user].append(item)

        for user in users:
            np.random.seed(self.config.seed)

            user_total = users[user]
            valid = np.random.choice(
                user_total, size=self.config.valid_samples, replace=False
            ).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid  # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

        return user_train, user_valid

    def generate_dok_matrix(self):
        R_train = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)
        R_valid = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)
        R_total = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)
        user_list = self.exist_users
        for user in user_list:
            train_items = self.user_train[user]
            valid_items = self.user_valid[user]

            for train_item in train_items:
                R_train[user, train_item] = 1.0
                R_total[user, train_item] = 1.0

            for valid_item in valid_items:
                R_valid[user, valid_item] = 1.0
                R_total[user, valid_item] = 1.0

        return R_train, R_valid, R_total

    def generate_ngcf_adj_matrix(self):
        adj_mat = sp.dok_matrix(
            (self.num_user + self.num_item, self.num_user + self.num_item),
            dtype=np.float32,
        )
        adj_mat = adj_mat.tolil()  # to_list
        R = self.R_train.tolil()

        adj_mat[: self.num_user, self.num_user :] = R
        adj_mat[self.num_user :, : self.num_user] = R.T
        adj_mat = adj_mat.todok()  # to_dok_matrix

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)

            return norm_adj.tocoo()

        ngcf_adj_matrix = normalized_adj_single(adj_mat)
        return ngcf_adj_matrix.tocsr()

    def sampling(self):
        users = random.sample(self.exist_users, self.config.batch_size)

        def sample_pos_items_for_u(u, num):
            pos_items = self.user_train[u]
            pos_batch = random.sample(pos_items, num)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = list(set(self.exist_items) - set(self.user_train[u]))
            neg_batch = random.sample(neg_items, num)
            return neg_batch

        pos_items, neg_items = [], []
        for user in users:
            pos_items += sample_pos_items_for_u(user, 1)
            neg_items += sample_neg_items_for_u(user, 1)

        return users, pos_items, neg_items

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def get_R_data(self):
        return self.R_train, self.R_valid, self.R_total

    def get_ngcf_adj_matrix_data(self):
        return self.ngcf_adj_matrix
