import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class MakeMatrixDataSet:
    """
    MatrixDataSet 생성
    """

    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, "train_ratings.csv"))

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder("item")
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder("user")
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df["item_idx"] = self.df["item"].apply(lambda x: self.item_encoder[x])
        self.df["user_idx"] = self.df["user"].apply(lambda x: self.user_encoder[x])

        self.user_train, self.user_valid = self.generate_sequence_data()

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

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def make_matrix(self, user_list, train=True):
        """
        user_item_dict를 바탕으로 행렬 생성
        """
        mat = torch.zeros(size=(user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, self.user_train[user.item()]] = 1
            else:
                mat[
                    idx, self.user_train[user.item()] + self.user_valid[user.item()]
                ] = 1
        return mat


class AEDataSet(Dataset):
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user = self.users[idx]
        return torch.LongTensor([user])
