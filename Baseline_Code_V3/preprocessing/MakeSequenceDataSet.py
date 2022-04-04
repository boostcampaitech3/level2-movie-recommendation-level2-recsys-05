import os
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MakeSequenceDataSet():
    """
    SequenceData 생성
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, 'train_ratings.csv'))

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder('item')
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder('user')
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['item'].apply(lambda x : self.item_encoder[x] + 1) # padding 고려
        self.df['user_idx'] = self.df['user'].apply(lambda x : self.user_encoder[x])
        self.df = self.df.sort_values(['user_idx', 'time']) # 시간에 따라 정렬
        self.user_train, self.user_valid = self.generate_sequence_data()

    def generate_encoder_decoder(self, col : str) -> dict:
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
        for user, item, time in zip(self.df['user_idx'], self.df['item_idx'], self.df['time']):
            users[user].append(item)
        
        for user in users:
            user_train[user] = users[user][:-1]
            user_valid[user] = [users[user][-1]] # 마지막 아이템을 예측

        return user_train, user_valid
    
    def get_train_valid_data(self):
        return self.user_train, self.user_valid


class SASRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        
        user_seq = self.user_train[user]

        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = user_seq[-1]
        idx = self.max_len - 1

        for pos_sample in reversed(user_seq[:-1]):
            seq[idx] = pos_sample
            pos[idx] = nxt
            if nxt != 0: # padding이 아니라면
                neg[idx] = self.random_neg_sampling(user_seq)
            nxt = pos_sample
            idx -= 1
            if idx == -1: break
        
        return seq, pos, neg

    def random_neg_sampling(self, rated_item : list):
        nge_sample = np.random.randint(1, self.num_item + 1)
        while nge_sample in rated_item:
            nge_sample = np.random.randint(1, self.num_item + 1)
        return nge_sample


class BERTRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item, mask_prob):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.mask_prob = mask_prob

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        
        user_seq = self.user_train[user]
        tokens = []
        labels = []
        for s in user_seq[-self.max_len:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    # noise
                    tokens.append(self.random_neg_sampling(user_seq))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s) # 학습에 사용 O
            else:
                tokens.append(s)
                labels.append(0) # 학습에 사용 X

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def random_neg_sampling(self, rated_item : list):
        nge_sample = np.random.randint(1, self.num_item + 1)
        while nge_sample in rated_item:
            nge_sample = np.random.randint(1, self.num_item + 1)
        return nge_sample