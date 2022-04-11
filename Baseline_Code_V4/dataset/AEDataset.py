from dataset.dataset import Dataset

import torch


class AEDataset(Dataset):
    def __init__(self, num_user) -> None:
        super().__init__()
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user = self.users[idx]
        return torch.LongTensor([user])

    def get_train_valid_data(self):
        pass
