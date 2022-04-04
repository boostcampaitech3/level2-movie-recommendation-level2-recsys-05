import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class EASE:
    def __init__(self, X, reg):
        self.X = self._convert_sp_mat_to_sp_tensor(X)
        self.reg = reg

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res

    def fit(self):
        """

        진짜 정말 간단한 식으로 모델을 만듬

        """
        G = self.X.to_dense().t() @ self.X.to_dense()
        diagIndices = torch.eye(G.shape[0]) == 1
        G[diagIndices] += self.reg

        P = G.inverse()
        B = P / (-1 * P.diag())
        B[diagIndices] = 0

        self.pred = self.X.to_dense() @ B
