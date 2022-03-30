import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

device = "cuda" if torch.cuda.is_available() else "cpu"


class NGCF(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        emb_dim,
        layers,
        reg,
        node_dropout,
        mess_dropout,
        adj_mtx,
    ):
        super().__init__()

        # initialize Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.l_matrix = adj_mtx
        self.l_plus_i_matrix = adj_mtx + sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Weights initialized.")

        # Create Matrix 'L+I', PyTorch sparse tensor of SP adjacency_mtx
        self.L_plus_I = self._convert_sp_mat_to_sp_tensor(self.l_plus_i_matrix)
        self.L = self._convert_sp_mat_to_sp_tensor(self.l_matrix)

    # initialize weights
    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_

        weight_dict["user_embedding"] = nn.Parameter(
            initializer(torch.empty(self.n_users, self.emb_dim).to(device))
        )
        weight_dict["item_embedding"] = nn.Parameter(
            initializer(torch.empty(self.n_items, self.emb_dim).to(device))
        )

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict["W_one_%d" % k] = nn.Parameter(
                initializer(
                    torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(device)
                )
            )
            weight_dict["b_one_%d" % k] = nn.Parameter(
                initializer(torch.empty(1, weight_size_list[k + 1]).to(device))
            )

            weight_dict["W_two_%d" % k] = nn.Parameter(
                initializer(
                    torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(device)
                )
            )
            weight_dict["b_two_%d" % k] = nn.Parameter(
                initializer(torch.empty(1, weight_size_list[k + 1]).to(device))
            )

        return weight_dict

    # convert sparse matrix into sparse PyTorch tensor
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

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X

        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """
        node_dropout_mask = (
            ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(device)
        )
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:, node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return X_dropout.mul(1 / (1 - self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass

        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        # apply drop-out mask
        L_plus_I_hat = (
            self._droupout_sparse(self.L_plus_I)
            if self.node_dropout > 0
            else self.L_plus_I
        )
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        # 논문 수식 (1)
        ego_embeddings = torch.cat(
            [self.weight_dict["user_embedding"], self.weight_dict["item_embedding"]], 0
        )

        final_embeddings = [ego_embeddings]

        # forward pass for 'n' propagation layers
        for k in range(self.n_layers):

            # ---------- Fill below -----------
            # -- 논문 수식 (7) --

            # (L+I)E
            side_L_plus_I_embeddings = torch.sparse.mm(
                L_plus_I_hat, final_embeddings[k]
            )  # 힌트 : use torch.sparse.mm

            # (L+I)EW_1 + b_1
            simple_embeddings = (
                torch.matmul(side_L_plus_I_embeddings, self.weight_dict["W_one_%d" % k])
                + self.weight_dict["b_one_%d" % k]
            )  # 힌트 : use torch.matmul, self.weight_dict['W_one_%d' % k], self.weight_dict['b_one_%d' % k]

            # LE
            side_L_embeddings = torch.sparse.mm(
                L_hat, final_embeddings[k]
            )  # 힌트 : use torch.sparse.mm

            # LEE
            interaction_embeddings = torch.mul(
                side_L_embeddings, final_embeddings[k]
            )  # 힌트 : use torch.mul

            # LEEW_2 + b_2
            interaction_embeddings = (
                torch.matmul(interaction_embeddings, self.weight_dict["W_two_%d" % k])
                + self.weight_dict["b_two_%d" % k]
            )  # 힌트 : use torch.matmul, self.weight_dict['W_two_%d' % k], self.weight_dict['b_two_%d' % k]

            # non-linear activation
            ego_embeddings = F.leaky_relu(
                simple_embeddings + interaction_embeddings
            )  # 힌트: use simple_embeddings, interaction_embeddings

            # ---------- Fill above -----------

            # add message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # Perform L2 normalization
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            # -- 논문 수식 (9) --
            final_embeddings.append(norm_embeddings)

        final_embeddings = torch.cat(final_embeddings, 1)

        # back to user/item dimension
        u_final_embeddings, i_final_embeddings = final_embeddings.split(
            [self.n_users, self.n_items], 0
        )

        self.u_final_embeddings = nn.Parameter(u_final_embeddings)
        self.i_final_embeddings = nn.Parameter(i_final_embeddings)

        u_emb = u_final_embeddings[u]  # user embeddings
        p_emb = i_final_embeddings[i]  # positive item embeddings
        n_emb = i_final_embeddings[j]  # negative item embeddings

        # -------- Fill below ---------
        # -- 논문 수식 (10) --

        y_ui = torch.sum(
            torch.mul(u_emb, p_emb), dim=1
        )  # 힌트 : use torch.mul, sum() method
        y_uj = torch.sum(
            torch.mul(u_emb, n_emb), dim=1
        )  # 힌트 : use torch.mul, sum() method

        # -------- Fill above --------

        # -------- Fill below ---------
        # -- 논문 수식 (11) --

        log_prob = torch.mean(
            torch.log(torch.sigmoid(y_ui - y_uj))
        )  # 힌트 : use torch.log, torch.sigmoid, mean() method
        bpr_loss = -log_prob
        if self.reg > 0.0:
            l2norm = (
                torch.sum(u_emb**2) / 2.0
                + torch.sum(p_emb**2) / 2.0
                + torch.sum(n_emb**2) / 2.0
            ) / u_emb.shape[0]
            l2reg = self.reg * l2norm  # FILL HERE #
            bpr_loss += l2reg

        # ---------- Fill above ----------

        return bpr_loss
