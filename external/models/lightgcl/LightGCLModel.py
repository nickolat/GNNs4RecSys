import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class LightGCLModel(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 n_layers,
                 dropout,
                 temp,
                 lambda_1,
                 lambda_2,
                 u_mul_s,
                 v_mul_s,
                 ut,
                 vt,
                 adj_norm,
                 random_seed,
                 name="LightGCL"):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.dropout = dropout
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # Initialize embeddings
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_users, embed_k)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_items, embed_k)))

        # Input matrices
        self.adj_norm = adj_norm
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        # Storage for user/item embeddings for each layer
        # local embeddings (GNN)
        self.E_u_list = [None] * (n_layers + 1)
        self.E_i_list = [None] * (n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        # embeddings for message passing
        self.Z_u_list = [None] * (n_layers + 1)
        self.Z_i_list = [None] * (n_layers + 1)

        # global embeddings (SVD-based)
        self.G_u_list = [None] * (n_layers + 1)
        self.G_i_list = [None] * (n_layers + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        # aggregation
        self.E_u = None
        self.E_i = None
        self.G_u = None
        self.G_i = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)


    def forward(self, is_training=True):
        for layer in range(1, self.n_layers + 1):
            # GNN propagation
            dropped_adj = self.sparse_dropout(self.adj_norm, self.dropout) if is_training else self.adj_norm

            self.Z_u_list[layer] = torch.spmm(dropped_adj, self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.spmm(dropped_adj.transpose(0, 1), self.E_u_list[layer - 1])

            # SVD_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)

            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # Aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        # Aggregate across layers (sum)
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        return self.E_u, self.E_i

    def train_step(self, batch):
        uids, pos, neg = batch
        uids = uids.clone().detach().long().to(self.device)
        pos = pos.clone().detach().long().to(self.device)
        neg = neg.clone().detach().long().to(self.device)

        iids = torch.concat([pos, neg], dim=0)

        # Forward pass
        self.forward(is_training=True)

        # Loss
        loss_s = self.calc_cl_loss(uids, iids)
        loss_r = self.calc_bpr_loss(uids, pos, neg)
        loss_reg = self.calc_reg_loss(self.lambda_2, self.parameters())
        # Total Loss
        loss = loss_r + self.lambda_1 * loss_s + loss_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def calc_cl_loss(self, uids, iids):
        G_u = self.G_u
        E_u = self.E_u
        G_i = self.G_i
        E_i = self.E_i

        neg_score = torch.log(torch.exp(G_u[uids] @ E_u.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i[iids] @ E_i.T / self.temp).sum(1) + 1e-8).mean()

        pos_score = (torch.clamp((G_u[uids] * E_u[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                    (torch.clamp((G_i[iids] * E_i[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()

        # InfoNCE
        loss_s = -pos_score + neg_score
        return loss_s

    def calc_bpr_loss(self, uids, pos, neg):
        u_emb = self.E_u[uids]
        pos_emb = self.E_i[pos]
        neg_emb = self.E_i[neg]

        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()
        return loss_r

    @staticmethod
    def calc_reg_loss(reg, params):
        loss_reg = 0
        for param in params:
            loss_reg += param.norm(2).square()
        return loss_reg * reg

    def predict(self, uids):
        preds = self.E_u[uids] @ self.E_i.T
        return preds

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
