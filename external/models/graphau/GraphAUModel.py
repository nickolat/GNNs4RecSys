from abc import ABC

from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random
import torch.nn.functional as F


class GraphAUModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 n_layers,
                 gamma,
                 decaying_base,
                 weight_decay,
                 adj,
                 normalize,
                 random_seed,
                 name="GraphAU",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.gamma = gamma
        self.decay_base = decaying_base
        self.weight_decay = weight_decay
        self.adj = adj
        self.normalize = normalize

        # Initialize embeddings
        self.Gu = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embed_k)
        self.Gi = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embed_k)
        torch.nn.init.normal_(self.Gu.weight, std=0.1)
        torch.nn.init.normal_(self.Gi.weight, std=0.1)

        propagation_network_list = []

        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=self.normalize), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Weights decay
        ls = [1.0]
        for l in range(self.n_layers - 1): # If n_layers=2, range(1), appends once. ls=[1, decay]
            ls.append(ls[-1] * self.decay_base)
        self.decay_weight = torch.tensor(ls).to(self.device)

    def propagate_embeddings(self):
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            layer_embeddings = list(self.propagation_network.children())[layer](
                all_embeddings[layer].to(self.device),
                self.adj.to(self.device)
            )
            all_embeddings.append(layer_embeddings)
            
        return all_embeddings

    def forward(self, gu, gi):
        return torch.sum(gu * gi, dim=1)

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, batch):
        user, positive = batch
        user = user[:, 0]
        positive = positive[:, 0]

        all_embs = self.propagate_embeddings()
        
        # Layer 0 embeddings
        h0 = all_embs[0]
        gu0, gi0 = torch.split(h0, [self.num_users, self.num_items], 0)
        
        user_e = gu0[user]
        item_e = gi0[positive]

        # L0 Alignment
        align_list = [self.alignment(user_e, item_e)]
        
        # L0 Uniformity
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2


        for k in range(1, self.n_layers): # range(1, 2) -> k=1.
            hk = all_embs[k] # Layer k embeddings
            guk, gik = torch.split(hk, [self.num_users, self.num_items], 0)
            
            user_e_agg = guk[user]
            item_e_agg = gik[positive]

            a = (self.alignment(user_e, item_e_agg) + self.alignment(user_e_agg, item_e)) / 2
            align_list.append(a)
        
        # Weighted mean
        align_tensor = torch.stack(align_list)
        align_val = torch.mean(self.decay_weight * align_tensor)

        loss = align_val + self.gamma * uniform
        return loss

    def train_step(self, batch):
        loss = self.calculate_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

    def get_final_embeddings(self):
        return self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)
