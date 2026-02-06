from tqdm import tqdm
import numpy as np
import torch
import scipy.sparse as sp

from .custom_sampler import TrnData
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .LightGCLModel import LightGCLModel
import math
import random


class LightGCL(RecMixin, BaseRecommenderModel):
    r"""
    LightGCL: Simple Graph Contrastive Learning for Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/2302.08191>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size for training (inter_batch)
        n_layers: Number of stacked propagation layers
        lambda1: Regularization weight for CL loss
        lambda2: L2 regularization weight
        temp: Temperature parameter
        q: Rank for SVD
        dropout: Dropout rate

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        external.LightGCL:
          meta:
            save_recs: True
          lr: 1e-3
          epochs: 100
          factors: 64
          batch_size: 4096
          lambda1: 0.2
          lambda2: 0.0
          n_layers: 2
          temp: 0.2
          dropout: 0.0
          q: 5
          seed: 42

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_n_layers", "n_layers", "n_layers", 2, int, None),
            ("_lambda1", "lambda1", "lambda1", 0.2, float, None),
            ("_lambda2", "lambda2", "lambda2", 1e-7, float, None),
            ("_temp", "temp", "temp", 0.2, float, None),
            ("_dropout", "dropout", "dropout", 0.0, float, None),
            ("_q", "q", "q", 5, int, None)
        ]
        self.autoset_params()

        # set seed
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


        self.ui_dict = {u: list(set(self._data.i_train_dict[u])) for u in self._data.i_train_dict}
        self.edge_index = self._data.edge_index.to_numpy().tolist()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get sparse matrix
        train_matrix = self._data.sp_i_train.copy()  # scipy csr_matrix

        train_coo = train_matrix.tocoo()

        # Normalize adjacency matrix
        rowD = np.array(train_coo.sum(1)).squeeze()
        colD = np.array(train_coo.sum(0)).squeeze()

        for i in range(len(train_coo.data)):
            row = train_coo.row[i]
            col = train_coo.col[i]
            train_coo.data[i] /= pow(rowD[row] * colD[col], 0.5)

        # Data loader original
        self.train_loader = torch.utils.data.DataLoader(TrnData(train_coo), batch_size=self._batch_size, shuffle=True, num_workers=0)

        # Normalized torch sparse tensor
        self.adj_norm = self.scipy_sparse_mat_to_torch_sparse_tensor(train_coo)
        self.adj_norm = self.adj_norm.coalesce().to(self.device)

        # SVD lowrank
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self._q)
        self.u_mul_s = (svd_u @ (torch.diag(s))).to(self.device)
        self.v_mul_s = (svd_v @ (torch.diag(s))).to(self.device)

        self.ut = svd_u.T.to(self.device)
        self.vt = svd_v.T.to(self.device)

        del s, svd_u, svd_v

        self._model = LightGCLModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            n_layers=self._n_layers,
            dropout=self._dropout,
            temp=self._temp,
            lambda_1=self._lambda1,
            lambda_2=self._lambda2,
            u_mul_s=self.u_mul_s,
            v_mul_s=self.v_mul_s,
            ut=self.ut,
            vt=self.vt,
            adj_norm=self.adj_norm,
            random_seed=self._seed
        )

        self._model.to(self.device)

    @property
    def name(self):
        return "LightGCL" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            self.train_loader.dataset.neg_sampling()
            with tqdm(total=int(self._data.transactions / self._batch_size), disable=not self._verbose) as t:
                for _, batch in enumerate(self.train_loader):
                    steps += 1
                    current_loss = self._model.train_step(batch)
                    loss += current_loss

                    if math.isnan(loss) or math.isinf(loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        self._model.eval()
        eval_batch_size = 256  # defined in the original parser

        with (torch.no_grad()):
            for index, offset in enumerate(range(0, self._num_users, eval_batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                uids = torch.arange(offset, offset_stop).long().to(self.device)

                predictions = self._model.predict(uids)

                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        self._model.train()
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @staticmethod
    def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)