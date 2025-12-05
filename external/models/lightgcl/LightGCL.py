from tqdm import tqdm
import numpy as np
import torch
import os
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
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_n_layers", "n_layers", "n_layers", 2, int, None),
            ("_lambda1", "lambda1", "lambda1", 0.2, float, None),
            ("_lambda2", "lambda2", "lambda2", 1e-7, float, None),
            ("_temp", "temp", "temp", 0.2, float, None),
            ("_dropout", "dropout", "dropout", 0.0, float, None),
            ("_q", "q", "q", 5, int, None)  # Rank for SVD
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

        # --- Data Preparation (Mirroring main.py logic) ---
        print("Preparing data and performing SVD for LightGCL...")

        # Get Sparse Matrix
        train_matrix = self._data.sp_i_train.copy()  # scipy csr_matrix

        train_coo = train_matrix.tocoo()
        #self.train_mask = (train_coo != 0).astype(np.float32)  # binary mask for training data

        # Normalize Adjacency Matrix
        rowD = np.array(train_coo.sum(1)).squeeze()
        colD = np.array(train_coo.sum(0)).squeeze()

        for i in range(len(train_coo.data)):
            row = train_coo.row[i]
            col = train_coo.col[i]
            train_coo.data[i] /= pow(rowD[row] * colD[col], 0.5)

        # Construct Data Loader (original code)
        self.train_loader = torch.utils.data.DataLoader(TrnData(train_coo), batch_size=self._batch_size, shuffle=True, num_workers=0)

        # Create Normalized Torch Sparse Tensor
        self.adj_norm = self.scipy_sparse_mat_to_torch_sparse_tensor(train_coo)
        self.adj_norm = self.adj_norm.coalesce().to(self.device)

        # Perform SVD Lowrank
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self._q)
        self.u_mul_s = (svd_u @ (torch.diag(s))).to(self.device)
        self.v_mul_s = (svd_v @ (torch.diag(s))).to(self.device)

        self.ut = svd_u.T.to(self.device)
        self.vt = svd_v.T.to(self.device)

        del s, svd_u, svd_v
        print("SVD computation finished.")

        # --- Initialize Model ---
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

        #random.seed(self._seed)
        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            # The original LightGCL calculates neg_sampling once per epoch in TrnData dataset.
            self.train_loader.dataset.neg_sampling()
            with tqdm(total=int(self._data.transactions / self._batch_size), disable=not self._verbose) as t:
                for _, batch in enumerate(self.train_loader):
                # for batch in next_batch_pairwise(data=self.edge_index,
                #                                  num_items=self._num_items,
                #                                  batch_size=self._batch_size,
                #                                  ui_dict=self.ui_dict,
                #                                  seed=self._seed):
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

        # Set model to eval mode
        self._model.eval()

        # smaller batch size (defined in the original parser)
        eval_batch_size = 256

        # In test phase, we just need the final embeddings E_u and E_i.
        with (torch.no_grad()):
            #self._model.forward(is_training=False)

            for index, offset in enumerate(range(0, self._num_users, eval_batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                uids = torch.arange(offset, offset_stop).long().to(self.device)

                # Predict using dot product of embeddings stored in model
                predictions = self._model.predict(uids)

                ### masking from original code
                # mask = self.train_mask[uids.cpu().numpy()].toarray()
                # mask = torch.Tensor(mask).to(self.device)
                # preds = preds * (1 - mask) - 1e8 * mask
                # predictions = preds.argsort(descending=True)

                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        # Reset to train mode
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