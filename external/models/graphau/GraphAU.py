from tqdm import tqdm
import numpy as np
import torch
import math

from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .GraphAUModel import GraphAUModel

from torch_sparse import SparseTensor


class GraphAU(RecMixin, BaseRecommenderModel):
    r"""
    Graph-based Alignment and Uniformity for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3583780.3615185>`_

    Args:
        lr: Learning rate
        epochs: Number of maximum epochs
        factors: Number of latent factors
        batch_size: Batch size
        n_layers: Number of stacked propagation layers
        gamma_au: Gamma parameter
        alpha_n: Alpha parameter
        weight_decay: Weight decay

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

        models:
        external.GraphAU:
          meta:
            save_recs: True
          lr: 0.1
          factors: 32
          n_layers: 2
          gamma_au: 1.7
          alpha_n: 0.1
          weight_decay: 0.0
          decaying_base: 1.4
          batch_size: 2048
          epochs: 1000
          normalize: True
          seed: 2022
          early_stopping:
            patience: 10
            mode: auto
            monitor: loss
            verbose: True
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.1, float, None),
            ("_factors", "factors", "factors", 32, int, None),
            ("_n_layers", "n_layers", "n_layers", 4, int, None),
            ("_gamma_au", "gamma_au", "gamma_au", 0.4, float, None),
            ("_decaying_base", "decaying_base", "decaying_base", 1.4, float, None),
            ("_weight_decay", "weight_decay", "weight_decay", 1e-6, float, None),
            ("_alpha_n", "alpha_n", "alpha_n", 0.1, float, None),
            ("_normalize", "normalize", "normalize", True, bool, None)
        ]
        self.autoset_params()

        self._sampler = Sampler(self._data.i_train_dict, seed=self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = GraphAUModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            n_layers=self._n_layers,
            gamma=self._gamma_au,
            decaying_base=self._decaying_base,
            weight_decay=self._weight_decay,
            adj=self.adj,
            normalize=self._normalize,
            random_seed=self._seed
        )
        self.val_data = self.get_val_data()

    @property
    def name(self):
        return "GraphAU" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            self._model.train()
            loss = 0
            steps = 0
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            val_loss = self.validate_loss()
            # print(f"Epoch {it+1} | Val loss: {val_loss:.5f}") ## check
            self.evaluate(it, val_loss)

    def validate_loss(self):
        self._model.eval()
        if len(self.val_data) == 0:
            return 0.0
            
        val_loss = 0
        steps = 0
        batch_size = self._batch_size
        n_val = len(self.val_data)
        
        with torch.no_grad():
            for offset in range(0, n_val, batch_size):
                end = min(offset + batch_size, n_val)
                users_batch = torch.tensor(self.val_data[offset:end, 0]).view(-1, 1).to(self._model.device)
                items_batch = torch.tensor(self.val_data[offset:end, 1]).view(-1, 1).to(self._model.device)
                
                loss = self._model.calculate_loss((users_batch, items_batch))
                val_loss += loss.item()
                steps += 1
        
        return val_loss / steps if steps > 0 else 0.0

    def get_val_data(self):
        val_dict = self._data.get_validation()
        val_data = []
        if val_dict:
            for u, items in val_dict.items():
                if u in self._data.public_users:
                    for i in items.keys():
                        if i in self._data.public_items:
                            val_data.append((self._data.public_users[u], self._data.public_items[i]))
        return np.array(val_data)

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.get_final_embeddings()
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

