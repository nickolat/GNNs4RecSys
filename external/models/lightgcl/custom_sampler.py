# import random
# from random import shuffle, choice
# from copy import deepcopy

import numpy as np
import torch.utils.data as data


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        #np.random.seed(42) #
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


# def next_batch_pairwise(data, batch_size, ui_dict, num_items, n_negs=1, seed=42):
#     random.seed(seed)
#     training_data = deepcopy(data)
#     shuffle(training_data)
#     ptr = 0
#     data_size = len(training_data)
#     while ptr < data_size:
#         if ptr + batch_size < data_size:
#             batch_end = ptr + batch_size
#         else:
#             batch_end = data_size
#         users = [training_data[idx][0] for idx in range(ptr, batch_end)]
#         items = [training_data[idx][1] for idx in range(ptr, batch_end)]
#         ptr = batch_end
#         u_idx, i_idx, j_idx = [], [], []
#         item_list = list(range(num_items))
#         for i, user in enumerate(users):
#             i_idx.append(items[i])
#             u_idx.append(user)
#             for m in range(n_negs):
#                 neg_item = choice(item_list)
#                 while neg_item in ui_dict[user]:
#                     neg_item = choice(item_list)
#                 j_idx.append(neg_item)
#         yield u_idx, i_idx, j_idx