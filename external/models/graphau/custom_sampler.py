import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, seed=2022):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, events: int, batch_size: int):
        n_users = self._nusers
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        
        # Randomly select users for the batch
        users = np.random.randint(0, n_users, events)

        def sample(current):
            u = users[current]
            ui = sorted(ui_dict[u])
            lui = lui_dict[u]
            if lui == 0: # Handle edge case of user with no items equivalent to infinite loop check
                 # Pick another user
                u = np.random.randint(0, n_users)
                return sample(current) 
            
            posindex = np.random.randint(0, len(ui))
            i = ui[posindex]
            
            return u, i

        bui, bii = [], []
        for idx in range(events):
            u, p = sample(idx)
            bui.append(u)
            bii.append(p)

        bui = np.array(bui)
        bii = np.array(bii)

        shuffle_indices = np.arange(bui.shape[0])
        np.random.shuffle(shuffle_indices)

        bui = bui[shuffle_indices]
        bii = bii[shuffle_indices]

        for batch_start in range(0, events, batch_size):
            buii, biii = map(np.array, zip(*[(bui[idx], bii[idx]) for idx in range(batch_start, min(batch_start + batch_size, events))]))
            yield buii[:, None], biii[:, None]
