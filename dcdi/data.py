import os
import numpy as np
import csv
import torch


class DataManagerFile(object):
    def __init__(self, file_path, i_dataset, train_samples=0.8, test_samples=None, train=True, normalize=False,
                 mean=None, std=None, random_seed=42, intervention=False,
                 intervention_knowledge="known", dcd=False):
        """
        Parameters:
        -----------
        file_path: str
            Path to the data and the DAG
        i_dataset: uint
            Exemplar to use (usually 1-10)
        train_samples: uint or float, default=0.8
            If float, specifies the proportion of data used for training and the rest is used for testing. If an
            integer, specifies the exact number of examples to use for training.
        test_samples: uint, default=None
            Specifies the number of examples to use for testing. The default value uses all examples that are not used
            for training.
        random_seed: uint
            Random seed to use for data set shuffling and splitting
        intervention: boolean
            If True, use interventional data with interventional targets
        intervention_knowledge: str
            Determine if the interventional target are known or unknown
        dcd: boolean
            If True, use the baseline DCD that use interventional data, but
            with a loss that doesn't take it into account (intervention should
            be set to False)
        """
        self.random = np.random.RandomState(random_seed)
        self.dcd = dcd
        self.file_path = file_path
        self.i_dataset = i_dataset
        self.intervention = intervention
        self.regime_idx = {}
        if intervention_knowledge == "known":
            self.interv_known = True
        elif intervention_knowledge == "unknown":
            self.interv_known = False
        else:
            raise ValueError("intervention_knowledge should either be 'known' \
                             or 'unknown'")

        data, masks, regimes = self.load_data()

        # Determine train/test partitioning
        if isinstance(train_samples, float):
            train_samples = int(data.shape[0] * train_samples)
        if test_samples is None:
            test_samples = data.shape[0] - train_samples
        assert train_samples + test_samples <= data.shape[0], "The number of examples to load must be smaller than " + \
            "the total size of the dataset"

        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]
        if intervention:
            masks = [masks[i] for i in shuffle_idx[: train_samples + test_samples]]

        # Train/test split
        if not train:
            if train_samples == data.shape[0]: # i.e. no test set
                self.dataset = None
                self.masks = None
            else:
                self.dataset = torch.as_tensor(data[train_samples: train_samples + test_samples]).type(torch.Tensor)
                if intervention:
                    self.masks = masks[train_samples: train_samples + test_samples]
        else:
            self.dataset = torch.as_tensor(data[: train_samples]).type(torch.Tensor)
            if intervention:
                self.masks = masks[: train_samples]

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if self.mean is None or self.std is None:
                self.mean = torch.mean(self.dataset, 0, keepdim=True)
                self.std = torch.std(self.dataset, 0, keepdim=True)
            self.dataset = (self.dataset - self.mean) / self.std

        self.num_samples = self.dataset.size(0)
        self.dim = self.dataset.size(1)


    def load_data(self):
        # Load the graph
        adjacency = np.load(os.path.join(self.file_path, f"DAG{self.i_dataset}.npy"))
        self.adjacency = torch.as_tensor(adjacency).type(torch.Tensor)
        if self.dcd:
            assert not self.intervention, "DCD must be used with intervention==False"
            name_data = f"data_interv{self.i_dataset}.npy"
        else:
            name_data = f"data{self.i_dataset}.npy"

        # Load intervention mask and set regime index
        masks = []
        regimes = []
        n_regime = 0
        if self.intervention:
            name_data = f"data_interv{self.i_dataset}.npy"
            interv_path = os.path.join(self.file_path, f"intervention{self.i_dataset}.csv")
            tmp_interv = []

            with open(interv_path, 'r') as f:
                interventions_csv = csv.reader(f)
                for row in interventions_csv:
                    mask = [int(x) for x in row]
                    masks.append(mask)
                    if str(mask) not in self.regime_idx:
                        self.regime_idx[str(mask)] = n_regime
                        tmp_interv_vec = np.zeros(self.adjacency.shape[0])
                        for i in range(self.adjacency.shape[0]):
                            if i in mask:
                                tmp_interv_vec[i] = 1
                        tmp_interv.append(tmp_interv_vec)
                        n_regime += 1
                    regimes.append(self.regime_idx[str(mask)])
            self.nb_interv = len(self.regime_idx)
            self.gt_interv = np.array(tmp_interv).T
        else:
            self.nb_interv = 1
            self.gt_interv = None

        # Load data
        self.data_path = os.path.join(self.file_path, name_data)
        data = np.load(self.data_path)

        return data, masks, regimes


    def convert_masks(self, idxs):
        """Convert mask index to mask vectors"""
        masks_list = [self.masks[i] for i in idxs]

        if self.interv_known:
            masks = torch.ones((idxs.shape[0], self.dim))
            for i, m in enumerate(masks_list):
                for j in m:
                    masks[i, j] = 0
        else:
            masks = torch.zeros((idxs.shape[0], self.dim))

        return masks


    def get_regime(self, idxs):
        """Return the regime for each index"""
        masks_list = [self.masks[i] for i in idxs]
        regime = torch.zeros(len(masks_list))

        for i, m in enumerate(masks_list):
            j = self.regime_idx[str(m)]
            regime[i] = j

        return regime


    def sample(self, batch_size):
        sample_idxs = self.random.choice(np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False)
        samples = self.dataset[torch.as_tensor(sample_idxs).long()]
        if self.intervention:
            masks = self.convert_masks(sample_idxs)
            regimes = self.get_regime(sample_idxs)
        else:
            masks = torch.ones_like(samples)
            regimes = torch.zeros(masks.size(0))
        return samples, masks, regimes
