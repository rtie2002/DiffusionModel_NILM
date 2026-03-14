import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=2024,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        # Create indices instead of materializing all windows
        indices = np.arange(self.sample_num_total)
        
        # Divide indices into train/test
        train_indices, test_indices = self.divide_indices(indices, proportion, seed)
        
        if self.save2npy:
            print(f"  [Info] save2npy is enabled. Saving base sequence and indices to save space.")
            # Saving 1M overlapping windows is redundant (takes 512x more space).
            # We save the base sequence and the indices instead.
            np.save(os.path.join(self.dir, f"{self.name}_base_data_norm.npy"), self.data)
            np.save(os.path.join(self.dir, f"{self.name}_indices_{self.period}.npy"), train_indices if self.period == 'train' else test_indices)

        return train_indices, test_indices

    @staticmethod
    def divide_indices(indices, ratio, seed=2024):
        size = len(indices)
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        id_rdm = np.random.permutation(size)
        train_num = int(np.ceil(size * ratio))
        
        train_id = indices[id_rdm[:train_num]]
        test_id = indices[id_rdm[train_num:]]
        
        np.random.set_state(st0)
        return train_id, test_id

    def normalize(self, sq):
        # sq shape: (N, L, V)
        N, L, V = sq.shape
        d = sq.reshape(-1, V)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(N, L, V)

    def unnormalize(self, sq):
        # sq shape: (N, L, V)
        N, L, V = sq.shape
        d = sq.reshape(-1, V)
        if self.auto_norm:
            d = unnormalize_to_zero_to_one(d)
        d = self.scaler.inverse_transform(d)
        return d.reshape(N, L, V)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data.astype(np.float32)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def divide(data, ratio, seed=2024):
        # Legacy support
        return CustomDataset.divide_indices(data, ratio, seed)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads target column efficiently without loading entire CSV
        """
        # 1. Identify correct column index
        headers = pd.read_csv(filepath, nrows=0).columns.tolist()
        use_cols = [0]
        if len(headers) > 1:
            matched = [i for i, c in enumerate(headers) if c.lower() == name.lower()]
            if matched:
                use_cols = [matched[0]]
                print(f"  [Data] Selecting column: '{name}' (Index: {matched[0]})")
            else:
                print(f"  [Data] '{name}' not found, defaulting to first column.")
        
        # 2. Optimized read
        df = pd.read_csv(filepath, usecols=use_cols, engine='c')
        data = df.values.astype(np.float32)
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2024):
        # For testing/imputation periods
        masks = np.ones((len(self.samples), self.window, self.var_num), dtype=bool)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for i, idx in enumerate(self.samples):
            x = self.data[idx : idx + self.window] 
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                               self.distribution) 
            masks[i, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        np.random.set_state(st0)
        return masks

    def __getitem__(self, ind):
        idx = self.samples[ind]
        # Dynamically slice the window from the sequence (Lazy Loading)
        x = self.data[idx : idx + self.window] 
        
        if self.period == 'test':
            m = self.masking[ind]
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
