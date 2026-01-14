import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from numpy.lib.format import open_memmap
from Utils.masking_utils import noise_mask


class LazyWindows:
    def __init__(self, data, indices, window):
        self.data = data
        self.indices = indices
        self.window = window
        self.var_num = data.shape[-1]
        self.shape = (len(indices), window, self.var_num)
        self.dtype = data.dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            start = self.indices[item]
            return self.data[start : start + self.window]
        
        if isinstance(item, tuple):
            idx = item[0]
            res = self.__getitem__(idx)
            if len(item) > 1:
                return res[item[1:]]
            return res

        # Handle slice or array
        curr_indices = self.indices[item]
        if len(curr_indices) == 0:
            return np.empty((0, self.window, self.var_num), dtype=self.dtype)
        return np.stack([self.data[idx : idx + self.window] for idx in curr_indices])


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
        # data is (L, V), normalized
        indices = np.arange(self.sample_num_total)
        train_indices, test_indices = self.divide(indices, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                self._save_chunked_npy(data, test_indices, os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), unnormalize=True)
                self._save_chunked_npy(data, test_indices, os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize=False)
            
            self._save_chunked_npy(data, train_indices, os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), unnormalize=True)
            self._save_chunked_npy(data, train_indices, os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize=False)

        train_data = LazyWindows(data, train_indices, self.window)
        test_data = LazyWindows(data, test_indices, self.window)

        return train_data, test_data

    def _save_chunked_npy(self, data, indices, filename, unnormalize=True, chunk_size=1000):
        if len(indices) == 0:
            return
        
        print(f"Memory-efficient saving (chunked) to {filename}...")
        shape = (len(indices), self.window, self.var_num)
        # Using open_memmap to write a valid .npy file piece by piece
        # Default to float32 to save space
        fp = open_memmap(filename, dtype='float32', mode='w+', shape=shape)
        
        for i in range(0, len(indices), chunk_size):
            end = min(i + chunk_size, len(indices))
            chunk_idx = indices[i:end]
            # Create the windows for this chunk
            chunk_windows = np.stack([data[idx : idx + self.window] for idx in chunk_idx])
            
            if unnormalize:
                # Unnormalize this chunk
                d = chunk_windows.reshape(-1, self.var_num)
                if self.auto_norm:
                    d = unnormalize_to_zero_to_one(d)
                d = self.scaler.inverse_transform(d)
                chunk_windows = d.reshape(-1, self.window, self.var_num)
            
            fp[i:end] = chunk_windows.astype(np.float32)
            
        # Flush and close
        del fp
        print(f"✓ Saved {filename}")

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        if isinstance(sq, LazyWindows):
            # If someone really wants the whole thing, materialize it
            sq = sq[:]
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2024):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        Supports both single-column and two-column (aggregate, power) formats.
        If 'power' column exists, extracts only that column for diffusion model training.
        """
        df = pd.read_csv(filepath, header=0)
        # If CSV has 'power' column, use it directly (for two-column format: aggregate, power)
        if 'power' in df.columns:
            data = df[['power']].values
        # Otherwise, use all columns (for backward compatibility)
        else:
            data = df.values
        
        # Use float32 to save memory
        data = data.astype(np.float32)
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2024):
        masks = np.ones(self.samples.shape, dtype=bool)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind]  # (seq_length, feat_dim) array
            m = self.masking[ind]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind]  # (seq_length, feat_dim) array
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
