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
        
        # Decide Windowing Strategy based on style
        if self.style == 'non_overlapping':
            # Non-Overlapping: for Sampling (full coverage)
            self.sample_num_total = max(self.len // self.window, 0)
            print(f"[Dataset] {name}: Using NON-OVERLAPPING windows (for Sampling)")
        else:
            # Sliding Window: for Training (maximum data)
            self.sample_num_total = max(self.len - self.window + 1, 0)
            print(f"[Dataset] {name}: Using SLIDING windows (for Training)")
            
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
                self.masking = np.ones(self.samples.shape, dtype=bool)
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        # data is (L, V), normalized
        
        if self.style == 'non_overlapping':
            # Non-Overlapping indices: [0, 512, 1024, ...]
            indices = np.arange(self.sample_num_total) * self.window
            print(f"  -> Created {len(indices)} non-overlapping blocks")
        else:
            # Sliding Window indices: [0, 1, 2, 3...]
            indices = np.arange(self.sample_num_total)
            print(f"  -> Created {len(indices)} sliding windows")
            
        train_indices, test_indices = self.divide(indices, proportion, seed)

        # DENSITY & CONTINUITY BOOSTER (Apply to Training only)
        if self.period == 'train' and len(train_indices) > 0:
            if self.name.lower() == 'fridge':
                print(f"  [Continuity Booster] Skipping for {self.name} as requested (Avoiding over-boosting)")
            else:
                print(f"  [Continuity Booster] Analyzing training windows for transitions...")
                active_ids = []
                for idx in train_indices:
                    if np.max(data[idx : idx + self.window, 0]) > 0.05:
                        active_ids.append(idx)
                
                active_ids = np.array(active_ids)
                if len(active_ids) > 0:
                    boost_factor = 4
                    boosted_versions = [train_indices]
                    
                    for _ in range(boost_factor - 1):
                        # Apply Jitter (random shifting) for other appliances
                        jitter = np.random.randint(-2, 3, size=len(active_ids))
                        jittered_active = np.clip(active_ids + jitter, 0, self.sample_num_total - 1)
                        boosted_versions.append(jittered_active)
                    
                    train_indices = np.concatenate(boosted_versions)
                    print(f"  [Continuity Booster] Found {len(active_ids)} active windows. Training set boosted to {len(train_indices)} samples.")

        # CRITICAL FIX: Sort indices to maintain temporal order (Jan -> Dec)
        # Without this, 'divide' returns shuffled random indices!
        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

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
                
                # For multivariate (9 cols), only inverse-transform power
                if self.var_num == 9:
                    power = d[:, 0:1]
                    time_features = d[:, 1:]
                    
                    if self.auto_norm:
                        power = unnormalize_to_zero_to_one(power)
                    power = self.scaler.inverse_transform(power)
                    
                    d = np.concatenate([power, time_features], axis=1)
                else:
                    # Single column case
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
        
        # For multivariate (9 cols), only scale power (col 0)
        if self.var_num == 9:
            power = d[:, 0:1]
            time_features = d[:, 1:]
            
            power_scaled = self.scaler.transform(power)
            if self.auto_norm:
                power_scaled = normalize_to_neg_one_to_one(power_scaled)
            
            d = np.concatenate([power_scaled, time_features], axis=1)
        else:
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
        # For multivariate (9 cols), only scale power (col 0), keep time features as-is
        if rawdata.shape[-1] == 9:
            power = rawdata[:, 0:1]  # Power column
            time_features = rawdata[:, 1:]  # Time features (already in [-1,1])
            
            # Scale only power
            power_scaled = self.scaler.transform(power)
            if self.auto_norm:
                power_scaled = normalize_to_neg_one_to_one(power_scaled)
            
            # Concatenate scaled power with original time features
            data = np.concatenate([power_scaled, time_features], axis=1)
        else:
            # Single column case
            data = self.scaler.transform(rawdata)
            if self.auto_norm:
                data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        # For multivariate (9 cols), only unscale power (col 0)
        if data.shape[-1] == 9:
            power = data[:, 0:1]  # Power column
            time_features = data[:, 1:]  # Time features
            
            # Unscale only power
            if self.auto_norm:
                power = unnormalize_to_zero_to_one(power)
            power_original = self.scaler.inverse_transform(power)
            
            # Concatenate with time features
            result = np.concatenate([power_original, time_features], axis=1)
        else:
            # Single column case
            if self.auto_norm:
                data = unnormalize_to_zero_to_one(data)
            result = self.scaler.inverse_transform(data)
        return result
    
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

        regular_data = data[regular_train_id]
        irregular_data = data[irregular_train_id]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        Supports both single-column and multi-column formats.
        Automatically identifies appliance power column and time features (sin/cos).
        """
        df = pd.read_csv(filepath, header=0)
        
        # 1. Try to find the appliance column
        app_col = None
        for col in df.columns:
            if col.lower() == name.lower() or col.lower() == 'power':
                app_col = col
                break
        
        # 2. Find time columns (look for _sin or _cos)
        time_cols = [col for col in df.columns if '_sin' in col or '_cos' in col]
        
        if app_col and time_cols:
            print(f"[OK] Found target column '{app_col}' and {len(time_cols)} time features.")
            data = df[[app_col] + time_cols].values
            
            # CRITICAL FIX: Only fit scaler on power column (column 0)
            # Time features are already sin/cos in [-1,1], don't scale them!
            scaler = MinMaxScaler()
            scaler = scaler.fit(data[:, 0:1])  # Fit only on power column
            print(f"  [Scaler] Fitted on power column only (range: {scaler.data_min_[0]:.2f} to {scaler.data_max_[0]:.2f}W)")
        elif app_col:
            print(f"[OK] Found target column '{app_col}', no time features found.")
            data = df[[app_col]].values
            scaler = MinMaxScaler()
            scaler = scaler.fit(data)
        else:
            # Fallback to standard behavior if column identification fails
            print(f"[Warning] Could not identify column '{name}', using all {len(df.columns)} columns.")
            data = df.values
            scaler = MinMaxScaler()
            scaler = scaler.fit(data)
        
        # Use float32 to save memory
        data = data.astype(np.float32)
        
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
