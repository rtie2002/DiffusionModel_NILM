import os
import torch
import numpy as np
import pandas as pd
import yaml

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
        mean_mask_length=3,
        boost_factor=None,
        boost_threshold=0.2,
        save_train_npy=False
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.boost_factor = boost_factor
        self.boost_threshold = boost_threshold
        self.save_train_npy = save_train_npy
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
                threshold = self.boost_threshold
                for idx in train_indices:
                    if np.max(data[idx : idx + self.window, 0]) > threshold:
                        active_ids.append(idx)
                
                active_ids = np.array(active_ids)
                if len(active_ids) > 0:
                    # Default to 4 (previous behavior) unless manually overridden
                    current_boost = self.boost_factor if self.boost_factor is not None else 4
                    
                    if current_boost > 1:
                        boosted_versions = [train_indices]
                        for _ in range(int(current_boost) - 1):
                            jitter = np.random.randint(-2, 3, size=len(active_ids))
                            jittered_active = np.clip(active_ids + jitter, 0, self.sample_num_total - 1)
                            boosted_versions.append(jittered_active)
                        
                        train_indices = np.concatenate(boosted_versions)
                        print(f"  [Continuity Booster] Found {len(active_ids)} active windows. Training set boosted to {len(train_indices)} samples (Factor: {current_boost}).")
                    else:
                        print(f"  [Continuity Booster] Boost factor is 1. No dataset expansion applied.")

        # CRITICAL FIX: Sort indices to maintain temporal order (Jan -> Dec)
        # Without this, 'divide' returns shuffled random indices!
        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

        if self.save2npy:
            # Always save test set (needed for evaluation metrics)
            if 1 - proportion > 0:
                self._save_chunked_npy(data, test_indices, os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), unnormalize=True)
                self._save_chunked_npy(data, test_indices, os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize=False)
            
            # Save training set ONLY if explicitly requested (prevents 100GB+ disk usage)
            if self.save_train_npy:
                self._save_chunked_npy(data, train_indices, os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), unnormalize=True)
                self._save_chunked_npy(data, train_indices, os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize=False)
            else:
                if len(train_indices) > 500000:
                    print(f"  [Dataset] Skipping training NPY save (Dataset is large: {len(train_indices)} samples). Use save_train_npy=True to override.")

        train_data = LazyWindows(data, train_indices, self.window)
        test_data = LazyWindows(data, test_indices, self.window)

        return train_data, test_data

    def _save_chunked_npy(self, data, indices, filename, unnormalize=True, chunk_size=250):
        if len(indices) == 0:
            return
        
        # Estimate size to warn user
        total_size_gb = (len(indices) * self.window * self.var_num * 4) / (1024**3)
        print(f"  [Save] {os.path.basename(filename)} ({total_size_gb:.2f} GB)...")
        
        shape = (len(indices), self.window, self.var_num)
        # Using open_memmap to write a valid .npy file piece by piece
        fp = open_memmap(filename, dtype='float32', mode='w+', shape=shape)
        
        # Pre-allocate one chunk buffer to reuse memory
        chunk_buffer = np.zeros((chunk_size, self.window, self.var_num), dtype=np.float32)
        
        for i in range(0, len(indices), chunk_size):
            end = min(i + chunk_size, len(indices))
            current_batch_size = end - i
            chunk_idx = indices[i:end]
            
            # Use manual copy to avoid creating many small list/slice objects
            for j, idx in enumerate(chunk_idx):
                chunk_buffer[j] = data[idx : idx + self.window]
            
            # Take view of the filled portion
            current_chunk = chunk_buffer[:current_batch_size]
            
            if unnormalize:
                # In-place/efficient unnormalization
                d = current_chunk.reshape(-1, self.var_num)
                
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
                
                current_chunk = d.reshape(-1, self.window, self.var_num)
            
            fp[i:end] = current_chunk.astype(np.float32)
            
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
        # 0. Load Preprocess Config to get "Source of Truth" for thresholds
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(curr_dir))
        PREPROCESS_CONFIG_PATH = os.path.join(project_root, 'Config', 'preprocess', 'preprocess_multivariate.yaml')
        
        max_power = None
        real_max_power = None
        
        if os.path.exists(PREPROCESS_CONFIG_PATH):
            try:
                with open(PREPROCESS_CONFIG_PATH, 'r') as f:
                    conf = yaml.safe_load(f)
                    app_key = name.lower().replace(' ', '')
                    if app_key in conf.get('appliances', {}):
                        max_power = conf['appliances'][app_key].get('max_power')
                        real_max_power = conf['appliances'][app_key].get('real_max_power')
                        print(f"  [Auto-Config] Loaded '{app_key}': max_power={max_power}W, real_max_power={real_max_power}W")
            except Exception as e:
                print(f"  [Warning] Failed to parse preprocess config: {e}")

        df = pd.read_csv(filepath, header=0)
        
        # 1. Try to find the appliance column
        app_col = None
        for col in df.columns:
            if col.lower() == name.lower() or col.lower() == 'power':
                app_col = col
                break
        
        # 2. Find time columns
        time_cols = [col for col in df.columns if '_sin' in col or '_cos' in col]
        
        if app_col:
            power_data = df[app_col].values.astype(np.float32)
            
            # ALL DATA IS ALREADY [0, 1] IN CSV (MinMax relative to real_max_power)
            
            # 1. SPECIAL CASE: Fridge Resolution Restoration
            # If max_power (target) is different from real_max_power (original normalization),
            # we need to adjust the signal contrast so visibility is restored.
            if max_power is not None and real_max_power is not None and max_power != real_max_power:
                gain = real_max_power / max_power
                power_data = np.clip(power_data * gain, 0, 1.0)
                print(f"[DATA] Appliance '{name}' signal resolution-gain applied: {gain:.2f}x")
                actual_max_val = max_power
            else:
                # 2. STATUS QUO: Kettle, Dishwasher, etc.
                # Use directly [0, 1] as is. No changes to your successful logic.
                print(f"[DATA] Appliance '{name}' using verified MinMax logic from yesterday.")
                actual_max_val = max_power if max_power else 1.0

            # 3. PROTECTION AGAINST DOUBLE-NORMALIZATION
            # We set data_max_ to 1.0 because power_data is ALREADY the final [0, 1] signal 
            # we want the model to learn. This ensures self.__normalize() does nothing.
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array([0.0])
            scaler.data_max_ = np.array([actual_max_val]) # Use the physical max for un-normalization during eval
            scaler.scale_ = np.array([1.0])               # Force identity transform (1.0 ratio)
            scaler.min_ = np.array([0.0])
            
            # Monkey-patch transform to be an identity function for [0, 1] data
            scaler.transform = lambda x: x 

            if time_cols:
                data = np.concatenate([power_data.reshape(-1, 1), df[time_cols].values], axis=1)
            else:
                data = power_data.reshape(-1, 1)
        else:
            # Fallback
            print(f"[Warning] No NILM config. Using data-driven defaults.")
            data = df.values
            scaler = MinMaxScaler().fit(data)
        
        return data.astype(np.float32), scaler
    
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
