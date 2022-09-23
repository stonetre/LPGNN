import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd

def normalized(data, normalize_method, norm_statistic=None, normalize_column=False):
    if normalize_method == 'min_max':
        if not norm_statistic:
            if normalize_column:
                norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
            else:
                norm_statistic = dict(max=np.max(data), min=np.min(data))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min'])
        data = (data - np.array(norm_statistic['min'])) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            if normalize_column:
                norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
            else:
                norm_statistic = dict(mean=np.mean(data), std=np.std(data)) 
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        #std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        #norm_statistic['std'] = std

    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic, normalize_column=False):
    if normalize_method == 'min_max':
        if not norm_statistic:
            if normalize_column:
                norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
            else:
                norm_statistic = dict(max=np.max(data), min=np.min(data))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min'])
        data = data * scale + np.array(norm_statistic['min'])
    elif normalize_method == 'z_score':
        if not norm_statistic:
            if normalize_column:
                norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
            else:
                norm_statistic = dict(mean=np.mean(data), std=np.std(data)) 
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        #std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, normalize_column=False, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        self.normalize_column = normalize_column
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        # self.data.shape (31535, 137)
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()

        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic, normalize_column)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        # x_index_set
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]

        return x_end_idx
