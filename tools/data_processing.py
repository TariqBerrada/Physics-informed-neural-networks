import numpy as np
import joblib, tqdm, os

from torch.utils.data import Dataset

class DatasetClass(Dataset):
    def __init__(self, data_0, data_b, data_f):
        self.data_0 = data_0
        self.data_b = data_b
        self.data_f = data_f

    def __len__(self):
        return self.data_f.shape[0]
    
    def __getitem__(self, idx):
        idx_b = idx%self.data_b.shape[0]
        idx_0 = idx%self.data_0.shape[0]
        sample = {'data_0' : self.data_0[idx_0, :], 'data_b' : self.data_b[idx_b, :], 'data_f' : self.data_f[idx, :]}
        return sample
