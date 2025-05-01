from typing import Any
import torch
import numpy as np
import pandas as pd
from pathlib import PosixPath

class PyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file:PosixPath, features:list[str] | str, target_label:str) -> None:
        super().__init__()
        
        self._file = dataset_file
        self._target = target_label
        self._sep = ''

        if self._file.exists():
            if self._file.suffix == '.csv':
                self._sep  = ',' 
            elif self._file.suffix == '.tsv':
                self._sep = '\t'
            else:
                raise Exception('Dataset format not valid, save data in follow formats: [csv, tsv] ')
        else:
            raise FileNotFoundError(f'File {self._file} not found')
        
        self._data = pd.read_csv(self._file, sep=self._sep)
        
        self._y = self._data[self._target].copy(deep=True).astype(np.float32).to_numpy()
        
        if features == '*':
            self._X = self._data.drop(self._target, axis=1).astype(np.int64).to_numpy()
        else:
            self._X = self._data.drop(self._target, axis=1)[features].astype(np.int64).to_numpy()
    
    
    def __len__(self) -> int:
        return self._y.size
    
