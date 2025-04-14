
import pandas as pd
import pathlib as path
import os
import tqdm
from datasets import load_dataset
from utils import extract_entity_id, extract_entity_name

class CU_Dataset_Factory:
    def __init__(self, out_dir:path.PosixPath, format:str, batch_size:int=1,fectures_enable:list[str]=['label'], fectures_add:list[str]=[]) -> None:
        

        self.out_dir = out_dir
        if format == 'csv':
            self.format = 'csv'
        elif format == 'pd':
            self.format = 'pandas'
        else:
            self.format = 'pandas'
        
        if not out_dir.exists():
            os.mkdir(self.out_dir)
        
        self.train = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['train'].to_pandas()             # type: ignore
        self.validation = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['validation'].to_pandas()   # type: ignore
    
    def produce(self) -> pd.DataFrame:

        for num_batch, batch in enumerate()
        pass

    def produce_one_entry(self, entry:pd.Series) -> pd.Series:
        pass
    