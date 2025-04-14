
import pandas as pd
import pathlib as path
import os
from tqdm import tqdm
from datasets import load_dataset
from utils import extract_entity_id, extract_entity_name, batch_generator
from features import count_references, dominant_langs, langs_length, G_factor
from Connection import Wiki_high_conn
class CU_Dataset_Factory:
    def __init__(self, out_dir:path.PosixPath, batch_size:int=1,features_enable:list[str]=['label']) -> None:
        
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.features_enable = features_enable
        self.conn = Wiki_high_conn()
    
        
        if not out_dir.exists():
            os.mkdir(self.out_dir)
        
        self.train:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['train'].to_pandas()            # type: ignore
        self.validation:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['validation'].to_pandas()   # type: ignore
    
    def __produce(self, dataset:pd.DataFrame) -> pd.DataFrame:
        
        prc_result = pd.DataFrame(columns=self.features_enable)
        tmp=[]
        for feature in self.features_enable:
            if feature in dataset.columns.tolist(): 

                prc_result[feature] = dataset[feature]
            else:
                tmp.append(feature)

        print(prc_result)

        for batch in tqdm(batch_generator(dataset, batch_size=self.batch_size), desc='batch compute'):
            for feature in tmp:
                if feature == 'reference':
                    count_references(batch, self.conn)
                elif feature == 'languages':
                    dominant_langs(batch, self.conn)
                elif feature == 'length_lan':
                    langs_length(batch, self.conn)
                elif feature == 'G':
                    for v in batch:
                        G_factor(v)
                        pd.concat([prc_result, batch], ignore_index=True, sort=False)
                elif feature == []:
                    break
                else:
                    raise ValueError(f'Label:{feature} not valid')
        
        return prc_result
    
    def produce(self) -> list[pd.DataFrame]:

        #prc_train = self.__produce(self.train)
        prc_validation = self.__produce(self.validation)

        #prc_train.to_csv(self.out_dir.joinpath('train.tsv'), sep='\t')
        prc_validation.to_csv(self.out_dir.joinpath('validation.tsv'), sep='\t')

        return [[], prc_validation]
    
    def produce_one_entry(self, entry:pd.Series) -> pd.Series:
        pass


if __name__ == '__main__':
        fat = CU_Dataset_Factory(out_dir=path.PosixPath('.'), batch_size=5, features_enable=['label', 'reference']).produce()

    
    