
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
        self.__gf = {'G_mean_pr','G_nodes', 'G_diameter', 'G_num_cliques', 'G_rank', 'G'}
        
        pd.set_option('mode.chained_assignment', None)
    
        
        if not out_dir.exists():
            os.mkdir(self.out_dir)
        
        self.train:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',cache_dir='dataset')['train'].to_pandas()       # type: ignore
        self.validation:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',cache_dir='dataset')['validation'].to_pandas()    # type: ignore
    
    def __produce(self, dataset:pd.DataFrame) -> pd.DataFrame:
        
        prc_result = pd.DataFrame(columns=self.features_enable)
        tmp=[]
        for feature in self.features_enable:
            if feature in dataset.columns.tolist(): 

                prc_result[feature] = dataset[feature]
            else:
                tmp.append(feature)

        batch_cc = 0

        t = tqdm(desc='batch compute', total=(dataset.size//self.batch_size) + 1)
        t.set_postfix({'batch': batch_cc})
        for batch in batch_generator(dataset, batch_size=self.batch_size): # type: ignore
            batch_cc +=1
            
            t.set_postfix({'batch': batch_cc})
            for feature in tmp:
                if feature == 'reference':
                    batch = count_references(batch.copy(), self.conn)
                elif feature == 'languages':
                    batch = dominant_langs(batch.copy(), self.conn)
                elif feature == 'length_lan':
                    batch = langs_length(batch.copy(), self.conn)
                elif feature in self.__gf:
                    mask = list(self.__gf.intersection(self.features_enable))
                    batch = G_factor(batch.copy(), 8, 3)
                    prc_result.loc[batch.index, mask] = batch[mask]
                    continue
                else:
                    raise ValueError(f'Label:{feature} not valid')
                
                prc_result.loc[batch.index, feature] = batch[feature]

                t.update(len(batch))
                
        return prc_result
    
    def produce(self, train:bool=True) -> list[pd.DataFrame]:
        """
        Transform Cultural dataset in new dataset with additional or subset of features
        """
        
        product = []
        prc_validation = self.__produce(self.validation)
        prc_validation.to_csv(self.out_dir.joinpath('validation.tsv'), sep='\t')
        product.append(prc_validation)
        if train: # some cases need train set
            prc_train = self.__produce(self.train)
            prc_train.to_csv(self.out_dir.joinpath('train.tsv'), sep='\t')
            product.append(prc_train)
        return product
    
    def produce_one_entry(self, entry:pd.Series) -> pd.Series:
        pass


if __name__ == '__main__':
        fat = CU_Dataset_Factory(out_dir=path.PosixPath('.'), batch_size=10, features_enable=['label', 'reference', 'languages','G_mean_pr','G_nodes', 'G_num_cliques', 'G_rank']).produce(False)

    
    