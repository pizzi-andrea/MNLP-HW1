
import pandas as pd
import pathlib as path
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datasets import load_dataset
from utils import  batch_generator
from features import count_references, dominant_langs, langs_length, G_factor
from Connection import Wiki_high_conn
class CU_Dataset_Factory:
    def __init__(self, out_dir:path.PosixPath, target_feature:str, batch_size:int=1,encoding:bool=False, features_enable:list[str]=['label']) -> None:
        
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.features_enable = features_enable
        self.conn = Wiki_high_conn()
        self.encode = encoding
        self.label_e = LabelEncoder()
        self.__gf = {'G_mean_pr','G_nodes', 'G_diameter', 'G_num_cliques', 'G_rank', 'G_avg'}
        if not target_feature in features_enable:
            raise ValueError('target feature must be in features_enable_')
        self.target = target_feature
        pd.set_option('mode.chained_assignment', None)
    
        if not out_dir.exists():
            os.mkdir(self.out_dir)
        
        self.train:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',cache_dir='dataset')['train'].to_pandas()       # type: ignore
        self.validation:pd.DataFrame = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',cache_dir='dataset')['validation'].to_pandas()    # type: ignore
    
    def __produce(self, dataset:pd.DataFrame) -> pd.DataFrame:
        
        prc_result = pd.DataFrame(columns=self.features_enable)
        exstra=[]
        for feature in tqdm(self.features_enable, desc='copy dataset'):
            if feature in dataset.columns.tolist(): 
                if self.encode and (dataset[feature].dtype == pd.StringDtype() or dataset[feature].dtype == object):
                    if feature == self.target:
                        prc_result[feature] = self.label_e.fit_transform(dataset[feature])
                    else:
                        dummies = pd.get_dummies(dataset[feature], dtype=pd.Int32Dtype(), prefix=feature)
                        prc_result = pd.concat([prc_result, dummies], axis=1)
                        prc_result = prc_result.drop(labels=feature, axis=1)

                else:
                        prc_result[feature] = dataset[feature]
                        
            else:
                exstra.append(feature)

        batch_cc = 0

        t = tqdm(desc='batch compute', total=(dataset.size))
        t.set_postfix({'batch': batch_cc})
        for batch in batch_generator(dataset, batch_size=self.batch_size): # type: ignore
            batch_cc +=1
            
            t.set_postfix({'batch': batch_cc})
            for feature in exstra:
                if feature == 'reference':
                    batch = count_references(batch.copy(), self.conn)
                elif feature == 'languages':
                    batch = dominant_langs(batch.copy(), self.conn)
                elif feature == 'length_lan':
                    batch = langs_length(batch.copy(), self.conn)
                elif feature in self.__gf:
                    mask = list(self.__gf.intersection(self.features_enable))
                    batch = G_factor(batch.copy(), 3, 10)
                    prc_result.loc[batch.index, mask] = batch[mask]
                    continue
                else:
                    raise ValueError(f'Label:{feature} not valid')
                
                prc_result.loc[batch.index, feature] = batch[feature]

                t.update(len(batch))
                
        return prc_result
    
    def produce(self, train:bool=True) -> pd.DataFrame:
        """
        Transform Cultural dataset in new dataset with additional or subset of features
        """
        
        product = None
        
        if train: # some cases need train set
            prc_train = self.__produce(self.train)
            prc_train.to_csv(self.out_dir.joinpath('train.tsv'), sep='\t', mode='w')
            product = prc_train
        else:
            prc_validation = self.__produce(self.validation)
            prc_validation.to_csv(self.out_dir.joinpath('validation.tsv'), sep='\t', mode='w')
            product = prc_validation

        return product
    
    def exists(self, train:bool) -> bool:
        return (not train and path.Path('./validation.tsv').exists()) or (train and path.Path('./train.tsv').exists())


    def load(self, train:bool=True) -> pd.DataFrame:
        result = None
        if train:
            if path.Path('./train.tsv').exists():
                result = pd.read_csv('./train.tsv', sep='\t').drop(columns=['Unnamed: 0'])
            else:
                raise FileNotFoundError('dataset not found, please use produce() before')
        else:

            if path.Path('./validation.tsv').exists():
                result = pd.read_csv('./validation.tsv', sep='\t').drop(columns=['Unnamed: 0'])
            else:
                raise FileNotFoundError('dataset not found, please use produce() before')
        
        return result

    def produce_one_entry(self, entry:pd.Series) -> pd.DataFrame:
        result = self.__produce(pd.DataFrame(entry))
        return result


if __name__ == '__main__':
        fat = CU_Dataset_Factory(out_dir=path.PosixPath('.'), target_feature='label', encoding=True, batch_size=10, features_enable=['label', 'category']).produce(False)

    
    