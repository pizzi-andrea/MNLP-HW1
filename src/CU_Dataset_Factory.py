import pandas as pd
import pathlib as path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from requests import get
from datasets import load_dataset
from utils import batch_generator
from features import *
from datasets.splits import NamedSplit
from Connection import Wiki_high_conn
from Loader import Loader
from datasets import load_dataset
from pathlib import PosixPath

class Hf_Loader(Loader):
    def __init__(self, hf_url:str, split:str|NamedSplit, limit:int|None = None) -> None:
        super().__init__()
        self.hf_url = hf_url
        self.split = split
        self.limit = limit
        self.hf_cache = './hugging_dataset/'
   
    def get(self) -> pd.DataFrame:
        df:pd.DataFrame = load_dataset(self.hf_url, cache_dir=self.hf_cache, split=self.split).to_pandas() # type: ignore
        if self.limit:
            return df.iloc[:self.limit, :]
        else:
            return df
        

class Local_Loader(Loader):
    def __init__(self, file_path:str|PosixPath, limit:int|None = None) -> None:
        super().__init__()
        self.file_path = PosixPath(file_path)
        self.limit = limit
       
   
    def get(self) -> pd.DataFrame:
        df = None
        if self.file_path.suffix == 'tsv':
            df  = pd.read_csv(self.file_path, sep='\t')
            df = df.drop('Unnamed: 0', axis=1)
        elif self.file_path.suffix == 'csv':
            df  = pd.read_csv(self.file_path, sep=',')
        else:
            raise TypeError('File format Not supported')
        
        if self.limit:
            return df.iloc[:self.limit, :]
        else:
            return df

            
        



# Class to append the new features to the dataset and produce the new dataset
class CU_Dataset_Factory:
    """
    Builder Class use to generate train or test dataset with required features
    """
    def __init__(
        self,
        out_dir:PosixPath|str
    ) -> None:

        
        self.out_dir = PosixPath(out_dir)
        self.conn = Wiki_high_conn()
        self.label_e = LabelEncoder()
        self.sgf = {"G_mean_pr", "G_nodes", "G_num_cliques", "G_avg"}   # features about wikipedia network
        self.pgf = {'languages', 'reference', 'ambiguos'}                           # features about page
        self.pef = {'n_mod', 'n_visits'}                                # features about users
        self.id = {'qid', 'wiki_name'}                                  # identification fields
        pd.set_option("mode.chained_assignment", None)
       
        #self.train: pd.DataFrame = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset", cache_dir="dataset")["train"].to_pandas()  # type: ignore
        #self.validation: pd.DataFrame = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset", cache_dir="dataset")["validation"].to_pandas()  # type: ignore


    def __wiki_name(self, qids: list[str]) -> dict[str, str]:
            def fetch_batch(qids_batch: list[str]) -> dict[str, str]:
                # Parametri per l'API di Wikidata
                params = {
                    'action': 'wbgetentities',
                    'ids': '|'.join(qids_batch),
                    'sites': 'wikipedia',
                    'props': 'sitelinks',
                    'format': 'json',
                    'utf8': 1
                }
           
                data = get('https://www.wikidata.org/w/api.php', params).json().get('entities', {})
                batch_map = {}
                for qid, entity in data.items():
                    sitelinks = entity.get('sitelinks', {})
                    lang_key = f"enwiki"  # es. "enwiki"
                    title = sitelinks.get(lang_key, {}).get('title', '')
                    batch_map[qid] = title
                return batch_map

            qid_to_title: dict[str, str] = {}
            batch_size = 50
            for i in range(0, len(qids), batch_size):
                batch = qids[i:i + batch_size]
                qid_to_title.update(fetch_batch(batch))

            return qid_to_title
    
    def __produce(self, dataset: pd.DataFrame, enable_feature:list[str], targe_feature:str, batch_s:int = 1, encode:bool= True) -> pd.DataFrame:

        prc_result = pd.DataFrame()
        exstra = []

        # Copia diretta delle colonne esistenti nel dataset
        for feature in tqdm(enable_feature, desc="copy dataset"):
           
            if feature == 'G':
                for c in self.sgf:
                        prc_result.insert(0, c, None)
            else:
                prc_result.insert(0, feature, None)

            if feature in dataset.columns.tolist():
                if encode and not(feature in self.id) and (
                    dataset[feature].dtype == pd.StringDtype()
                    or dataset[feature].dtype == object
                ):
                    if feature == targe_feature:
                        prc_result[feature] = self.label_e.fit_transform(dataset[feature])
                    else:
                        
                        dummies = pd.get_dummies(
                            dataset[feature], dtype=pd.Int32Dtype(), prefix=feature
                        )
                        
                        prc_result = prc_result.drop(feature, axis=1)
                        prc_result = pd.concat([prc_result, dummies], axis=1)

                else:
                    prc_result[feature] = dataset[feature]
            else:
                exstra.append(feature)
                prc_result[feature] = 0

        # Elaborazione batch
        batch_cc = 0
        t = tqdm(desc="batch compute", total=len(dataset))  # usa len(dataset) invece di dataset.size
        for batch in batch_generator(dataset, batch_size=batch_s):  # type: ignore
            batch_cc += 1
            t.set_postfix({"batch": batch_cc})
            original_batch_len = len(batch)

            for feature in exstra:
                
                t.set_description(feature, refresh=True)
                if feature == "reference":
                    join_fe = 'wiki_name' 
                    r = count_references(batch[join_fe], self.conn)
                elif feature == "languages":
                    join_fe = 'qid'
                    r = dominant_langs(batch[join_fe], self.conn)
                elif feature == "length_lan":
                    join_fe = 'qid' 
                    r = langs_length(batch[join_fe], self.conn)
                elif feature == 'G':
                    join_fe = 'wiki_name'
                    mask = list(self.sgf)
                    r = G_factor(batch[join_fe], batch['qid'], 1, 1, 1, 10, threads=1)
                    
                    prc_result.loc[r.index, mask] = r[mask]
                    prc_result  = prc_result.drop('G')
                    continue # add new features  this ...
                elif feature == 'n_mod': # contare il numero medio di modifiche in un intervallo di tempo specifico
                    join_fe = 'wiki_name' 
                    r = num_mod(batch[join_fe], self.conn)
                    
                elif feature == 'n_visits': # contare il numero medio di visite al giorno in un intervallo di tempo
                    pass 
                elif feature == 'ambiguos':
                    join_fe = 'qid' 
                    batch = is_disambiguation(batch[join_fe], self.conn)
                else:
                    raise ValueError(f"Label:{feature} not valid")

                # (dopo aver inizializzato la colonna fuori dal loop)
                delta = prc_result[join_fe].map(r).fillna(0)
                prc_result.loc[:, feature] = prc_result[feature].add(delta, fill_value=0)
                
            t.update(original_batch_len)

        t.close()
        return prc_result

    # Function that calls back __produce and returns the new dataset
    def produce(self, loader:Loader, out_file: path.PosixPath|str, enable_feature:list[str], targe_feature:str, batch_s:int = 1, encoding: bool = False, train: bool = True) -> pd.DataFrame|None:
        """
        Transforms Cultural dataset in new dataset with additional features or with a subset of features
        """

        out_file = PosixPath(out_file)
        
        try:
            dataset  = loader.get()
        except Exception as err:
            print(f'loader error {err}')
            return None

        dataset['qid'] = dataset['item'].str.extract(r'(Q\d+)', expand=False)
        dataset['wiki_name'] = dataset['qid'].map(self.__wiki_name(dataset['qid'].to_list())).fillna(0)
        dataset = dataset.drop(['item', 'name'], axis=1)
        prc = self.__produce(dataset, enable_feature, targe_feature, batch_s, encoding)
        prc.to_csv(self.out_dir.joinpath(out_file), sep="\t", mode="w")

       
        return prc


if __name__ == '__main__':
    l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", 'validation', None)
    d = CU_Dataset_Factory(out_dir='.')
    #print(d.validation.head(10))
    d.produce(l, 'validation.tsv', batch_s=10, enable_feature=['label', 'wiki_name', 'qid', 'languages', 'reference', 'ambiguos'], targe_feature='label', train=False)