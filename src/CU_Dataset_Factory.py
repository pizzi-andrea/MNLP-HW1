import pandas as pd
import pathlib as path
import os
from sklearn.preprocessing import LabelEncoder
from requests import get
from tqdm import tqdm
from datasets import load_dataset
from utils import batch_generator
from features import *
from Connection import Wiki_high_conn

# Class to append the new features to the dataset and produce the new dataset
class CU_Dataset_Factory:
    """
    Builder Class use to generate train or test dataset with required features
    """
    def __init__(
        self,
        load:bool = False,
    ) -> None:


        def __wiki_name(qids: list[str]) -> dict[str, str]:
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

        self._cache = path.PosixPath('./_cache')
        self.conn = Wiki_high_conn()
        self.label_e = LabelEncoder()
        self.sgf = {"G_mean_pr", "G_nodes", "G_num_cliques", "G_avg"}
        self.pgf = {'languages', 'reference'}
        self.pef = {'n_mod', 'n_visits'}
        self.id = {'qid', 'wiki_name'}
        pd.set_option("mode.chained_assignment", None)
        if not self._cache.exists():
            os.mkdir(self._cache)
        
        if load and self._cache.exists() and self._cache.joinpath('base_train.csv').exists() and self._cache.joinpath('base_validation.csv').exists():
            self.validation = pd.read_csv(self._cache.joinpath('base_validation.csv'), sep='\t')
            self.train = pd.read_csv(self._cache.joinpath('base_train.csv'), sep='\t')

            self.train = self.train.drop('Unnamed: 0', axis=1)
            self.validation = self.validation.drop('Unnamed: 0', axis=1)
        else:
            self.train: pd.DataFrame = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset", cache_dir="dataset")["train"].to_pandas()  # type: ignore
            self.validation: pd.DataFrame = load_dataset("sapienzanlp/nlp2025_hw1_cultural_dataset", cache_dir="dataset")["validation"].to_pandas()  # type: ignore
            self.train['qid'] = self.train['item'].str.extract(r'(Q\d+)', expand=False)
            self.train = self.train.drop(['item', 'name'], axis=1)
            self.validation['qid'] = self.validation['item'].str.extract(r'(Q\d+)', expand=False)
            self.validation = self.validation.drop(['item', 'name'], axis=1)
            self.train['wiki_name'] = self.train['qid'].map(__wiki_name(self.train['qid'].to_list())).fillna(0)
            self.validation['wiki_name'] = self.validation['qid'].map(__wiki_name(self.validation['qid'].to_list())).fillna(0)

            self.train.to_csv(self._cache.joinpath('base_train.csv'), sep='\t')
            self.validation.to_csv(self._cache.joinpath('base_validation.csv'), sep='\t')

                    
               

    # Hidden function that recursively appends the new features through a series of if
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
                    batch = is_disambiguation(batch[join_fe], conn)
                else:
                    raise ValueError(f"Label:{feature} not valid")

                # (dopo aver inizializzato la colonna fuori dal loop)
                delta = prc_result[join_fe].map(r).fillna(0)
                prc_result.loc[:, feature] = prc_result[feature].add(delta, fill_value=0)
                
            t.update(original_batch_len)

        t.close()
        return prc_result

    # Function that calls back __produce and returns the new dataset
    def produce(self, out_dir: path.PosixPath, enable_feature:list[str], targe_feature:str, batch_s:int = 1, encoding: bool = False, train: bool = True) -> pd.DataFrame:
        """
        Transforms Cultural dataset in new dataset with additional features or with a subset of features
        """

        product = None
        if not out_dir.exists():
            os.mkdir(out_dir)

        if train:  # some cases need training set
            prc_train = self.__produce(self.train, enable_feature, targe_feature, batch_s, encoding)
            prc_train.to_csv(out_dir.joinpath("train.tsv"), sep="\t", mode="w")
            product = prc_train
        else:
            prc_validation = self.__produce(self.validation, enable_feature, targe_feature, batch_s, encoding)
            prc_validation.to_csv(
                out_dir.joinpath("validation.tsv"), sep="\t", mode="w"
            )
            product = prc_validation

        return product
    
    def produce_one_entry(self, entry: pd.Series, encoding: bool = False) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    d = CU_Dataset_Factory(load=True)
    #print(d.validation.head(10))
    d.produce(out_dir=path.PosixPath('.'), batch_s=10, enable_feature=['label', 'wiki_name', 'qid', 'languages', 'G'],targe_feature='label', train=False)