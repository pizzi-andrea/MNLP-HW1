import pathlib as path  # Imports for file system access
from os import mkdir
from pathlib import PosixPath

import pandas as pd  # Imports for data manipulation
from datasets import load_dataset
from datasets.splits import NamedSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Connection import Wiki_high_conn  # Imports custom code for project
from features import *
from Loader import Loader
from utils import batch_generator


class Hf_Loader(Loader):
    """Dataset Loader from Hugging-Face Hub"""

    def __init__(
        self, hf_url: str, split: str | NamedSplit, limit: int | None = None
    ) -> None:
        super().__init__()
        self.hf_url = hf_url
        self.split = split
        self.limit = limit
        self.hf_cache = "./hugging_dataset/"

    """Download dataset from Hugging-Face Hub and return it"""

    def get(self) -> pd.DataFrame:
        df = load_dataset(self.hf_url, cache_dir=self.hf_cache, split=self.split).to_pandas()  # type: ignore
        if self.limit:
            return df.iloc[: self.limit, :]
        else:
            return df


class Local_Loader(Loader):
    """Dataset Loader for local dataset"""

    def __init__(self, file_path: str | PosixPath, limit: int | None = None) -> None:
        super().__init__()
        self.file_path = PosixPath(file_path)
        self.limit = limit

        if not self.file_path.exists():
            raise FileNotFoundError(f"file {self.file_path} not found")

    def get(self) -> pd.DataFrame:
        """read data from `file_path` and return it"""
        df = None
        if self.file_path.suffix == ".tsv":
            df = pd.read_csv(self.file_path, sep="\t")

        elif self.file_path.suffix == ".csv":
            df = pd.read_csv(self.file_path, sep=",")
        else:
            raise TypeError("File format Not supported")

        if "Unnamed:0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        if self.limit:
            return df.iloc[: self.limit, :]

        return df


class CU_Dataset_Factory:
    """
    Builder Class use to generate train or test dataset with required features
    """

    def __init__(self) -> None:

        self.conn = Wiki_high_conn()
        self.label_e = LabelEncoder()

        ##################
        # Known features #
        ##################

        self.sgf = {
            "G_mean_pr",
            "G_nodes",
            "G_num_cliques",
            "G_avg",
            "G_num_components",
            "G_density",
            "G_largest_component_size",
        }  # features about wikipedia network
        self.pgf = {"languages", "reference", "ambiguos"}  # features about page
        self.pef = {"n_mod", "n_visits"}  # features about users
        self.id = {"qid", "wiki_name", "name", "item"}  # identification fields
        self.tf = {
            "relevant_words",
            "intro",
            "full_page",
        }  # features about text for transformers
        pd.set_option("mode.chained_assignment", None)

    def __wiki_name(self, qids: list[str]) -> dict[str, str]:
        conn = Wiki_high_conn()

        def fetch_batch(qids_batch: list[str], conn: Wiki_high_conn) -> dict[str, str]:

            # Parameters for Wikidata's API

            params = {
                "action": "wbgetentities",
                "sites": "wikipedia",
                "props": "sitelinks",
                "format": "json",
                "utf8": 1,
            }

            data = conn.get_wikidata(qids_batch, params).get("entities", {})
            batch_map = {}
            for qid, entity in data.items():
                sitelinks = entity.get("sitelinks", {})
                lang_key = f"enwiki"  # es. "enwiki"
                title = sitelinks.get(lang_key, {}).get("title", "")
                batch_map[qid] = title
            return batch_map

        qid_to_title: dict[str, str] = {}
        batch_size = 50
        for i in tqdm(range(0, len(qids), batch_size)):
            batch = qids[i : i + batch_size]
            qid_to_title.update(fetch_batch(batch, conn))
        return qid_to_title

    def __produce(
        self,
        dataset: pd.DataFrame,
        enable_feature: list[str],
        targe_feature: str | None,
        batch_s: int = 1,
        encode: bool = True,
    ) -> pd.DataFrame:

        prc_result = pd.DataFrame()
        extra = []

        for id_c in self.id:
            prc_result.insert(0, id_c, None)
            prc_result[id_c] = dataset[id_c]

        # Direct copy of existing columns in the dataset
        for feature in tqdm(enable_feature, desc="copy dataset"):

            if feature == "G":
                for c in self.sgf:
                    prc_result.insert(0, c, None)
                extra.append("G")
                continue

            prc_result.insert(0, feature, None)  # add empty column
            if feature in dataset.columns.tolist():  # iterate over enabled features
                if (
                    encode
                    and not (feature in self.id)
                    and (
                        dataset[feature].dtype == pd.StringDtype()
                        or dataset[feature].dtype == object
                    )
                ):
                    dummies = pd.get_dummies(
                        dataset[feature], dtype=pd.Int32Dtype(), prefix=feature
                    )
                    prc_result = prc_result.drop(feature, axis=1)
                    prc_result = pd.concat([prc_result, dummies], axis=1)
                else:
                    prc_result[feature] = dataset[feature]
            else:
                extra.append(feature)
                prc_result[feature] = 0

            if targe_feature:
                prc_result[targe_feature] = self.label_e.fit_transform(
                    dataset[targe_feature]
                )
            extra.sort()

        # Batch elaboration
        batch_cc = 0
        t = tqdm(
            desc="batch compute", total=len(dataset)
        )
        for batch in batch_generator(dataset, batch_size=batch_s):  # type: ignore
            batch_cc += 1
            t.set_postfix({"batch": batch_cc})
            original_batch_len = len(batch)

            ###############################
            # extract additional features #
            ###############################

            for feature in extra:

                t.set_description(feature, refresh=True)

                # COUNT_REFERENCES
                if feature == "reference":
                    join_fe = "wiki_name"
                    r = count_references(batch[join_fe], self.conn)

                # DOMINANT_LANGS
                elif feature == "languages":
                    join_fe = "qid"
                    r = dominant_langs(batch[join_fe], self.conn)

                # LANGS_LENGTH
                elif feature == "length_lan":
                    join_fe = "qid"
                    r = langs_length(batch[join_fe], self.conn)

                # G_FACTOR
                elif feature == "G":
                    join_fe = "wiki_name"
                    mask = list(self.sgf)
                    r = G_factor(
                        batch[join_fe], batch["qid"], 3, 15, 150, None, threads=32
                    )

                    for c in mask:
                        d = r.set_index(join_fe)[c].to_dict()
                        delta = prc_result[join_fe].map(d).fillna(0)
                        prc_result.loc[:, c] = prc_result[c].add(delta, fill_value=0)
                    continue

                # NUM_MOD
                elif feature == "n_mod":
                    join_fe = "wiki_name"
                    r = num_mod(batch[join_fe], self.conn)

                # NUM_USERS
                elif feature == "n_visits":
                    join_fe = "wiki_name"
                    r = num_users(
                        batch[join_fe], start_date="20150701", end_date="20240430"
                    )  # interval of almost 10 years

                # NUM_LANGS
                elif feature == "num_langs":
                    join_fe = "qid"
                    r = num_langs(batch[join_fe], self.conn)

                # BACK_LINKS
                elif feature == "back_links":
                    join_fe = "wiki_name"
                    r = back_links(batch[join_fe], self.conn)

                # RELEVANT_WORDS
                elif feature == "relevant_words":
                    join_fe = "wiki_name"
                    r = relevant_words(batch[join_fe], self.conn)

                # PAGE_INTRO
                elif feature == "intro":
                    join_fe = "wiki_name"
                    r = page_intros(batch[join_fe], self.conn)

                # FULL_PAGE
                elif feature == "full_page":
                    join_fe = "wiki_name"
                    r = page_full(batch[join_fe], self.conn)

                else:
                    raise ValueError(f"Label:{feature} not valid")

                ################
                # add features #
                ################

                delta = prc_result[join_fe].map(r)

                if delta.dtype == object:
                    old = "0"
                    fill = ""
                    ty = str

                else:
                    old = 0
                    ty = float
                    fill = 0

                prc_result[feature] = prc_result[feature].astype(ty).replace(old, fill)
                delta = delta.fillna(fill)

                new_feature = (
                    prc_result[feature].replace(old, fill).astype(ty).add(delta)
                    if ty == float
                    else prc_result[feature].astype(ty).replace(old, fill) + delta
                )

                prc_result.loc[:, feature] = new_feature

            # prc_result.to_csv('tmp.csv')
            t.update(original_batch_len)
        t.close()
        return prc_result

    def __save_with_format(self, df: pd.DataFrame, path: PosixPath) -> None:
        if path.suffix == ".csv":
            df.to_csv(path, sep=".", index=False, quoting=1)
        elif path.suffix == ".tsv":
            df.to_csv(path, sep="\t", index=False, quoting=1)
        else:
            raise ValueError(f"File exstension {path.suffix} not supported")
        return

    def __load_with_format(self, path: PosixPath) -> pd.DataFrame:
        if path.suffix == ".csv":
            df = pd.read_csv(path, sep=".")
        elif path.suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            raise ValueError(f"File exstension {path.suffix} not supported")

        return df

    def produce(
        self,
        loader: Loader,
        out_file: path.PosixPath | str | None,
        enable_feature: list[str],
        targe_feature: str | None,
        batch_s: int = 1,
        load: bool = False,
        encoding: bool = False,
    ) -> pd.DataFrame | None:
        """
        Transforms the Cultural dataset into a new augmented version according to the `enable_feature` list.

        Args:
            loader (Loader): The data loader instance used to access the dataset.
            out_file (path.PosixPath | str | None): Path to save the transformed dataset. If None, no file is saved.
            enable_feature (list[str]): List of feature names to be enabled or used during transformation.
            targe_feature (str | None): Optional target feature to be included or predicted. If None, no target is used.
            batch_s (int, optional): Batch size for processing the dataset. Defaults to 1.
            load (bool, optional): Whether to load the dataset from disk if available. Defaults to False.
            encoding (bool, optional): Whether to apply encoding to the dataset features. Defaults to False.

        Returns:
            pd.DataFrame | None: The transformed dataset as a DataFrame, or None if loading/saving is handled externally.
        """

        out_file = None if not out_file else PosixPath(out_file)

        if load and out_file and out_file.exists():
            df = self.__load_with_format(out_file)
            return df

        dataset = loader.get()

        dataset["qid"] = dataset["item"].str.extract(r"(Q\d+)", expand=False)
        dataset["wiki_name"] = (
            dataset["qid"].map(self.__wiki_name(dataset["qid"].to_list())).fillna(0)
        )
       

        # Function that calls back __produce and returns the new dataset
        prc = self.__produce(dataset, enable_feature, targe_feature, batch_s, encoding)
        self.__save_with_format(prc, out_file)

        return prc


if __name__ == "__main__":
    l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", "validation", None)
    d = CU_Dataset_Factory(out_dir=".")
    # print(d.validation.head(10))
    frame = d.produce(
        l,
        "validation_test.tsv",
        batch_s=32,
        enable_feature=["full_page"],
        targe_feature="label",
    )
    print(frame.head(5))
    d = pd.read_csv("validation_test.tsv", sep="\t")
    print(d.columns)
