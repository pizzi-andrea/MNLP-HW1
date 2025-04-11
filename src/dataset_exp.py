import requests_cache
import requests
from typing import Any
import pandas as pd
import os
import pathlib as path
from datasets import load_dataset

def split_list(lst, n):
    """
    Split a list into `n` sublists as evenly as possible.
    
    Args:
        lst (list): The list to split.
        n (int): The number of sublists to create.
        
    Returns:
        list of lists: A list containing `n` sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def extract_entity_id(url: list[str]) -> list[str]:
    """
    Extract Wikidata entity IDs from a list of URLs.
    
    Args:
        url (list[str]): List of Wikidata URLs.
        
    Returns:
        list[str]: List of extracted entity IDs.
    """
    return [l.strip().split("/")[-1] for l in url]


def extract_entity_name(url: list[str]) -> list[str]:
    """
    Extract Wikipedia entity names from a list of URLs.
    
    Args:
        url (list[str]): List of Wikipedia URLs.
        
    Returns:
        list[str]: List of extracted entity names.
    """
    return [l.strip().split("/")[-1].replace("_", " ") for l in url]


class Wiki_high_conn:
    """
    A high-efficiency client for accessing Wikipedia and Wikidata APIs.
    """

    def __init__(self, default_lang: str = 'en') -> None:
        """
        Initialize the connection client and set up caching.
        
        Args:
            default_lang (str): Default language for Wikipedia queries.
        """
        self._default_lang = default_lang
        requests_cache.clear()
        requests_cache.install_cache('wikidata_cache', expire_after=3600)
        requests_cache.install_cache('wikipedia_cache', expire_after=3600)

    def set_lang(self,lang:str) -> None:
        self._default_lang = lang

    def get_wikipedia(self, queries: list[str], params: dict[str, str], lang:str='') -> list[Any] | Any:
        """
        Perform a batch request to Wikipedia's API.
        
        Args:
            queries (list[str]): List of Wikipedia page titles.
            params (dict[str, str]): API parameters.
        
        Returns:
            dict: The JSON response from the Wikipedia API.
        """
        if lang == '':
            lang = self._default_lang
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params['action'] = 'query'
        params['format'] = 'json'
        params['titles'] = '|'.join(queries) if len(queries) > 1 else queries[0]
        try:
            response = requests.get(url, params=params)
            data = response.json()
            response.raise_for_status()
        except requests.HTTPError as err:
            print(f'err: {err}')
            return
        
        return data

    def get_wikidata(self, queries: list[str], params: dict[str, str]) -> list[Any] | Any:
        """
        Perform a batch request to the Wikidata API.
        
        Args:
            queries (list[str]): List of Wikidata entity URLs.
            params (dict[str, str]): API parameters.
        
        Returns:
            dict: The JSON response from the Wikidata API.
        """
        url = "https://www.wikidata.org/w/api.php"
        
        params['format'] = 'json'
        params['titles'] = '|'.join(queries) if len(queries) > 1 else queries[0]
        try:
            response = requests.get(url, params=params)
            data = response.json()
            response.raise_for_status()
        except requests.HTTPError as err:
            print(f'err: {err}')
            return
        
        return data
    
    def clear_cache(self):
        """
        Clear the currently installed cache (if any)
        """
        requests_cache.clear()
        if path.PosixPath('./wikidata_cache.sqlite').exists():
            os.remove("wikidata_cache.sqlite")
        if path.PosixPath('./wikimedia_cache.sqlite').exists:
            os.remove("wikipedia_cache.sqlite")
        
    

def count_references(queries: list[str], conn: Wiki_high_conn) -> dict[str, set[str]]:
    """
    Fectures Exstractor: Given a batch of Wikipedia links, 
    detterminant how many referance have each page in `queries`
        
    Args:
        queries (list[str]): List of Wikipedia entity URLs.
        params (dict[str, str]): API parameters.
        
    Returns:
        dict: The JSON response from the Wikipedia API.
    """
    params = {
        "action": "query",
        "titles": '|'.join(queries) if len(queries) > 1 else queries[0],
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json"
    }
    
    queries = extract_entity_name(queries)
    data = conn.get_wikipedia(queries, params=params)
    pages = data.get("query", {}).get("pages", {})
    num = {}
    for page_id, page in pages.items():
        title = page.get("title", f"page_{page_id}")
        links = page.get("extlinks", [])
        num[title] = len(links)
    return num


def dominant_langs(queries: list[str], conn: Wiki_high_conn) -> dict[str, set[str]]:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines in how many
    of the top 10 Wikimedia languages each page is available.

    Top languages considered: ['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja']
    
    Args:
        queries (list[str]): List of Wikidata entity URLs.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, set[str]]: A `dictionary` mapping entity IDs to sets of available language codes.
    """
   
    result = {}
    out = {}
    dominant = set(['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja'])
    
    ids = extract_entity_id(queries)
    r = conn.get_wikidata(ids, params={
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "sitelinks",
        "format": "json"
    })  # r['entities'] => dict{ id_page: {sitelinks} }
    
    result =r['entities']

    for page in result:
        sl = list(result[page]['sitelinks'].keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        lang_av = dominant.intersection(lg)
        out[page] = lang_av

    return out # TODO: may convert  one numeric value using hash function


def langs_length(queries: list[str], conn: Wiki_high_conn) -> dict[str, set[str]]:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines in how many
    of the top 10 Wikimedia languages each page is available.

    Top languages considered: ['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja']
    
    Args:
        queries (list[str]): List of Wikidata entity URLs.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, float]: A dictionary mapping entity IDs to average word count of available languages.
    """
    
    result = {}
    out = {}
    dominant = set(['enwiki', 'eswiki', 'frwiki', 'dewiki', 'ruwiki', 'zhwiki', 'ptwiki', 'arwiki', 'itwiki', 'jawiki'])
    
    
    # get wikimedia ids
    ids = extract_entity_id(queries)

    # perform query using Wikidata APIs
    r = conn.get_wikidata(ids, params={
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "sitelinks",
        "format": "json"
    })
    
    
    result = r['entities']

    # Collect titles in the dominant languages
    for page in result:
        q = []
        for lang, info in result[page]['sitelinks'].items():
            if lang in dominant:
                title = info["title"]
                q.append((lang.replace('wiki',''),title))
        
        out[page] = q

    # For each page, fetch extracts in the available languages
    word_counts = {}
    pages = {}

    for page, links in out.items():
        for l, link in links:
            
            r = conn.get_wikipedia(link, params={
                "action": "query",             # type of action
                "format": "json",              # response msg format 
                "titles": [link],              # pages id
                "prop": "extracts",            # required property
                "explaintext": True,           # get plaintext
                "redirects": True,             # expand links
                "exsectionformat": "plain"     # plaintext
            }, lang=l) # type: ignore

            pages.update(r['query']['pages'])
        
        total_words = 0
        valid_pages = 0
        
        # for each page write in different language count words
        for page_id in pages.keys():
            extract = pages[page_id].get('extract', '')
            if extract:
                word_count = len(extract.split())
                total_words += word_count
                valid_pages += 1
        
       # compute standard mean 
        if valid_pages > 0:
            mean_word_count = total_words // valid_pages
            word_counts[page] = mean_word_count
        else:
            word_counts[page] = 0  # If no valid pages found, set word count to 0

    return word_counts

# Test module

if __name__ == '__main__':
    link = ['http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947']
    conn = Wiki_high_conn()
    
    print(dominant_langs(link, conn))

    print(extract_entity_name(["https://en.wikipedia.org/wiki/Human"]))
    print(count_references(["https://en.wikipedia.org/wiki/Human"], conn))
    print(langs_length(["http://www.wikidata.org/entity/Q42"], conn))

    conn.clear_cache()
    


class CU_Dataset_Factory:
    def __init__(self) -> None:

        self.train = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['train'].to_pandas()             # type: ignore
        self.validation = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['validation'].to_pandas()   # type: ignore
        pass
    
    def produce(self) -> pd.DataFrame:
        pass
    