import requests_cache
import requests
from typing import Any


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
        self.default_lang = default_lang
        requests_cache.clear()
        requests_cache.install_cache('wikidata_cache', expire_after=3600)
        requests_cache.install_cache('wikipedia_cache', expire_after=3600)

    def get_wikipedia(self, queries: list[str], params: dict[str, str]) -> list[Any] | Any:
        """
        Perform a batch request to Wikipedia's API.
        
        Args:
            queries (list[str]): List of Wikipedia page titles.
            params (dict[str, str]): API parameters.
        
        Returns:
            dict: The JSON response from the Wikipedia API.
        """
        url = f"https://{self.default_lang}.wikipedia.org/w/api.php"
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


def dominant_langs(queries: list[str], conn: Wiki_high_conn, batch: int = 1) -> dict[str, set[str]]:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines in how many
    of the top 10 Wikimedia languages each page is available.

    Top languages considered: ['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja']
    
    Args:
        queries (list[str]): List of Wikidata entity URLs.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, set[str]]: A dictionary mapping entity IDs to sets of available language codes.
    """
    batches = split_list(queries, batch)
    result = {}
    out = {}
    dominant = set(['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja'])
    for bt in batches:
        ids = extract_entity_id(bt)
        r = conn.get_wikidata(bt, params={
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": "sitelinks",
            "format": "json"
        })  # r['entities'] => dict{ id_page: {sitelinks} }
        result.update(r['entities'])

    for page in result:
        sl = list(result[page]['sitelinks'].keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        lang_av = dominant.intersection(lg)
        out[page] = lang_av

    return out

if __name__ == '__main__':
    link = ['http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947']
    conn = Wiki_high_conn()
    
    print(dominant_langs(link, conn, batch=10))



    


