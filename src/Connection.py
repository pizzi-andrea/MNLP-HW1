import requests_cache
import requests
import os
import pathlib as path
from typing import Any

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
        requests_cache.install_cache('wiki_cache', expire_after=3600, backend='sqlite')
        self.session = requests.Session()
        

    def set_lang(self,lang:str) -> None:
        self._default_lang = lang

    def get_wikipedia(self, queries: list[str], params: dict[str, str], lang:str='') -> dict[str, Any]:
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
            response = self.session.get(url, params=params)
            data = response.json()
            response.raise_for_status()
        except requests.HTTPError as err:
            print(f'err: {err}')
            return {}
        
        return data

    def get_wikidata(self, queries: list[str], params: dict[str, str]) -> dict[str, Any]:
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
        params['titles'] = '|'.join(queries)
        try:
            response = self.session.get(url, params=params)
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
        
