from typing import Any

import requests
import requests_cache


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
        requests_cache.install_cache(
            'wiki_cache', 
            backend='sqlite',
            allowable_methods=('GET',),
            cache_control=True
        )
        self.session = requests_cache.CachedSession()

    def set_lang(self,lang:str) -> None:
        """
        set default request lang
        """
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
        params['action'] = 'query' if not params.get('action') else params['action']
        params['format'] = 'json'  
        params['titles'] = '|'.join(queries)
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
        params['ids'] = '|'.join(queries)
        try:
            response = self.session.get(url, params=params)
            data = response.json()
            response.raise_for_status()
        except requests.HTTPError as err:
            print(f'err: {err}')
            return
        
        return data
    
    def get_wikidata2wikipedia(self, queriesId: list[str], params:dict[str, str]) -> dict[str, Any]:
        """
        Method that allow get wikipedia pages in batch specificated wikidata ids
        Args:
            queries (list[str]): List of Wikidata entity URLs.
            params (dict[str, str]): API parameters.
        
        Returns:
            dict: The JSON response from the Wikipedia API.
        """
        p = {
        'action': 'wbgetentities',
        'sites': 'wikipedia',  # Indica che vogliamo i link a Wikipedia
        'props': 'sitelinks',  # Chiediamo i sitelinks
        'format': 'json',  # Risposta in formato JSON
        'utf8': 1
        }

        r = self.get_wikidata(queriesId, p).get('entities', {})

        queriesName = []
        for page_id, data in r.items():
            if 'sitelinks' in data:
                lang_link:str = data.get('sitelinks', {}).get(f'{self._default_lang}wiki', {}).get('title', '')
                name = lang_link.split(sep='/')[-1] if len(lang_link) > 0 else ''
                queriesName.append(name)
            else:
                continue
        
        print(queriesName)
        r = self.get_wikipedia(queriesName, params)
        return r

    def clear_cache(self):
        """
        Clear the currently installed cache (if any)
        """
        requests_cache.clear()

if __name__ == '__main__':
    print(f'[Test script for module {__file__}]')
    queris = ['Q1450662','Q178', 'Q223655', 'Q1723884']

    conn = Wiki_high_conn()

    r = conn.get_wikidata2wikipedia(queris, {})
    print(r)