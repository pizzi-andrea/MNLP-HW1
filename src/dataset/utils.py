import requests
from wikidata.client import Client

def extract_entity_id(url:str) -> str:
    return url.strip().split("/")[-1]

class Wiki_Scrapter:
    def __init__(self, wikidata_id:str, default_lang:str='en', wiki_t:str='wiki') -> None:
        
        self.conn:Client = Client()
        self.page = None
        self.wiki_t = wiki_t
        self.sitelinks = None
        try:
            self.page = self.conn.get(wikidata_id, load=True)

        except requests.HTTPError as err:
            print(f'Connection Error {err}')
            exit(-1)
        
        self.id = wikidata_id
        self.lang = default_lang
    
    def _getsite(self) -> None:
        self.sitelinks = self.page.data.get("sitelinks", {}) if not self.sitelinks else self.sitelinks
        return


    def load_languages(self) -> dict[str,str]:

        """
        Gets language wiki page from wikimedia request

        Parameters:
            wikidata_id(str): Wikidata unique resource identificator
            default_lang(str): Default language for page
            wiki_t(str): prefix wikidata project (wikipedia etc...)

        Returns:
            out(dict[str,str]):
                dictionary with form 'lang':'https://{lang}.wikipedia.org/...'
        """
       
        self._getsite() # load wiki links
        links = {}
        # parse and get all links
        for site_key, site_data in self.sitelinks.items():
            if site_key.endswith(self.wiki_t) and not site_key.startswith("commons"):
                lang = site_key.replace("wiki", "")
                title = site_data["title"]
                links[lang] = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                

        return links
    
    def set_lang(self, new_lang:str):
        """
            Sets new default language and updates all internal data

            Parameters:
                lang(str): new language to set
            
        """
        self.lang = new_lang

    def get_title(self) -> str:
        """Returns site title in language set"""
        self._getsite()
        return  self.sitelinks[f'{self.lang}{self.wiki_t}']['title']

    def get_wikidata_links(self) -> list[str]:
        """
            Returns all wikidata links

            Returns:
                out(List[str]):
                    wiki links
        """
        pass
