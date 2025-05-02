from typing import Any
import requests_cache


class Wiki_high_conn:
    """
    A high-efficiency client to access Wikipedia and Wikidata APIs.
    """

    def __init__(self, default_lang: str = "en") -> None:
        """
        Initializes the connection client and set up caching.

        Args:
            default_lang (str): Default language for Wikipedia queries.
        """
        self._default_lang = default_lang
        requests_cache.clear()
        requests_cache.install_cache(
            "wiki_cache",
            backend="sqlite",
            allowable_methods=("GET",),
            cache_control=True,
        )
        self.session = requests_cache.CachedSession()

    def set_lang(self, lang: str) -> None:
        """
        Sets default request language for Wikipedia API.
        """
        self._default_lang = lang

    def get_wikipedia(
        self, queries: list[str], params: dict[str, str], lang: str = ""
    ) -> dict[str, Any]:
        """
        Performs a batch request to Wikipedia API.

        Args:
            queries (list[str]): List of Wikipedia page titles.
            params (dict[str, str]): API parameters.

        Returns:
            dict: The JSON response from the Wikipedia API.
        """
        if lang == "":
            lang = self._default_lang
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params["format"] = "json"
        params["titles"] = "|".join(queries)

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        

        return data

    def get_wikidata(
        self, queries: list[str], params: dict[str, str]
    ) -> dict[str, Any]:
        """
        Performs a batch request to the Wikidata API.

        Args:
            queries (list[str]): List of Wikidata entity URLs.
            params (dict[str, str]): API parameters.

        Returns:
            dict: The JSON response from the Wikidata API.
        """
        url = "https://www.wikidata.org/w/api.php"

        params["format"] = "json"
        params["ids"] = "|".join(queries)
     
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
            
       

        return data

    def get_wikidata2wikipedia(
        self, queriesId: list[str], feature: str = ""
    ) -> dict[str, str]:
        """
        Returns a map {QID: Wikipedia title} for every specified QID.

        Args:
            queriesId (list[str]): List of Wikidata ID (es. ['Q42', 'Q7259']).
            params (dict[str, str]): API parameters (not used here, kept for compatibility).

        Returns:
            dict[str, str]: QID Map → Wikipedia title (in language self._default_lang).
        """
        p = {
            "action": "wbgetentities",
            "sites": "wikipedia",
            "props": "sitelinks",
            "format": "json",
            "utf8": 1,
        }

        r = self.get_wikidata(queriesId, p).get("entities", {})

        qid_to_title = {}
        for qid, data in r.items():
            sitelinks = data.get("sitelinks", {})
            # swith-case simulation
            if feature == "name":
                lang_key = f"{self._default_lang}wiki"
                title = sitelinks.get(lang_key, {}).get("title", "")
                qid_to_title[qid] = title
                continue
            if feature == "":
                qid_to_title[qid] = data
                continue
        return qid_to_title

    def clear_cache(self):
        """
        Clears the currently installed cache (if any)
        """
        requests_cache.clear()