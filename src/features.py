from Connection import Wiki_high_conn
from utils import extract_entity_name, extract_entity_id
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

if __name__ == '__main__':
    link = ['http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947']
    conn = Wiki_high_conn()
    
    print(dominant_langs(link, conn))

    print(extract_entity_name(["https://en.wikipedia.org/wiki/Human"]))
    print(count_references(["https://en.wikipedia.org/wiki/Human"], conn))
    print(langs_length(["http://www.wikidata.org/entity/Q42"], conn))

    conn.clear_cache()