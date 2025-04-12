import pandas as pd
import networkx as nx
import nltk
from datasets import load_dataset
from Connection import Wiki_high_conn
from utils import extract_entity_name, extract_entity_id, BFS_Links

def count_references(queries: pd.DataFrame, conn: Wiki_high_conn) -> pd.DataFrame:
    """
    Fectures Exstractor: Given a batch of Wikipedia links, 
    detterminant how many referance have each page in `queries`
        
    Args:
        queries (list[str]): List of Wikipedia entity URLs.
        params (dict[str, str]): API parameters.
        
    Returns:
        dict: The JSON response from the Wikipedia API.
    """

    ids = queries['name']
    ids = extract_entity_name(ids)
    params = {
        "action": "query",
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json"
    }
    
    data = conn.get_wikipedia(ids, params=params)
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        title = page.get("title", f"page_{page_id}")
        links = page.get("extlinks", [])
        queries.loc[queries['name'].str.contains(title, case=False, na=False), 'reference'] = len(links)
    
    
    return queries


def dominant_langs(queries: pd.DataFrame, conn: Wiki_high_conn) -> dict[str, set[str]]:
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
    
    ids = queries['item']
    ids = extract_entity_id(ids)
    
    r = conn.get_wikidata(ids, params={
        "action": "wbgetentities",
        "props": "sitelinks",
        "ids": "|".join(ids),
        "format": "json"
    })  # r['entities'] => dict{ id_page: {sitelinks} }
    
    
    result =r.get('entities', {})

    for page in result:
        
        sl = list(result[page]['sitelinks'].keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        lang_av = dominant.intersection(lg)
        out[page] = lang_av
        queries.loc[queries['item'].str.contains(page, case=False, na=False), 'languages'] = len(lang_av)
    
    return out


def langs_length(queries: pd.DataFrame, conn: Wiki_high_conn) -> dict[str, set[str]]:
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
    dominant = set(['enwiki', 'eswiki', 'frwiki', 'dewiki', 'ptwiki', 'itwiki'])
    
    
    # get wikimedia ids
    ids = queries['item']
    ids = extract_entity_name(ids)
    
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
            
            pages.update(r.get('query', {}).get('pages', {}))
        
        total_words = 0
        valid_pages = 0
        
        # for each page write in different language count words
        for page_id in pages.keys():
            extract = pages[page_id].get('extract', '')
            if extract:
                word_count = len(nltk.word_tokenize(extract))
                total_words += word_count
                valid_pages += 1
        
       # compute standard mean 
        if valid_pages > 0:
            mean_word_count = total_words //valid_pages
            
            queries.loc[queries['item'].str.contains(page, case=False, na=False), 'length_lan'] = mean_word_count
        else:
           queries.loc[queries['item'].str.contains(page, case=False, na=False), 'length_lan'] = 0  # If no valid pages found, set word count to 0

    return word_counts

def G_factor(queries: pd.DataFrame, conn:Wiki_high_conn) -> dict[str,float]:
    
    r = {}
    for q in queries['name']:
        g  = BFS_Links(q, 10, 2)
        queries.loc[queries['name'].str.contains(q, case=False, na=False), 'G'] = g
        r[q] = g
    
    return r
        


if __name__ == '__main__':
    link = ['http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947']
    conn = Wiki_high_conn()
    dataset_t = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['validation'].to_pandas().loc[50:56]  # type: ignore
    
    #extract_entity_name(["https://en.wikipedia.org/wiki/Human"])
    count_references(dataset_t, conn)
    dominant_langs(dataset_t, conn)
    langs_length(dataset_t, conn)
    G_factor(dataset_t, conn)
    print(dataset_t)
    print(dataset_t['label'])
    conn.clear_cache()
    dataset_t.to_csv('prova.csv')

    