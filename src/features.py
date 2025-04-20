import pandas as pd
import networkx as nx
import nltk
import numpy as np
from datasets import load_dataset
from Connection import Wiki_high_conn
from utils import extract_entity_name, extract_entity_id, BFS2_Links_Parallel

def count_references(queries: pd.DataFrame, conn: Wiki_high_conn) -> pd.DataFrame:
    """
    Features Exstractor: Given a batch of Wikipedia links, 
    determines how many references each page has in `queries`
        
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
        
        queries.loc[queries['name'].str.contains(title, case=False, na=False, regex=False), 'reference'] = len(links)
    
    
    return queries

def dominant_langs(queries: pd.DataFrame, conn: Wiki_high_conn) -> pd.DataFrame:
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
    
    
    result:dict =r.get('entities', {})

    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        lang_av = dominant.intersection(lg)
        out[page] = lang_av
        queries.loc[queries['item'].str.contains(page, case=False, na=False), 'languages'] = len(lang_av)
    
    return queries

# wrong description, change it
def langs_length(queries: pd.DataFrame, conn: Wiki_high_conn) -> pd.DataFrame:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines how many words each page 
    of the top 10 Wikimedia languages contains.

    Top languages considered: ['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja']
    
    Args: CHANGE!!!
        queries (list[str]): List of Wikidata entity URLs.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, float]: A dictionary mapping entity IDs to average word count of available languages.
    """
    
    result = {}
    out = {}
    dominant = set(['enwiki', 'eswiki', 'frwiki', 'dewiki', 'ptwiki', 'itwiki'])
    
    
    # Gets Wikimedia ids
    ids = queries['item']
    ids = extract_entity_name(ids)
    
    # Performs query using Wikidata APIs
    r = conn.get_wikidata(ids, params={
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "sitelinks",
        "format": "json"
    })
    
  
    result = r['entities']

    # Collects titles in the dominant languages
    for page in result:
        
        q = []
        for lang, info in result[page]['sitelinks'].items():
            if lang in dominant:
                title = info["title"]
                q.append((lang.replace('wiki',''),title))
        
        out[page] = q

    # For each page, fetch extracts in the available languages
    pages = {}

    for page, links in out.items():
        for l, link in links:
            
            r = conn.get_wikipedia(link, params={
                "action": "query",             # type of action
                "format": "json",              # response msg format 
                "titles": [link],              # pages id
                "prop": "extracts",            # required property
                "explaintext": True,           # get plaintext
                "exintro": True,               # expand links
                "exsectionformat": "plain"     # plaintext
            }, lang=l) # type: ignore
            
            pages.update(r.get('query', {}).get('pages', {}))
        
        total_words = 0
        valid_pages = 0
        # for each page written in a different language, count words
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

    return queries

def G_factor(queries: pd.DataFrame, depth:int, limit:int, time_limit) -> pd.DataFrame:
    
    # For each query in DataFrame
    for q in queries['name']:
        G = BFS2_Links_Parallel(q, limit, depth, time_limit)
        
        # Computes the mean number of occurrences (recurrent nodes)
        total_count = sum(G.nodes[node].get('count', 0) for node in G.nodes)
        avg_count = total_count / G.number_of_nodes() if G.number_of_nodes() else 0
        
        # Converts the graph in a undirected version only once
        UG = G.to_undirected()
        
        # Number of nodes in the graph
        num_nodes = G.number_of_nodes()
        
        # Computes the PageRank only once
        pr = nx.pagerank(G)
        page_rank = pr.get(q, 0)
        # Uses the median of the PageRanks as "G_mean_pr"
        pr_values = list(pr.values())
        mean_pr = np.median(pr_values) if pr_values else 0
        
        # Computes the number of clicks in UG: we iterate directly on the generator
        num_cliques = sum(1 for _ in nx.find_cliques(UG))
        
        # Creates a mask to update the rows containing the name 'q'
        mask = queries['name'].str.contains(q, case=False, regex=False, na=False)
        queries.loc[mask, 'G_mean_pr'] = mean_pr
        queries.loc[mask, 'G_nodes'] = num_nodes
        queries.loc[mask, 'G_num_cliques'] = num_cliques
        queries.loc[mask, 'G_rank'] = page_rank
        queries.loc[mask, 'G_avg'] = avg_count
        
    return queries

if __name__ == '__main__':
    link = ['http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947','http://www.wikidata.org/entity/Q32786', 'http://www.wikidata.org/entity/Q371', 'http://www.wikidata.org/entity/Q3729947']
    conn = Wiki_high_conn()
    dataset_t = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset',)['validation'].to_pandas().loc[90:100]  # type: ignore
    
    #extract_entity_name(["https://en.wikipedia.org/wiki/Human"])
    count_references(dataset_t, conn)
    dominant_langs(dataset_t, conn)
    langs_length(dataset_t, conn)
    G_factor(dataset_t, 5, 10, 0.009)
    #print(dataset_t)
    print(dataset_t[['label','G_mean_pr', 'G_nodes', 'G_diameter', 'G_num_cliques', 'G_rank', 'G']])
    conn.clear_cache()
    dataset_t.to_csv('prova.csv')

    