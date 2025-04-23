import pandas as pd
import networkx as nx
import nltk
import numpy as np
import requests
from Connection import Wiki_high_conn
from utils import extract_entity_name, BFS2_Links_Parallel

#################
# Page features #
#################

def is_disambiguation(queries:pd.Series, conn:Wiki_high_conn):
    """
    Controlla se una singola pagina Wikipedia è una pagina di disambiguazione.
    """
    titles = queries.to_list()
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops"
    }
    r = {}
    response = conn.get_wikipedia(titles, params=params)
    pages = response["query"]["pages"]
    for page in pages.values():
        r[page.get('title', '')] = "disambiguation" in page.get("pageprops", {})
    
    return r
            

#OK
def count_references(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, list[str]]: # SERVE REVISIONE SUL NOME UTILIZZATI PER LA JOINT
    """
    Features Exstractor: Given a batch of Wikipedia links, 
    determines how many references each page has in `queries`
        
    Args:
        queries (pd.Dataframe):  Wikipedia entity URLs.
        params (dict[str, str]): API parameters.
        
    Returns:
        dict: The JSON response from the Wikipedia API.
    """

    params = {
        "action": "query",
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json"
    }
    
    data = conn.get_wikipedia(queries.to_list(), params=params)
    r = {}
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        title = page.get("title", f"page_{page_id}")
        links = page.get("extlinks", [])
        r[title] = len(links)

    return r
#OK
def dominant_langs(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, list[str]]:
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
    
    r = conn.get_wikidata(queries.to_list(), params={
        "action": "wbgetentities",
        "props": "sitelinks",
        "format": "json"
    })  # r['entities'] => dict{ id_page: {sitelinks} }
    
    
    result:dict =r.get('entities', {})

    r = {}
    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        lang_av = dominant.intersection(lg)
        r[page] = len(lang_av)
    
    return r

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

####################
# Network features # 
####################

def G_factor(titles: pd.Series,qids: pd.Series, limit: int, depth: int, max_nodes: int, time_limit: float | None = None, threads: int = 16) -> pd.DataFrame:
    
    # Initialize columns for raw metrics
    raw_cols = [
        'G_mean_pr', 'G_nodes', 'G_num_cliques', 'G_avg',
        'G_num_components', 'G_largest_component_size',
        'G_largest_component_ratio', 'G_avg_component_size',
        'G_isolated_nodes', 'G_density'
    ]

    fe = {}
    r = {}
    

    # Compute raw metrics per query
    for q, qid in zip(titles, qids):
        for col in raw_cols:
            r[col] = 0.0

        try:
            G = BFS2_Links_Parallel(qid, limit, depth, max_nodes, time_limit, threads)
        except requests.HTTPError as err:
            print(f"HTTP error for {q}: {err}")
            continue

        if G.number_of_nodes() == 0:
            continue

        # Mean occurrences
        total_count = sum(G.nodes[n].get('count', 0) for n in G.nodes)
        avg_count = total_count / G.number_of_nodes()

        # PageRank
        pr = nx.pagerank(G)
        pr_values = list(pr.values())
        mean_pager = np.median(pr_values) if pr_values else 0.0

        # Undirected graph
        UG = G.to_undirected()
        num_nodes = UG.number_of_nodes()

        # Clique count
        raw_cliques = sum(1 for _ in nx.find_cliques(UG))

        # Component features
        components = list(nx.connected_components(UG))
        component_sizes = [len(c) for c in components]
        largest_component_size = max(component_sizes)
        largest_component_ratio = largest_component_size / num_nodes
        avg_component_size = np.mean(component_sizes)
        isolated_nodes = sum(1 for n in UG.nodes if UG.degree[n] == 0)
        density = nx.density(UG)

        # Assign metrics
        
        r['G_mean_pr'] = mean_pager
        r['G_nodes'] = num_nodes
        r['G_num_cliques'] = raw_cliques
        r['G_avg'] = avg_count

        r['G_num_components'] = len(components)
        r['G_largest_component_size'] = largest_component_size
        r['G_largest_component_ratio'] = largest_component_ratio
        r['G_avg_component_size'] = avg_component_size
        r['G_isolated_nodes'] = isolated_nodes
        r['G_density'] = density
        fe[q] = r.copy()
    # Normalize selected metrics
        
    to_norm = [
        'G_nodes', 'G_num_cliques', 'G_avg',
        'G_num_components', 'G_largest_component_size',
        'G_avg_component_size', 'G_isolated_nodes', 'G_density'
    ]
    

    fe = pd.DataFrame(fe).transpose()
    
    fe.insert(0,'wiki_name', fe.index)
    fe = fe.reset_index()
 
    for col in to_norm:
        min_val = fe[col].min()
        max_val = fe[col].max()
        if max_val > min_val:
            fe[col] = (fe[col] - min_val) / (max_val - min_val)
        else:
            fe[col] = 0.0
    return fe


def back_links(queries: pd.Series, conn:Wiki_high_conn) -> dict[str, int]:
    
   
    
    # Ottieni titoli Wikipedia dalle entità Wikidata
    r = {} 
    for title in queries:
        r[title] = 0
        PARAMS = {
            "action": "query",
            "format": "json",
            "list": "backlinks",
            "bltitle": title,
            "bllimit": "max",
            "blnamespace": 0
            }
        while True:
            data = conn.get_wikipedia(title,PARAMS)
            links = data.get("query", {}).get("backlinks", [])
            r[title] += len(links)
        
            if "continue" in data:
                PARAMS.update(data["continue"])
            else:
                break

    # Aggiungi la colonna con i backlink nel DataFrame
    #queries['backlink_count'] = queries['item'].map(backlinks_count).fillna(0).astype(int)
    return r

###################
# Users Feactures #
###################

def num_users(queries: pd.Series) -> dict[str, int]:
    return {}



def num_mod(queries:pd.Series, conn:Wiki_high_conn) -> dict[str, int]:
    result = {}
    for title in queries.tolist():
        users = set()

        # Costruiamo i parametri di base per questa pagina
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "user",    # prendo solo lo user
            "rvlimit": "500",    
            "titles": title
        }

        while True:
            # Esegui la chiamata
            response = conn.get_wikipedia([title], params)
            data = response.get("query", {})
            pages = data.get("pages", {})

            # Raccogliamo gli utenti da tutte le revisioni in questo batch
            for page in pages.values():
                for rev in page.get("revisions", []):
                    if "user" in rev:
                        users.add(rev["user"])

            # Se c'è il token di continuazione, aggiornalo e ripeti
            if "continue" in response:
                params.update(response["continue"])
            else:
                break

        # Salvo il conteggio degli utenti unici
        result[title] = len(users)

    return result
    
if __name__ == '__main__':
    df = pd.DataFrame({'wiki_name':['Rome', 'London', 'A', 'python'], 'qid': ['Q220', 'Q2', 'Q234', 'Q28865']})
   
    conn = Wiki_high_conn()

    ref = count_references(df['wiki_name'], conn)

    df['ref'] = df['wiki_name'].map(ref).fillna(0)
    dom = dominant_langs(df['qid'], conn)
    print(dom)
    df['lang'] = df['qid'].map(dom).fillna(0)
    g = G_factor(df['wiki_name'], df['qid'], 10, 50, 50, 300, 1)
    #c = back_links(df['wiki_name'], conn)
    dis = is_disambiguation(df['wiki_name'], conn)
    print(num_mod(df['wiki_name'], conn))
    
    
    

    