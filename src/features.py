from urllib.parse import quote
import pandas as pd
import networkx as nx
import nltk
import numpy as np
import requests
from Connection import Wiki_high_conn
from utils import extract_entity_name, BFS2_Links_Parallel, fetch_and_parse


#################
# Page features #
#################


def is_disambiguation(queries: pd.Series, conn):
    """
    Controls if any title in `queries` is a Wikipedia's disambiguation page.
    """

    titles = queries.to_list()
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops"
    }
    
    response = conn.get_wikipedia(titles,params=params)
    pages = response.get("query", {}).get("pages", {})

    results = {}
    for page in pages.values():
        title = page.get("title", "")
        is_disambig = "disambiguation" in page.get("pageprops", {})
        results[title] = is_disambig
    
    return results


def count_references(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, list[str]]:
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
        dict[str, set[str]]: A dictionary mapping entity IDs to sets of available language codes.
    """
   
    result = {}
    dominant = set(['en', 'es', 'fr', 'de', 'ru', 'zh', 'pt', 'ar', 'it', 'ja'])
    
    r = conn.get_wikidata(queries.to_list(), params={
        "action": "wbgetentities",
        "props": "sitelinks",
        "format": "json"
    })
    
    result:dict =r.get('entities', {})

    f = {}
    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        f[page] = len(dominant.intersection(lg))
    
    return f


def num_langs(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, list[str]]:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines in how many
    languages each page is available.
    
    Args:
        queries (list[str]): List of Wikidata entity URLs.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, set[str]]: A dictionary mapping entity IDs to sets of available language codes.
    """
   
    result = {}
    
    r = conn.get_wikidata(queries.to_list(), params={
        "action": "wbgetentities",
        "props": "sitelinks",
        "format": "json"
    })
    
    result:dict =r.get('entities', {})

    r = {}
    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        
        r[page] = len(lg)
    
    return r


def langs_length(queries: pd.DataFrame, conn: Wiki_high_conn) -> pd.DataFrame:
    """
    Feature extractor: Given a batch of Wikidata entity links, determines how many words each page 
    of the top 10 Wikimedia languages contains.

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
        # For each page written in a different language, count words
        for page_id in pages.keys():
            extract = pages[page_id].get('extract', '')
            if extract:
                word_count = len(nltk.word_tokenize(extract))
                total_words += word_count
                valid_pages += 1
        
       # Compute standard mean 
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
    """
    Feature extractor: Given a batch of Wikipedia entity links, computes various network metrics.

    Args:
        titles (pd.Series): List of Wikipedia page titles.
        qids (pd.Series): List of Wikidata entity IDs.
        limit (int): Maximum number of nodes to consider.
        depth (int): Depth for BFS traversal.
        max_nodes (int): Maximum number of nodes to fetch.
        time_limit (float | None, optional): Time limit for BFS traversal. Defaults to None.
        threads (int, optional): Number of threads for parallel processing. Defaults to 16.

    Returns:
        pd.DataFrame: A DataFrame containing various network metrics for each title.
    """

    # Initialize columns for raw metrics
    raw_cols = [
        'G_mean_pr', 'G_nodes', 'G_num_cliques', 'G_avg',
        'G_num_components', 'G_largest_component_size','G_density'
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
 

        # Cliques count
        raw_cliques = sum(1 for _ in nx.find_cliques(UG))

        # Component features
        components = list(nx.connected_components(UG))
        component_sizes = [len(c) for c in components]
        largest_component_size = max(component_sizes)
        density = nx.density(UG)

        # Assign metrics
        r['G_mean_pr'] = mean_pager
        r['G_nodes'] = G.number_of_nodes()
        r['G_num_cliques'] = raw_cliques
        r['G_avg'] = avg_count
        r['G_num_components'] = len(components)
        r['G_largest_component_size'] = largest_component_size
        r['G_density'] = density
        fe[q] = r.copy()
    
    # Normalize selected metrics  
  

    fe = pd.DataFrame(fe).transpose()
    fe = fe.reset_index().rename({'index':'wiki_name'}, axis=1)
    
    return fe


def back_links(queries: pd.Series, conn:Wiki_high_conn) -> dict[str, int]:
    """
    Feature extractor: Given a batch of Wikipedia entity links, returns the number of backlinks for each page.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.

    Returns:
        dict[str, int]: A dictionary mapping Wikipedia page titles to their backlink counts.
    """
    
    # The function uses the MediaWiki API to fetch backlinks for each title
    # The API may have rate limits, so we use a delay between requests if processing a large number of titles
    
    # Obtain Wikipedia's titles from Wikidata's entities
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
    return r

####################################
# Features for Transformers models #
####################################

def page_intros(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, str]:
    """
    Feature extractor: Given a batch of Wikipedia page titles, returns the intro extract for each page.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.

    Returns:
        dict[str, str]: A dictionary mapping each Wikipedia page title to its introductory extract (plain text).
    """
    
    # Parametri per ottenere l'estratto introduttivo in testo semplice
    base_params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,      # solo la parte introduttiva
        "explaintext": True,  # senza markup HTML/Wiki
    }
    data = conn.get_wikipedia(queries.to_list(), base_params)
    pages = data.get("query", {}).get("pages", {})
    r = {}
    for page in pages.values():
        title = page.get("title", "")
        r[title] = page.get("extract", "")

    return r
def relevant_words(queries: pd.Series, conn) -> dict[str, list[str]]:
    """
    For each query, return the list of linked page titles,
    skipping links with bad prefixes.
    """
    skip_prefixes = (
        "List of", "Outline of", "Index of", "History of",
        "Category:", "Template:", "User:", "Help:", "Portal:", "Module:", "Wikipedia:", ":" 
    )

    page_links = {}

    wd_params = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "pllimit": "max"  # get max links per request
    }

    titles = queries.to_list()
    r = conn.get_wikipedia(titles, wd_params)

    while True:
        pages = r.get("query", {}).get("pages", {})
        for _, data in pages.items():
            page_title = data.get("title", "UNKNOWN_PAGE")
            links = data.get("links", [])
            valid_links = []
            for link in links:
                linked_title = link["title"]
                if any(linked_title.startswith(prefix) for prefix in skip_prefixes):
                    continue
                valid_links.append(linked_title)
            
            if page_title not in page_links:
                page_links[page_title] = []
            page_links[page_title].extend(valid_links)

        if 'continue' in r:
            wd_params.update(r['continue'])
            r = conn.get_wikipedia(titles, wd_params)
        else:
            r = {}
            for k, v in page_links.items():
                r[k] = ','.join(v)

            break

    return r


###################
# Users Features #
###################


# Number of users who visited a page
def num_users(queries: pd.Series) -> dict[str, int]:
    return {}


# Number of edits of a page
def num_mod(queries:pd.Series, conn:Wiki_high_conn) -> dict[str, int]:
    """
    Feature extractor: Given a batch of Wikipedia page titles, returns the number of unique users who have edited each page.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.
        batch (int, optional): Batch size for API requests. Defaults to 1.
    
    Returns:
        dict[str, int]: A dictionary mapping Wikipedia page titles to their unique user edit counts.
    """

    # The function uses the MediaWiki API to fetch revision history for each title
    # The API may have rate limits, so we use a delay between requests if processing a large number of titles

    result = {}
    for title in queries.tolist():
        users = set()

        # Build base parameters for this page
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "user",    # only take the user
            "rvlimit": "500",    
            "titles": title
        }

        while True:
            # Execute the call
            response = conn.get_wikipedia([title], params)
            data = response.get("query", {})
            pages = data.get("pages", {})

            # Gather users from all revisions in this batch
            for page in pages.values():
                for rev in page.get("revisions", []):
                    if "user" in rev:
                        users.add(rev["user"])

            # If continuity token exists, update it and repeat
            if "continue" in response:
                params.update(response["continue"])
            else:
                break

        # Save the count of unique users
        result[title] = len(users)

    return result
    

if __name__ == '__main__':
    df = pd.DataFrame({'wiki_name':['Rome', 'London', 'A', 'python'], 'qid': ['Q220', 'Q2', 'Q234', 'Q28865']})
   
    conn = Wiki_high_conn()

    #ref = count_references(df['wiki_name'], conn)

    #df['ref'] = df['wiki_name'].map(ref).fillna(0)
    #dom = dominant_langs(df['qid'], conn)
    #print(dom)
    #df['lang'] = df['qid'].map(dom).fillna(0)
    #g = G_factor(df['wiki_name'], df['qid'], 10, 50, 50, 300, 1)
    #c = back_links(df['wiki_name'], conn)
    #dis = is_disambiguation(df['wiki_name'], conn)
    #print(num_mod(df['wiki_name'], conn))
    #print(relevant_words(df['wiki_name'], conn))
    print(page_intros(df['wiki_name'], conn))
