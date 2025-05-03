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


def count_references(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, int]:
    """
    Features Exstractor: Given a batch of Wikipedia links, 
    determines how many references each page has in `queries`
        
    Args:
        queries (pd.Dataframe):  Wikipedia entity URLs.
        params (dict[str, str]): API parameters.
        
    Returns:
        dict (dict[str,str]): The JSON response from the Wikipedia API.
    """

    params = {
        "action": "query",
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json"
    }
    try:
        data = conn.get_wikipedia(queries.to_list(), params=params)
    except requests.HTTPError as err:
        print(err)
        return {}
    r = {}
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        title = page.get("title", f"page_{page_id}")
        links = page.get("extlinks", [])
        r[title] = len(links)

    return r


def dominant_langs(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, int]:
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
    try:
        r = conn.get_wikidata(queries.to_list(), params={
            "action": "wbgetentities",
            "props": "sitelinks",
            "format": "json"
        })
    except requests.HTTPError as err:
        print(err)
        return {}

    
    result:dict =r.get('entities', {})

    f = {}
    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        f[page] = len(dominant.intersection(lg))
    
    return f


def num_langs(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, int]:
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
    try:
        r = conn.get_wikidata(queries.to_list(), params={
            "action": "wbgetentities",
            "props": "sitelinks",
            "format": "json"
        })
    except requests.HTTPError as err:
        print(err)
        return {}
    result:dict =r.get('entities', {})

    r = {}
    for page in result:
        
        sl = list(result[page].get('sitelinks', {}).keys())
        lg = [l.removesuffix('wiki') for l in sl] 
        
        r[page] = len(lg)
    
    return r


import nltk
import pandas as pd

def langs_length(queries: pd.Series, conn) -> dict[str, int]:
    """
    Estrae il numero medio di parole nei primi paragrafi Wikipedia di una voce Wikidata,
    nelle versioni localizzate delle lingue dominanti.

    Args:
        queries (pd.Series): Lista di QID Wikidata (es: 'Q28865')
        conn (Wiki_high_conn): Oggetto con i metodi .get_wikidata e .get_wikipedia

    Returns:
        dict[str, int]: Dizionario {qid: numero_medio_parole}
    """
    nltk.download('punkt', quiet=True)

    qids = queries.tolist()
    dominant = {'eswiki', 'frwiki', 'dewiki', 'ptwiki', 'itwiki', 'enwiki'}
    qid_to_langtitles = {}

    # Query Wikidata per ottenere sitelinks
    response = conn.get_wikidata(qids, params={
        "action": "wbgetentities",
        "props": "sitelinks",
        "format": "json"
    })

    entities = response.get('entities', {})

    # Costruisci mappatura: QID → [(lang, title), ...]
    for qid, data in entities.items():
        lang_titles = []
        for sitelink_key, info in data.get('sitelinks', {}).items():
            if sitelink_key in dominant:
                lang = sitelink_key.replace('wiki', '')
                title = info['title']
                lang_titles.append((lang, title))
        qid_to_langtitles[qid] = lang_titles

    # Ora interroga ogni pagina in ciascuna lingua e calcola word count
    qid_to_avg_words = {}

    for qid, lang_titles in qid_to_langtitles.items():
        total_words = 0
        valid_pages = 0

        for lang, title in lang_titles:
            conn.set_lang(lang)
            r = conn.get_wikipedia([title], params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": True,
                "exintro": True,
                "exsectionformat": "plain"
            })  # type: ignore

            pages = r.get('query', {}).get('pages', {})
            for page_data in pages.values():
                extract = page_data.get('extract', '')
                if extract:
                    word_count = len(nltk.word_tokenize(extract))
                    total_words += word_count
                    valid_pages += 1

        
        qid_to_avg_words[qid] = int(total_words // (valid_pages + 0.1))
        

    return qid_to_avg_words



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

    Parameters:
        'G_mean_pr': the mean page rank
        'G_nodes': the number of nodes of the graph
        'G_num_cliques': the cardinality of a subset of nodes in which every node is connected to every other
        'G_avg': the average number of visits of a node
        'G_density': how many times a page appears in the graph
    """

    # Initialize columns for raw metrics
    raw_cols = ['G_mean_pr', 'G_nodes', 'G_num_cliques', 'G_avg', 'G_density']
    fe = {}

    # Compute raw metrics per query
    for q, qid in zip(titles, qids):
        r = {}
        for col in raw_cols:
            r[col] = 0.0

        try:
            G = BFS2_Links_Parallel(qid, limit, depth, max_nodes, time_limit, threads)
        except requests.HTTPError as err:
            print(f"HTTP error for {q}: {err}")
            continue

        if G.number_of_nodes() == 0:
            continue
        
        #:debug
        #draw_and_save_graph(G, 'g.png', figsize=(12,15), dpi=300, layout='kamada_kawai')
       
        # Mean occurrences
        total_count = sum(G.nodes[n].get('count', 0) for n in G.nodes)
        avg_count = total_count / G.number_of_nodes()

        # PageRank
        pr = nx.pagerank(G)
        pr_values = list(pr.values())
        mean_pager = np.median(pr_values) if pr_values else 0.0

        # Undirected graph
        UG = G.to_undirected()

        # cliques count
        raw_cliques = sum(1 for _ in nx.find_cliques(UG))

        # Graph Density
        density = nx.density(UG)

        # Assign metrics
        r['G_mean_pr'] = mean_pager
        r['G_nodes'] = G.number_of_nodes()
        r['G_num_cliques'] = raw_cliques
        r['G_avg'] = avg_count
        r['G_density'] = density
        fe[q] = r.copy()

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

            try:
                data = conn.get_wikipedia(title,PARAMS)
            except requests.HTTPError as err:
                print(err)
                r[title] = 0
                break
            links = data.get("query", {}).get("backlinks", [])
            r[title] += len(links)
        
            if "continue" in data:
                PARAMS.update(data["continue"])
            else:
                break
    return r


###################
# Users Features #
###################


def num_users(queries: pd.Series, start_date: str, end_date: str) -> dict[str, int]:
    """
    Feature extractor: Given a batch of Wikipedia page titles, returns the number of unique users 
    who have visited each page in a specific interval of time.
    Args:
        queries (pd.Series): List of Wikipedia page titles.
        start_date (str): Start date in YYYYMMDD format.
        end_date (str): End date in YYYYMMDD format.
    Returns:
        dict[str, int]: A dictionary mapping Wikipedia page titles to their unique user visit counts.
    """

    endpoint = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{}/daily/{}/{}"
    result = {}
    headers = {'User-Agent': 'WikipediaViewsBot/1.0 (dani@gmail.com)'}

    for title in queries.tolist():
        title_formatted = title.strip().replace(' ', '_')
        title_encoded = requests.utils.quote(title_formatted, safe='')
        url = endpoint.format(title_encoded, start_date, end_date)
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            views = sum(item['views'] for item in data.get('items', []))
            result[title] = views
        else:
            result[title] = 0  # if there's an error, return 0 visits

    return result


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
            try:
                response = conn.get_wikipedia([title], params)
                data = response.get("query", {})
                pages = data.get("pages", {})
            except requests.HTTPError as err:
                print(err)
                result[title] = []
                break

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
 

####################################
# Features for Transformers Models #
####################################


def page_intros(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, str]:
    """
    Feature extractor: Given a batch of Wikipedia page titles, returns the intro extract for each page,
    handling API pagination via 'continue'.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.

    Returns:
        dict[str, str]: A dictionary mapping each Wikipedia page title to its introductory extract (plain text).
    """
    titles = queries.to_list()
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,       # only the introductory part
        "explaintext": True,   # plain text (no HTML/Wiki markup)
        "exlimit": "max",    # maximum extracts per request
    }

    # Initialize result dict with empty strings
    results = {title: "" for title in titles}

    # Loop until API pagination is complete
    while True:
        try:
            data = conn.get_wikipedia(titles, params)
        except requests.HTTPError as err:
            print(err)

        pages = data.get("query", {}).get("pages", {})

        # Accumulate extracts for each page
        for page in pages.values():
            title = page.get("title", "")
            if "missing" in page:
                # Page does not exist; leave empty
                pages[title] = "[NO_WIKI]"
                continue
            extract = page.get("extract", "")
            # Concatenate new fragment
            results[title] += extract

        # Check for continuation token
        cont = data.get("continue")
        if cont:
            params.update(cont)
        else:
            break

    return results


def page_full(queries: pd.Series, conn: Wiki_high_conn) -> dict[str, str]:
    """
    Extract plain-text of all Wikipedia pages indicated in 'queries',
    automatically dealing with 'continue' blocks of the API.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.

    Returns:
        Dict[str, str]: A dictionary mapping each Wikipedia page title to its wikipage (full text).

    """

    # Join all titles in a single string separated by "|"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
    }
    results: dict[str, str] = {}
    
    while True:

        try:

            data = conn.get_wikipedia(queries.to_list(), params)
        except requests.HTTPError as err:
            print(err)

        pages = data.get("query", {}).get("pages", {})
        
        for page in pages.values():
            title = page.get("title", "")
            if "missing" in page:
                # Page not found
                pages[title] = "[NO_WIKI]"
            else:
                # The 'extract' field is already plain-text
                results[title] =  results.get(title, '') + page.get("extract", "")
        
        # If l'API reports a 'continue', updates params and resend the request
        cont = data.get("continue")
        if cont:
            params.update(cont)
        else:
            break
    
    return results


def relevant_words(queries: pd.Series, conn) -> dict[str, list[str]]:
    """
    For each query, return the list of linked page titles,
    skipping links with bad prefixes.

    Args:
        queries (pd.Series): List of Wikipedia page titles.
        conn (Wiki_high_conn): An active Wiki_high_conn instance.

    Returns:
        Dict[str, str]: A dictionary mapping each Wikipedia page title to its wikipedia links.
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


if __name__ == '__main__':
    df = pd.DataFrame({'wiki_name':['Rome', 'London', 'Flandres (Bélgica)', 'Python (programming language)'], 'qid': ['Q220', 'Q2', 'Q234', 'Q28865']})
   
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
    #print(page_intros(df['wiki_name'], conn))
    #print(num_users(df['wiki_name'], start_date="20150701", end_date="20250430"))
    #print(df.columns)
    print(langs_length(df['qid'], conn))