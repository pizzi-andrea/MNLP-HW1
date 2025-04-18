import pandas as pd
import re
import requests
import networkx as nx
import time
from collections import deque
from random import shuffle
from collections import defaultdict
from bs4 import BeautifulSoup



def split_list(lst, n):
    """
    Split a list into `n` sublists as evenly as possible.
    
    Args:
        lst (list): The list to split.
        n (int): The number of sublists to create.
        
    Returns:
        list of lists: A list containing `n` sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def extract_entity_id(url: pd.DataFrame) -> list[str]:
    """
    Extract Wikidata entity IDs from a list of URLs.
    
    Args:
        url (list[str]): List of Wikidata URLs.
        
    Returns:
        list[str]: List of extracted entity IDs.
    """
    return [l.strip().split("/")[-1] for l in url]

def extract_entity_name(url:pd.Series) -> list[str]:
    """
    Extract Wikipedia entity names from a list of URLs.
    
    Args:
        url (list[str]): List of Wikipedia URLs.
        
    Returns:
        list[str]: List of extracted entity names.
    """
    return [l.strip().split("/")[-1].replace("_", " ") for l in url]

def BFS_Links(title: str, limit: int, max_depth: int, max_runtime: float = None) -> nx.DiGraph:
    """
    Esegue una ricerca BFS sui link di Wikipedia, a partire dal titolo fornito.
    
    Ottimizzazioni:
      - Utilizza una coda per il BFS.
      - Evita richieste duplicate usando una cache.
      - Controlla il tempo di esecuzione massimo (max_runtime).
    """
    G = nx.DiGraph()
    response_cache = {}
    session = requests.Session()
    # Aggiunge il nodo iniziale
    G.add_node(title, count=1, visited=False)

    queue = deque()
    queue.append((title, max_depth))

    

    while queue:

        base, depth = queue.popleft()

        # Salta se la profondità è 0 o se è già stato visitato
        if depth <= 0 or G.nodes[base].get('visited', False):
            continue

        # Recupera i link
        if base in response_cache:
            links = response_cache[base]
        else:

            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": base,
                "prop": "text",
                "format": "json"
            }
            try:
                start_time = time.time()
                response = session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                html = data["parse"]["text"]["*"]

                # Usa BeautifulSoup per estrarre link visibili
                soup = BeautifulSoup(html, "lxml")  # usa "lxml" per migliori performance
                content_div = soup.find("div", class_="mw-parser-output")

                links = []
                seen_titles = set()

                if content_div:
                    for a in content_div.find_all("a", href=True):
                        href = a["href"]
                        # Filtra i link interni a Wikipedia, escludendo quelli con ":" e "#" (namespace non di primo livello)
                        if href.startswith("/wiki/") and (":" not in href and "#" not in href):
                            title = a.get("title")
                            if title and title not in seen_titles:
                                links.append(title)
                                seen_titles.add(title)
                                if len(links) >= limit:
                                    break
                if time.time() - start_time > 0.8:
                    break
                #print(f'response get in {time.time() - start_time}')
                response_cache[base] = links

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error su '{base}': {http_err}")
                continue
            except requests.exceptions.Timeout:
                print(f"Request timed out for '{base}'.")
                continue
            except requests.exceptions.RequestException as err:
                print(f"Request error for '{base}': {err}")
                continue
            except (ValueError, KeyError) as json_err:
                print(f"JSON parsing error for '{base}': {json_err}")
                continue

        # Aggiunge i link al grafo e alla coda
        for link in links:
            if not G.has_node(link):
                G.add_node(link, count=1, visited=False)
            else:
                G.nodes[link]['count'] += 1

            if not G.has_edge(base, link):
                G.add_edge(base, link)

            if not G.nodes[link]['visited']:
                queue.append((link, depth - 1))

        # Segna come visitato
        G.nodes[base]['visited'] = True
    return G

def batch_generator(df:pd.DataFrame, batch_size:int):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

# Esempio di test
if __name__ == "__main__":
    # Costruisce un grafo diretto
    G = nx.DiGraph()
    start_page = "Caponata"
    limit = 1       # Numero massimo di link per pagina
    max_depth = 100    # Profondità massima
    G = BFS_Links(start_page, limit, max_depth)

    print("G parameter:", G)
  
