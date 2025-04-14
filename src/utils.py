import pandas as pd
import re
import requests
import networkx as nx
from collections import deque
from random import shuffle
from collections import defaultdict
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


def extract_relevant_links(response, maxlinks):
    # Precompila l'espressione regolare per controllare i caratteri validi nel titolo
    valid_regex = re.compile(r'^[A-Za-z0-9 \-()%]+$')
    
    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    valid_links = []
    seen = set()  # Set per memorizzare i titoli già aggiunti e garantire unicità

    # Prefissi da escludere
    excluded_prefixes = ("File:", "Category:", "Help:", "Portal:", "Special:", "Talk:")

    for page_id, page in pages.items():
        links = page.get("links", [])
        #print(f"Number of Total Links Found: {len(links)}")

        # Mischia i link per garantire casualità
        shuffle(links)

        for link in links:
            title = link.get("title", "")
            
            # Filtro: salta link con namespace o ancore nel titolo
            if ":" in title or "#" in title:
                continue
            
            # Filtro: controlla i caratteri validi
            if not valid_regex.match(title):
                continue
            
            # Filtro: lunghezza minima e titoli banali
            if len(title) < 2 or title.lower() in {'edit', 'citation'}:
                continue
            
            # Filtro: esclude link che cominciano con prefissi non pertinenti
            if title.startswith(excluded_prefixes):
                continue
            
            # Filtro: esclude link esterni (http:// o https://)
            if title.startswith("http://") or title.startswith("https://"):
                continue
            
            # Filtro: esclude link che sono solo numeri
            if title.isdigit():
                continue
            
            # Aggiungi il titolo se non già presente
            if title in seen:
                continue

            seen.add(title)
            valid_links.append(title)

            # Ritorna subito se il numero massimo di link è raggiunto
            if len(valid_links) >= maxlinks:
                #print(f"Number of Parsed Links Found: {len(valid_links)}")
                return valid_links

    #print(f"Number of Parsed Links Found: {len(valid_links)}")
    return valid_links
def BFS_Links(title: str, limit: int, max_depth: int) -> nx.DiGraph:
    """
    Esegue una ricerca BFS (Breadth First Search) sui link di Wikipedia.
    
    Ottimizzazioni:
      - Utilizza una coda per gestire l’iterazione sul grafo.
      - Cache delle pagine già visitate per evitare richieste duplicate.
    """
    # Inizializzazione: 
    # (visited memorizza per ogni pagina se l’abbiamo già interrogato; 
    #  e cache per le risposte API)
    visited = set()
    G =  nx.DiGraph()
    response_cache = {}  # title -> list of links

    # Aggiornamento del nodo di partenza
    if not G.has_node(title):
        G.add_node(title, count=1, visited=False)
    else:
        G.nodes[title]['count'] += 1

    # Coda per il BFS: ogni elemento è una tupla (page_title, current_depth)
    queue = deque()
    queue.append((title, max_depth))

    while queue:
        base, depth = queue.popleft()

        # Salta se la profondità raggiunta è 0 o se abbiamo già visitato la pagina
        if depth <= 0 or base in visited:
            continue

        #print(f"Processing '{base}' at depth {depth}...")
        visited.add(base)

        # Recupera i link: uso della cache per evitare chiamate ripetute
        if base in response_cache:
            links = response_cache[base]
        else:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "links",
                "titles": base,
                "pllimit": "max",
                "plnamespace": "0",  # Solo namespace 0 per evitare link non pertinenti
                "format": "json"
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                links = extract_relevant_links(response, maxlinks=limit)
                response_cache[base] = links  # memorizza la risposta per evitare richieste future
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error su '{base}': {http_err}")
                continue
            except requests.exceptions.Timeout:
                print(f"Request timed out for '{base}'.")
                continue
            except requests.exceptions.RequestException as err:
                print(f"Request error for '{base}': {err}")
                continue
            except ValueError as json_err:
                print(f"JSON parsing error for '{base}': {json_err}")
                continue

        # Aggiorna il grafo con i link trovati
        for link in links:
            if not G.has_node(link):
                G.add_node(link, count=1, visited=False)
            else:
                G.nodes[link]['count'] += 1
            
            # Aggiunge l'arco solo se non esiste già
            if not G.has_edge(base, link):
                G.add_edge(base, link)
            
            # Aggiunge il link alla coda se non è stato ancora visitato
            if link not in visited:
                queue.append((link, depth - 1))
                
        # Segna il nodo come visitato (in-place modification)
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
  
