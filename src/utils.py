import asyncio
import time
import aiohttp
import requests
import networkx as nx
import pandas as pd
import requests
from matplotlib import pyplot as plt
from collections import deque
from bs4 import BeautifulSoup
from Connection import Wiki_high_conn
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from lxml import html

from urllib.parse import quote, unquote


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

def BFS_Links_base(title: str, limit: int, max_depth: int, max_runtime: float = None) -> nx.DiGraph:
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
    queue = deque([(title, max_depth)])
    time_global = time.time()

    while queue:
        base, depth = queue.popleft()

        # Controllo runtime

        # Salta se profondità 0 o già visitato
        if depth <= 0 or G.nodes[base].get('visited', False):
            continue

        # Ottieni link from cache o API
        if base in response_cache:
            links = response_cache[base]
        else:
            # Prepara titolo per API: sostituisci spazi con underscore, lascia comma
            encoded_title = base.replace(" ", "_")
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": encoded_title,
                "prop": "text",
                "format": "json"
            }

            try:
                time_global = time.time()
                response = session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                # Controllo presenza campo 'parse'
                time_current = time.time()
                if  time_current - time_global > max_runtime:

                    print(f'drop high lat connesion:{time_current - time_global}')
                    G.nodes[base]['visited'] = True
                    
                    continue

                


                if "parse" not in data or "text" not in data["parse"] or "*" not in data["parse"]["text"]:
                    print(f"Pagina non parsabile o assente per '{encoded_title}'")
                    #print(data)
                    response_cache[base] = []
                    continue

                html = data["parse"]["text"]["*"]
                soup = BeautifulSoup(html, "lxml")
                content_div = soup.find("div", class_="mw-parser-output")

                links = []
                seen_titles = set()
                if content_div:
                    for a in content_div.find_all("a", href=True):
                        href = a["href"]
                        if href.startswith("/wiki/") and (":" not in href and "#" not in href):
                            title_link = a.get("title")
                            if title_link and title_link not in seen_titles:
                                links.append(title_link)
                                seen_titles.add(title_link)
                                if len(links) >= limit:
                                    break

                response_cache[base] = links

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error su '{base}': {http_err}")
                response_cache[base] = []
                continue
            except requests.exceptions.Timeout:
                print(f"Request timed out for '{base}'.")
                response_cache[base] = []
                continue
            except requests.exceptions.RequestException as err:
                print(f"Request error for '{base}': {err}")
                response_cache[base] = []
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

        # Marca come visitato
        G.nodes[base]['visited'] = True

    return G


async def fetch_and_parse(
    session: aiohttp.ClientSession,
    title: str,
    limit: int = 50
) -> tuple[str, list[str]]:
    """
    Recupera fino a `limit` link interni (namespace principale) da EN-Wikipedia.
    """
    from urllib.parse import quote

    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=links"
        f"&pllimit={limit}&titles={quote(title)}"
    )
    async with session.get(url) as resp:
        resp.raise_for_status()
        data = await resp.json()

    # Prendiamo la lista dei links per la pagina (dovrebbe essere una sola, poiché
    # titles=singolo titolo)
    pages = data.get("query", {}).get("pages", {})
    links: list[str] = []
    for page in pages.values():
        for link in page.get("links", []):
            t = link.get("title", "")
            if ":" not in t:
                links.append(t)
            if len(links) >= limit:
                break
    return title, links


async def get_name_from_wikidata(conn:aiohttp.ClientSession, qid:str):
    # 1) Ottieni il titolo della pagina Wikipedia EN da Wikidata
    wd_url = "https://www.wikidata.org/w/api.php"
    wd_params = {
        "action":     "wbgetentities",
        "ids":        qid,
        "props":      "sitelinks",
        "sitefilter": "enwiki",
        "format":     "json"
    }

    async with conn.get(wd_url, params=wd_params) as wd_resp:
        wd_resp.raise_for_status()
        wd_data =  await wd_resp.json()

    title = (
        wd_data
        .get("entities", {})
        .get(qid, {})
        .get("sitelinks", {})
        .get("enwiki", {})
        .get("title", "")
    )

    if not title:
        raise ValueError(f"Nessun sitelink 'enwiki' trovato per QID {qid}")
    
    return title


async def _BFS_Links_Async(qid:str, limit: int, max_depth: int,
                          max_runtime: float = None, max_concurrent: int = 16) -> nx.DiGraph:
    """
    BFS parallela asincrona su link di Wikipedia.
    """
    
    G = nx.DiGraph()
    async with aiohttp.ClientSession() as session:
        title = await get_name_from_wikidata(session, qid)
    
        G.add_node(title, count=1)

        visited = {title}
        queue = deque([(title, 0)])
        start_time = asyncio.get_event_loop().time()
        tasks = {}  # mapping di Task -> (node, depth)

        while queue or tasks:
            # Pump: sottometti nuovi task fino a max_concurrent
            while queue and len(tasks) < max_concurrent:
                node, depth = queue.popleft()
                if max_runtime and (asyncio.get_event_loop().time() - start_time) > max_runtime:
                    queue.clear()
                    break
                task = asyncio.create_task(fetch_and_parse(session, node, limit))
                tasks[task] = (node, depth)

            if not tasks:
                break

            # Drain: processa i task completati
            done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                node, depth = tasks.pop(task)
                base_title, links = task.result()
                

                # Aggiorna grafo e accoda nuovi link
                for link in links:
                    if not G.has_node(link):
                        G.add_node(link, count=1)
                    else:
                        G.nodes[link]['count'] += 1

                    if not G.has_edge(base_title, link):
                        G.add_edge(base_title, link)

                    if depth + 1 <= max_depth and link not in visited:
                        visited.add(link)
                        queue.append((link, depth + 1))
                    
                    if link in visited:
                        for n in G.neighbors(link):
                            G.nodes[n].setdefault('count', 0)
                            G.nodes[n]['count'] += 1

    return G



def BFS2_Links_Parallel(qid:str, limit: int, max_depth: int,
                       max_runtime: float = None, max_concurrent: int = 16) -> nx.DiGraph:
    """
    Async BFS links search.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # In un loop attivo: crea un nuovo task e usa `await` (valido ad esempio in Jupyter Notebook)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                _BFS_Links_Async( qid,limit, max_depth, max_runtime, max_concurrent)
            )
    except RuntimeError:
        # Nessun loop attivo: possiamo usare asyncio.run normalmente
        return asyncio.run(
            _BFS_Links_Async(qid,limit, max_depth, max_runtime, max_concurrent)
        )



def draw_and_save_graph(G: nx.DiGraph,
                        path: str = "graph.png",
                        figsize: tuple = (12, 8),
                        dpi: int = 300,
                        with_labels: bool = True,
                        node_size: int = 300,
                        font_size: int = 10,
                        layout: str = "spring"):
    """
    Disegna il grafo G e lo salva come immagine.

    Args:
      G            : NetworkX graph (DiGraph o Graph).
      path         : Percorso di salvataggio (es. "graph.png").
      figsize      : Dimensioni della figura in pollici (w, h).
      dpi          : Risoluzione in dots per inch.
      with_labels  : Se True disegna le etichette dei nodi.
      node_size    : Grandezza dei nodi.
      font_size    : Grandezza del font delle etichette.
      layout       : Tipo di layout: "spring", "kamada_kawai", "circular", "shell"…
    """
    # Scegli il layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.random_layout(G)

    # Crea la figura
    plt.figure(figsize=figsize, dpi=dpi)
    # Disegna nodi e archi
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12, alpha=0.6)
    # Disegna etichette
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    plt.axis('off')
    plt.tight_layout()
    # Salva immagine
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Grafo salvato in: {path}")



def batch_generator(df:pd.DataFrame, batch_size:int):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

# Esempio di test
if __name__ == "__main__":
    # Costruisce un grafo diretto
    G = nx.DiGraph()
    start_page = "God"
    conn = Wiki_high_conn()
    limit = 10      # Numero massimo di link per pagina
    max_depth = 10    # Profondità massima
    G = BFS2_Links_Parallel('Q513',  3, 15, 0.50)

    print("G parameter:", G)
    draw_and_save_graph(G, layout='kamada_kawai')
  
