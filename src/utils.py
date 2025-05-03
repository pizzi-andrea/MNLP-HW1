import asyncio
import aiohttp
import networkx as nx
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from collections import deque
from Connection import Wiki_high_conn
from collections import deque
from lxml import html
from urllib.parse import quote, unquote


def split_list(lst, n):
    """
    Splits a list into `n` sublists as evenly as possible.
    
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
    Extracts Wikidata entity IDs from a list of URLs.
    
    Args:
        url (list[str]): List of Wikidata URLs.
        
    Returns:
        list[str]: List of extracted entity IDs.
    """

    return [l.strip().split("/")[-1] for l in url]


def extract_entity_name(url:pd.Series) -> list[str]:
    """
    Extracts Wikipedia entity names from a list of URLs.
    
    Args:
        url (list[str]): List of Wikipedia URLs.
        
    Returns:
        list[str]: List of extracted entity names.
    """

    return [l.strip().split("/")[-1].replace("_", " ") for l in url]


async def fetch_and_parse(session: aiohttp.ClientSession, title: str, limit: int):
    """
    Fetches a Wikipedia page, extracts internal links,
    scores them by structural importance, and returns
    the top `limit` titles by score.
    """

    title_encoded = quote(title.replace(" ", "_"), safe="()'!*")
    lang = "en"
    url = f"https://{lang}.wikipedia.org/wiki/{title_encoded}"

    # Retry mechanism (backoff)
    retries = 3
    delay = 1.0
    for attempt in range(retries):
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                body = await resp.read()
            break  # Success
        except (aiohttp.ClientError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError, RuntimeError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff before doing another request
                continue
            else:
                raise RuntimeError(f"Failed to fetch page '{title}'. Error: {e}") from e   

    # Parse HTML and extract main content
    try:
        tree = html.fromstring(body)
        content_div = tree.get_element_by_id('mw-content-text')
        if content_div is None:
            divs = tree.xpath('//div[contains(@class, "mw-parser-output")]')
            content_div = divs[0] if divs else None
        if content_div is None:
            print(f"Warning: Could not find main content for '{title}'.")
            return title, []
    except Exception as e:
        raise RuntimeError(f"Failed to parse HTML for page '{title}'. Error: {e}") from e

    seen = set()
    scored_links = []

    # Extract all candidate anchors inside paragraphs, list items, and table cells
    anchors = content_div.xpath('.//a[starts-with(@href, "/wiki/")]')
    for a in anchors:
        href = a.get('href', '')
        path = href.split('#', 1)[0]
        if not path.startswith('/wiki/'):
            continue
        key = path[len('/wiki/'):]
        if ':' in key or key == 'Main_Page':
            continue
        try:
            name = unquote(key).replace('_', ' ')
        except Exception:
            continue
        if not name.strip() or name in seen:
            continue

        # Structural scoring to links according to position
        score = 1.0
        lineno = getattr(a, 'sourceline', 1) or 1
        score += 1.0 / lineno
        text = a.text_content() or ''
        score += len(text.split()) * 0.1

        # If in a list item, table cell, or header, gets +0.5 bonus
        if a.xpath('ancestor::nav') or a.xpath('ancestor::header'):
            score += 0.5

        if score >= 1.5:
            seen.add(name)
            scored_links.append((name, score))

    scored_links.sort(key=lambda x: x[1], reverse=True)
    ordered = [name for name, _ in scored_links[:limit]]

    return title, ordered


async def get_name_from_wikidata(conn:aiohttp.ClientSession, qid:str):

    # 1) Get the title of the EN Wikipedia page from Wikidata
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

    title = (wd_data.get("entities", {}).get(qid, {}).get("sitelinks", {}).get("enwiki", {}).get("title", ""))

    if not title:
        title = ''
    
    return title


async def _BFS_Links_Async(qid:str, limit: int, max_depth: int, max_nodes:int|None = None, max_runtime: float = None, max_concurrent: int = 16) -> nx.DiGraph:
    """
    Asynchronous parallel BFS on Wikipedia links.
    """
  
    G = nx.DiGraph()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), connector=aiohttp.TCPConnector(limit=10)) as session:
        tasks = {}  # Mapping of Task -> (node, depth)
        title = await get_name_from_wikidata(session, qid)
       
        if title == '':
            return G
        G.add_node(title, count=1)
        visited = {title}
        queue = deque([(title, 0)])
        start_time = asyncio.get_event_loop().time()
        while queue or tasks:
            # Pump: submit new tasks until max_concurrent
            while queue and len(tasks) < max_concurrent:
                node, depth = queue.popleft()
                if max_runtime and (asyncio.get_event_loop().time() - start_time) > max_runtime:
                    queue.clear()
                    break
                task = asyncio.create_task(fetch_and_parse(session, node, limit))
                tasks[task] = (node, depth)

            if not tasks:
                break

            # Drain: processes completed tasks
            done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                node, depth = tasks.pop(task)
                try:
                    base_title, links = task.result()
                except Exception as e:
                    print(f"Errore sul nodo {node}: {e}")
                    continue

                # Updates graph and appends new links
                for link in links:
                    if not G.has_node(link):
                        G.add_node(link, count=1)
                    else:
                        G.nodes[link]['count'] += 1

                    if max_nodes is not None and G.number_of_nodes() >= max_nodes:
                       
                        return G

                    if not G.has_edge(base_title, link):
                        G.add_edge(base_title, link)

                    if depth + 1 <= max_depth and link not in visited:
                        visited.add(link)
                        queue.append((link, depth + 1))

    return G


def BFS2_Links_Parallel(qid:str, limit: int, max_depth: int, max_nodes:int|None = None, max_runtime: float = None, max_concurrent: int = 16) -> nx.DiGraph:
    """
    Synchronous Wrapper compatible with ambients already having an active event loop.
    """

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # In an active loop: creates a new task and uses `await` (valid for instance in Jupyter Notebook)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                _BFS_Links_Async(qid,limit, max_depth, max_nodes, max_runtime, max_concurrent)
            )
        
    except RuntimeError:
        # No active loop: we can also use asyncio.run normally
        return asyncio.run(
            _BFS_Links_Async(qid,limit, max_depth, max_nodes, max_runtime, max_concurrent)
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
    Draws the graph G and saves it as image.

    Args:
      G            : NetworkX graph (DiGraph or Graph).
      path         : Saving path (es. "graph.png").
      figsize      : Dimensions of the figure in inches (w, h).
      dpi          : Resolution in dots per inch.
      with_labels  : If True, draws the nodes' labels.
      node_size    : Size of the nodes.
      font_size    : Size of the labels' font.
      layout       : Type of layout: "spring", "kamada_kawai", "circular", "shell"…
    """
    
    matplotlib.use("Agg")  # usa un backend che salva su file, niente finestre
    # Choose the layout
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

    # Creates the figure
    plt.figure(figsize=figsize, dpi=dpi)
    # Draws nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12, alpha=0.6)
    # Draws labels
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    plt.axis('off')
    plt.tight_layout()
    # Saves image
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Graph saved in: {path}")


def batch_generator(df:pd.DataFrame, batch_size:int):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]


#  Test example
if __name__ == "__main__":
    # Builds a directed graph
    start_page = "Q2"
    conn = Wiki_high_conn()
    limit = 10      # Max number of links per page
    max_depth = 10    # Max depth
    G = BFS2_Links_Parallel(start_page, limit=5, max_depth=50, max_nodes=100, max_concurrent=1)

    print("G parameter:", G)
    draw_and_save_graph(G, layout='kamada_kawai')