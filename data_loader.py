import os
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "fb100" / "data"

def get_all_networks():
    """
    Retourne une liste de tous les noms de réseaux disponibles dans le dossier fb100/data.
    
    Returns:
    --------
    list of str: Noms des fichiers sans l'extension .gml
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Le dossier de données n'existe pas : {DATA_DIR}")
        
    networks = [file.stem for file in DATA_DIR.glob("*.gml")]
    return sorted(networks)

def load_network(name: str) -> nx.Graph:
    """
    Charge le fichier GML d'une université donnée et retourne un objet NetworkX Graph.
    
    Parameters:
    -----------
    name: str
        Le nom du réseau à charger (ex: 'Caltech36', 'MIT8'). Peut inclure ou non '.gml'.
        
    Returns:
    --------
    nx.Graph: Le graphe chargé.
    """
    name = name.replace(".gml", "")
    file_path = DATA_DIR / f"{name}.gml"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier GML est introuvable : {file_path}")
        
    print(f"Chargement du graphe : {name}...")
    G = nx.read_gml(file_path)
    return G

def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Extrait la plus grande composante connexe (LCC) d'un graphe.
    Utile pour la Question 2 notamment.
    
    Parameters:
    -----------
    G: nx.Graph
        Le graphe d'origine.
        
    Returns:
    --------
    nx.Graph: Le sous-graphe correspondant à la LCC.
    """
    if len(G) == 0:
        return G
        
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc_nodes = connected_components[0]
    
    LCC = G.subgraph(largest_cc_nodes).copy()
    return LCC