import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    print("Astuce : Installe 'tqdm' (pip install tqdm) pour voir la barre de progression.")
    tqdm = lambda x: x

def calculate_assortativities(G):
    """
    Calcule les assortativités pour un graphe donné.
    Gère les attributs manquants ou uniformes (qui renvoient NaN).
    """
    # Dictionnaire de correspondance : {Nom_Affichage: nom_attribut_gml}
    attributes_map = {
        'Status (Student/Fac)': 'student_fac',
        'Major': 'major_index',
        'Dorm': 'dorm',
        'Gender': 'gender'
    }
    
    results = {}
    
    # (iii) Vertex degree
    try:
        results['Degree'] = nx.degree_assortativity_coefficient(G)
    except Exception:
        results['Degree'] = np.nan
        
    # (i, ii, iiii, iiiii) Attributs de noeuds
    for display_name, attr_name in attributes_map.items():
        try:
            # Calcule l'assortativité d'attribut nominal
            r = nx.attribute_assortativity_coefficient(G, attr_name)
            results[display_name] = r
        except Exception:
            # L'attribut n'existe peut-être pas dans ce graphe spécifique
            results[display_name] = np.nan
            
    return results

def run_q3(network_names, load_func):
    """Fonction principale pour la Question 3 appelée par main.py."""
    print("\n" + "="*40)
    print("====== EXÉCUTION DE LA QUESTION 3 ======")
    print("="*40)
    print(f"Calcul sur {len(network_names)} réseaux en cours... (Prends un café ☕)")

    # Stockage des résultats
    # Clé: Attribut, Valeur: {'sizes': [], 'assorts': []}
    metrics = {
        'Status (Student/Fac)': {'sizes': [], 'assorts': []},
        'Major': {'sizes': [], 'assorts': []},
        'Degree': {'sizes': [], 'assorts': []},
        'Dorm': {'sizes': [], 'assorts': []},
        'Gender': {'sizes': [], 'assorts': []}
    }

    for name in tqdm(network_names):
        # 1. Charger le réseau
        G = load_func(name)
        
        # 2. Traiter comme un graphe simple (suppression des self-loops éventuelles)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        size = len(G)
        if size == 0:
            continue
            
        # 3. Calculer les assortativités
        assortativities = calculate_assortativities(G)
        
        # 4. Stocker les résultats non-NaN
        for attr, value in assortativities.items():
            if value is not None and not np.isnan(value):
                metrics[attr]['sizes'].append(size)
                metrics[attr]['assorts'].append(value)

    print("\nGénération des graphiques...")

    file_names = {
        'Status (Student/Fac)': 'q3_status',
        'Major': 'q3_major',
        'Degree': 'q3_degree',
        'Dorm': 'q3_dorm',
        'Gender': 'q3_gender'
    }
    
    for attr, data in metrics.items():
        sizes = data['sizes']
        assorts = data['assorts']
        
        # On crée une figure pour l'attribut courant (1 ligne, 2 colonnes)
        fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Analyse de l'Assortativité : {attr}", fontsize=14)

        # A. Scatter plot
        ax_scatter.scatter(sizes, assorts, color='tab:blue', alpha=0.6, s=15)
        ax_scatter.set_xscale('log')
        ax_scatter.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_scatter.set_xlabel("Network Size (log)")
        ax_scatter.set_ylabel("Assortativity Coefficient")
        ax_scatter.set_ylim(-0.3, 0.8)
        ax_scatter.grid(True, alpha=0.3)
        
        # B. Histogramme
        ax_hist.hist(assorts, bins=25, color='tab:blue', edgecolor='black', alpha=0.7)
        ax_hist.axvline(0, color='black', linestyle='--', linewidth=1)
        ax_hist.set_xlabel("Assortativity Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_xlim(-0.3, 0.8)
        ax_hist.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 3. Sauvegarde automatique pour le rapport LaTeX
        name_for_file = file_names.get(attr, attr.replace(" ", "_"))
        output_name = f"{name_for_file}.png"
        plt.savefig(output_name, dpi=300)
        print(f"  -> Graphique sauvegardé : {output_name}")
        
        # 4. Affichage
        plt.show()