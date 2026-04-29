import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    print("Veuillez installer scikit-learn : pip install scikit-learn")

def evaluate_communities(G, name):
    print(f"\n--- Détection de communautés sur {name} ---")
    
    # 1. Exécution de l'algorithme de Louvain (Maximisation de la modularité)
    print("  -> Exécution de l'algorithme de Louvain...")
    # Retourne une liste de sets (chaque set contient les noeuds d'une communauté)
    try:
        louvain_partition = list(nx.community.louvain_communities(G))
    except AttributeError:
        # Fallback pour d'anciennes versions de NetworkX
        louvain_partition = list(nx.community.greedy_modularity_communities(G))
        
    print(f"  -> {len(louvain_partition)} communautés détectées.")

    # Création d'un dictionnaire {node_id : community_id}
    pred_mapping = {}
    for comm_id, comm in enumerate(louvain_partition):
        for node in comm:
            pred_mapping[node] = comm_id

    # 2. Comparaison avec la vérité terrain (NMI)
    attributes = ['dorm', 'year', 'student_fac', 'major_index', 'gender']
    results = {}

    for attr in attributes:
        # On ne garde que les noeuds qui ont une valeur valide pour cet attribut
        valid_nodes = [n for n in G.nodes() if G.nodes[n].get(attr, 0) > 0]
        if not valid_nodes:
            results[attr] = 0.0
            continue

        y_true = [G.nodes[n][attr] for n in valid_nodes]
        y_pred = [pred_mapping[n] for n in valid_nodes]

        # Le NMI mesure la correspondance entre les 2 partitions (0 = aucune, 1 = parfaite)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        results[attr] = nmi
        print(f"  -> NMI avec l'attribut '{attr}': {nmi:.4f}")

    return results

def run_q6(loaded_graphs):
    print("\n" + "="*50)
    print("====== EXÉCUTION DE LA QUESTION 6 (Communities) ======")
    print("="*50)

    all_results = {}
    for name, G in loaded_graphs.items():
        all_results[name] = evaluate_communities(G, name)

    # --- Génération du graphique récapitulatif ---
    print("\nGénération du graphique comparatif...")
    
    labels = ['Dorm', 'Year', 'Status', 'Major', 'Gender']
    attributes = ['dorm', 'year', 'student_fac', 'major_index', 'gender']
    
    x = np.arange(len(labels))
    width = 0.35  # Largeur des barres

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # On suppose qu'on passe au moins Caltech et MIT
    names = list(all_results.keys())
    if len(names) >= 1:
        scores_1 = [all_results[names[0]].get(attr, 0) for attr in attributes]
        rects1 = ax.bar(x - width/2, scores_1, width, label=names[0], color='tab:blue')
    if len(names) >= 2:
        scores_2 = [all_results[names[1]].get(attr, 0) for attr in attributes]
        rects2 = ax.bar(x + width/2, scores_2, width, label=names[1], color='tab:orange')

    ax.set_ylabel('Normalized Mutual Information (NMI)', fontsize=12)
    ax.set_title('Correspondance entre Communautés Topologiques et Attributs Réels', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_name = "q6_communities_nmi.png"
    plt.savefig(output_name, dpi=300)
    print(f"-> Graphique sauvegardé sous {output_name}")
    plt.show()