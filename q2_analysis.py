import networkx as nx
import matplotlib.pyplot as plt

def analyze_part_a(G, name):
    """Calcule et affiche les métriques de la Partie A."""
    density = nx.density(G)
    global_cc = nx.transitivity(G)
    mean_local_cc = nx.average_clustering(G)
    
    print(f"--- Réseau : {name} ---")
    print(f"  Densité des arêtes : {density:.5f}")
    print(f"  Clustering Global  : {global_cc:.5f}")
    print(f"  Clustering Local Moyen : {mean_local_cc:.5f}")
    
    return [d for n, d in G.degree()]

def analyze_part_b(G):
    """Prépare les données pour le nuage de points de la Partie B."""
    degrees = dict(G.degree())
    local_ccs = nx.clustering(G)
    x = [degrees[node] for node in G.nodes()]
    y = [local_ccs[node] for node in G.nodes()]
    return x, y

def run_q2(loaded_graphs):
    """Fonction principale pour la Question 2 appelée par main.py."""
    print("\n" + "="*40)
    print("====== EXÉCUTION DE LA QUESTION 2 ======")
    print("="*40)

    # --- PARTIE A : Distributions et Métriques ---
    print("\n[Partie A] Calcul des métriques et distributions...")
    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 5))
    fig_a.suptitle("Question 2(a) : Distribution des degrés", fontsize=16)

    for i, (name, G) in enumerate(loaded_graphs.items()):
        degrees = analyze_part_a(G, name)
        axes_a[i].hist(degrees, bins=50, color='skyblue', edgecolor='black')
        axes_a[i].set_title(f"{name}")
        axes_a[i].set_xlabel("Degré")
        axes_a[i].set_ylabel("Fréquence")

    plt.tight_layout()
    plt.savefig("q2a.png", dpi=300)
    print("-> Graphique sauvegardé : q2a.png")

    # --- PARTIE B : Degré vs Clustering ---
    print("\n[Partie B] Génération des nuages de points...")
    fig_b, axes_b = plt.subplots(1, 3, figsize=(15, 5))
    fig_b.suptitle("Question 2(b) : Degré vs Clustering Local", fontsize=16)

    for i, (name, G) in enumerate(loaded_graphs.items()):
        x_deg, y_cc = analyze_part_b(G)
        axes_b[i].scatter(x_deg, y_cc, alpha=0.3, color='coral', s=10)
        axes_b[i].set_title(f"{name}")
        axes_b[i].set_xlabel("Degré du noeud")
        axes_b[i].set_ylabel("Clustering Local")

    plt.tight_layout()
    plt.savefig("q2b.png", dpi=300)
    print("-> Graphique sauvegardé : q2b.png")
    plt.show()