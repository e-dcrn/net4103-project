import random
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# =============================================================================
# PARTIE B : CLASSES DE PRÉDICTION TOPOLOGIQUE
# =============================================================================

class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v):
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")
        
    @abstractmethod
    def predict(self, u, v):
        raise NotImplementedError("Predict must be implemented")

class CommonNeighbors(LinkPrediction):
    def fit(self): pass
    def predict(self, u, v):
        return len(set(self.neighbors(u)).intersection(set(self.neighbors(v))))

class Jaccard(LinkPrediction):
    def fit(self): pass
    def predict(self, u, v):
        nu, nv = set(self.neighbors(u)), set(self.neighbors(v))
        union = len(nu.union(nv))
        return len(nu.intersection(nv)) / union if union > 0 else 0.0

class AdamicAdar(LinkPrediction):
    def fit(self): pass
    def predict(self, u, v):
        common = set(self.neighbors(u)).intersection(set(self.neighbors(v)))
        return sum(1.0 / math.log(self.graph.degree(w)) for w in common if self.graph.degree(w) > 1)


# =============================================================================
# PARTIE C : PROTOCOLE D'ÉVALUATION COMMUNE
# =============================================================================

def train_test_split_edges(G, f=0.1):
    """Retire aléatoirement une fraction f des arêtes pour créer la vérité terrain."""
    G_train = G.copy()
    num_edges_to_remove = int(f * G.number_of_edges())
    all_edges = list(G.edges())
    random.shuffle(all_edges)
    
    edges_removed = set((min(u, v), max(u, v)) for u, v in all_edges[:num_edges_to_remove])
    G_train.remove_edges_from(edges_removed)
    
    return G_train, edges_removed

def calculate_metrics(sorted_predictions, true_edges, k_values):
    """Calcule la Precision@k et le Recall@k."""
    precisions, recalls = [], []
    for k in k_values:
        top_k_pairs = set(sorted_predictions[:k])
        true_positives = len(top_k_pairs.intersection(true_edges))
        
        precisions.append(true_positives / k if k > 0 else 0)
        recalls.append(true_positives / len(true_edges) if len(true_edges) > 0 else 0)
    return precisions, recalls


# =============================================================================
# PARTIE D : ÉVALUATION DES MÉTHODES TOPOLOGIQUES
# =============================================================================

def evaluate_topological(G_train, edges_removed, non_edges, k_values):
    """Évalue CN, Jaccard et AA."""
    predictors = {
        "Common Neighbors": CommonNeighbors(G_train),
        "Jaccard": Jaccard(G_train),
        "Adamic/Adar": AdamicAdar(G_train)
    }
    
    results = {}
    for name, predictor in predictors.items():
        print(f"  -> Calcul topologique : {name}...")
        scores = []
        for u, v in tqdm(non_edges, desc=name, leave=False):
            score = predictor.predict(u, v)
            if score > 0:
                scores.append((score, (u, v)))
        
        # Tri décroissant
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_pairs = [pair for score, pair in scores]
        
        p, r = calculate_metrics(sorted_pairs, edges_removed, k_values)
        results[name] = {"precision": p, "recall": r}
        
    return results


# =============================================================================
# PARTIE E : RÉSEAU DE NEURONES SUR GRAPHES (GNN)
# =============================================================================

def evaluate_gnn(G_train, edges_removed, non_edges, k_values):
    """Évalue un GCN amélioré avec des Embeddings apprenables."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.utils import from_networkx
        from torch_geometric.nn import GCNConv
    except ImportError:
        return {}

    print("  -> Préparation des données pour le GNN...")
    mapping = {node: i for i, node in enumerate(G_train.nodes())}
    G_mapped = nx.relabel_nodes(G_train, mapping)
    data = from_networkx(G_mapped)
    
    # 2. Définition du modèle GCN avec Embedding
    class GCN(torch.nn.Module):
        def __init__(self, num_nodes, hidden_channels):
            super(GCN, self).__init__()
            # On crée un vecteur apprenable (embedding) pour CHAQUE noeud
            self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels)
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        def encode(self, edge_index):
            # Les features d'entrée sont maintenant nos embeddings appris
            x = self.node_emb.weight
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

        def decode(self, z, edge_label_index):
            src, dst = edge_label_index
            return (z[src] * z[dst]).sum(dim=-1)
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_nodes, 64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Entraînement poussé (300 epochs)
    print("  -> Entraînement du GNN (300 epochs)...")
    model.train()
    for epoch in tqdm(range(300), desc="GNN Training", leave=False):
        optimizer.zero_grad()
        z = model.encode(data.edge_index)
        
        pos_edge_index = data.edge_index
        pos_pred = model.decode(z, pos_edge_index)
        
        neg_edge_index = torch.randint(0, data.num_nodes, pos_edge_index.size(), dtype=torch.long, device=device)
        neg_pred = model.decode(z, neg_edge_index)
        
        loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) + \
               F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss.backward()
        optimizer.step()

    print("  -> Génération des prédictions GNN...")
    model.eval()
    with torch.no_grad():
        z = model.encode(data.edge_index)
        scores = []
        for u, v in non_edges:
            mapped_u, mapped_v = mapping[u], mapping[v]
            score = model.decode(z, torch.tensor([[mapped_u], [mapped_v]], device=device)).item()
            scores.append((score, (u, v)))
            
    scores.sort(key=lambda x: x[0], reverse=True)
    sorted_pairs = [pair for score, pair in scores]
    
    p, r = calculate_metrics(sorted_pairs, edges_removed, k_values)
    return {"GNN (GCN + Emb)": {"precision": p, "recall": r}}


# =============================================================================
# ORCHESTRATEUR GLOBAL Q4
# =============================================================================

def plot_combined_results(results_dict, k_values, name):
    """Génère les graphiques finaux comparant D et E."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Évolution des performances : Topologie vs GNN - {name}", fontsize=16)
    
    for pred_name, metrics in results_dict.items():
        linestyle = '-' if 'GNN' not in pred_name else '--'
        linewidth = 2 if 'GNN' not in pred_name else 3
        
        ax1.plot(k_values, metrics["precision"], marker='o', linestyle=linestyle, linewidth=linewidth, label=pred_name)
        ax2.plot(k_values, metrics["recall"], marker='o', linestyle=linestyle, linewidth=linewidth, label=pred_name)
        
    ax1.set_title("Precision@k")
    ax1.set_xlabel("Nombre de prédictions (k)")
    ax1.set_ylabel("Précision")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Recall@k")
    ax2.set_xlabel("Nombre de prédictions (k)")
    ax2.set_ylabel("Rappel")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"q4{name.lower()}.png"
    plt.savefig(output_name, dpi=300)
    print(f"-> Graphique sauvegardé : {output_name}")
    plt.show()

def run_q4(loaded_graphs):
    print("\n" + "="*50)
    print("====== EXÉCUTION DE LA QUESTION 4 (Link Pred) ======")
    print("="*50)
    
    k_values = [50, 100, 200, 300, 400]
    f_values = [0.05, 0.1, 0.15, 0.2] 
    
    for name, G in loaded_graphs.items():
        print(f"\n\n{'*'*50}")
        print(f"--- ANALYSE DU GRAPHE : {name} ({G.number_of_nodes()} noeuds) ---")
        print(f"{'*'*50}")
        
        # Préparation de l'en-tête du tableau récapitulatif
        print("\n=== TABLEAU RÉCAPITULATIF (Précision et Rappel max pour k=400) ===")
        print(f"{'Fraction (f)':<15} | {'Méthode':<25} | {'Précision':<15} | {'Rappel':<15}")
        print("-" * 75)
        
        for f in f_values:
            # 1. Cacher les arêtes selon la fraction f
            G_train, edges_removed = train_test_split_edges(G, f=f)
            
            # 2. Générer les candidats
            non_edges = [(min(u, v), max(u, v)) for u, v in nx.non_edges(G_train)]
            
            all_results = {}
            topo_results = evaluate_topological(G_train, edges_removed, non_edges, k_values)
            all_results.update(topo_results)

            gnn_results = evaluate_gnn(G_train, edges_removed, non_edges, k_values)
            all_results.update(gnn_results)
            
            for pred_name, metrics in all_results.items():
                max_prec = metrics["precision"][-1] # Valeur pour k=400
                max_rec = metrics["recall"][-1]     # Valeur pour k=400
                print(f"{f:<15} | {pred_name:<25} | {max_prec:.4f}{'':<9} | {max_rec:.4f}")
            print("-" * 75)

        print(f"\n-> Génération du graphique comparatif global pour {name} (basé sur le dernier f testé)...")
        plot_combined_results(all_results, k_values, name)