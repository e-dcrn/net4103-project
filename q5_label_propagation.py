import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import from_networkx
    from torch_geometric.nn import GCNConv
except ImportError:
    print("PyTorch Geometric non installé. Impossible de lancer la Q5.")

def prepare_data_for_classification(G, target_attr='dorm'):
    """Filtre les données invalides et prépare le tenseur PyTorch."""
    valid_nodes = [n for n, d in G.nodes(data=True) if d.get(target_attr, 0) > 0]
    G_sub = G.subgraph(valid_nodes).copy()
    
    if len(G_sub) == 0:
        return None, 0
        
    raw_labels = [G_sub.nodes[n][target_attr] for n in G_sub.nodes()]
    unique_labels = sorted(list(set(raw_labels)))
    label_mapping = {val: i for i, val in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    mapping = {node: i for i, node in enumerate(G_sub.nodes())}
    G_mapped = nx.relabel_nodes(G_sub, mapping)
    
    for n in G_mapped.nodes():
        original_node = list(mapping.keys())[list(mapping.values()).index(n)]
        G_mapped.nodes[n]['x'] = [1.0, float(G_sub.degree(original_node))]
        G_mapped.nodes[n]['y'] = label_mapping[G_sub.nodes[original_node][target_attr]]
        
    data = from_networkx(G_mapped)
    data.x = data.x.float()
    data.y = data.y.long()
    
    return data, num_classes

class GCNClassifier(torch.nn.Module):
    """Architecture GCN inspirée de Kipf & Welling (2016) avec Embeddings."""
    def __init__(self, num_nodes, hidden_channels, out_channels):
        super(GCNClassifier, self).__init__()
        # On crée un profil vectoriel (embedding) apprenable pour chaque noeud
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, edge_index):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def evaluate_fraction(data, num_classes, train_fraction, device):
    """Entraîne et évalue le GNN pour une fraction d'entraînement spécifique."""
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)
    train_size = int(train_fraction * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask

    model = GCNClassifier(data.num_nodes, 32, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        pred = model(data.edge_index).argmax(dim=1)
        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(data.test_mask.sum())
        test_mae = torch.abs(pred[data.test_mask] - data.y[data.test_mask]).float().mean().item()
        
    return test_acc, test_mae

def evaluate_node_classification(G, name, target_attr='dorm'):
    print(f"\n--- Apprentissage de l'attribut : {target_attr} sur {name} ---")
    data, num_classes = prepare_data_for_classification(G, target_attr)
    if data is None:
        print(f"  [!] Pas assez de données valides pour {target_attr}.")
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    missing_fractions = [0.1, 0.2, 0.3]
    accuracies = []
    
    print(f"\n{'Noeuds masqués':<18} | {'Accuracy':<15} | {'MAE':<15}")
    print("-" * 52)
    
    for missing in missing_fractions:
        train_fraction = 1.0 - missing
        
        runs_results = [evaluate_fraction(data, num_classes, train_fraction, device) for _ in range(3)]
        
        acc_runs = [res[0] for res in runs_results]
        mae_runs = [res[1] for res in runs_results]
        
        mean_acc = np.mean(acc_runs)
        mean_mae = np.mean(mae_runs)
        accuracies.append(mean_acc)
        
        print(f"{int(missing*100)}% {'':<14} | {mean_acc*100:.2f}%{'':<8} | {mean_mae:.4f}")
        
    return missing_fractions, accuracies

def run_q5(loaded_graphs):
    print("\n" + "="*50)
    print("====== EXÉCUTION DE LA QUESTION 5 (Label Propagation) ======")
    print("="*50)
    
    attributes_to_test = ['dorm', 'major_index', 'gender']
    
    for name, G in loaded_graphs.items():
        print(f"\n\n{'*'*50}")
        print(f"--- ANALYSE DU GRAPHE : {name} ({G.number_of_nodes()} noeuds) ---")
        print(f"{'*'*50}")
        
        plt.figure(figsize=(10, 6))
        
        for attr in attributes_to_test:
            res = evaluate_node_classification(G, name, attr)
            if res is not None:
                missing_fracs, accs = res
                # On trace en fonction du pourcentage de masquage
                plt.plot(np.array(missing_fracs)*100, np.array(accs)*100, marker='s', linewidth=2, label=f'Attribut: {attr.capitalize()}')
                
        plt.title(f"Label Propagation Accuracy vs. Missing Labels ({name})", fontsize=14)
        plt.xlabel("Pourcentage de nœuds masqués / missing (%)", fontsize=12)
        plt.ylabel("Accuracy sur le sous-ensemble masqué (%)", fontsize=12)
        
        # On force l'axe X à n'afficher que 10, 20 et 30
        plt.xticks([10, 20, 30])
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_name = f"q5_{name.lower()}_label_prop.png"
        plt.savefig(output_name, dpi=300)
        print(f"\n-> Graphique de {name} sauvegardé sous {output_name}")
        plt.show()