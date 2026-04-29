# Projet NET4103 : Network Science & Graph Learning
### Analyse du dataset Facebook100 (2005)

Ce projet implémente une analyse complète des réseaux sociaux universitaires à partir du jeu de données **Facebook100**, incluant l'étude topologique, l'assortativité, la prédiction de liens par Deep Learning (GNN) et la détection de communautés.

## Installation et Reproductibilité

Deux méthodes sont disponibles pour exécuter ce projet. La méthode Docker est recommandée pour garantir la compatibilité des versions de PyTorch Geometric.

### Option 1 : Docker (Recommandé)
Assurez-vous que Docker est installé sur votre machine.

1. **Construire l'image :**
   ```bash
   docker build -t projet-facebook100 .
   ```
2. **Exécuter l'ensemble des analyses :**
   ```bash
   docker run --rm -v $(pwd):/app projet-facebook100 python main.py --all
   ```

### Option 2 : Installation Locale (Python 3.12.3)
Il est fortement conseillé d'utiliser un environnement virtuel.

1. **Préparer l'environnement :**
   ```bash
   bash setup.sh
   source env/bin/activate
   ```
2. **Installer manuellement (si le script échoue) :**
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation

Le script `main.py` est l'orchestrateur central. Il utilise des drapeaux (flags) pour lancer chaque question indépendamment :

| Question | Description | Commande |
| :--- | :--- | :--- |
| **Q2** | Analyse topologique (Degrés, Clustering, Densité) | `python main.py --q2` |
| **Q3** | Étude de l'assortativité sur les 100 réseaux | `python main.py --q3` |
| **Q4** | Prédiction de liens (Heuristiques vs GNN) | `python main.py --q4` |
| **Q5** | Propagation de Labels (GCN pour Dorm/Major/Gender) | `python main.py --q5` |
| **Q6** | Détection de communautés (Louvain vs Attributs) | `python main.py --q6` |
| **Toutes**| Exécute l'intégralité du pipeline | `python main.py --all` |

---

## Organisation du Code

L'architecture est modulaire pour faciliter la lecture et le débogage :

- `main.py` : Point d'entrée unique et gestion des arguments.
- `data_loader.py` : Chargement des fichiers `.gml` et extraction de la LCC.
- `q2_analysis.py` : Métriques globales et distributions.
- `q3_assortativity.py` : Calculs de corrélation et histogrammes d'assortativité.
- `q4_link_prediction.py` : Algorithmes Common Neighbors, Jaccard, Adamic/Adar et GCN.
- `q5_label_propagation.py` : Classification semi-supervisée via Graph Convolutional Networks.
- `q6_communities.py` : Algorithme de Louvain et métrique NMI.
- `fb100/data/` : Dossier contenant les 100 fichiers `.gml` (non inclus dans le dépôt par défaut).

---

## 🛠️ Dépendances principales
- `networkx` : Manipulation des graphes.
- `torch` & `torch-geometric` : Réseaux de neurones profonds sur graphes.
- `scikit-learn` : Métriques d'évaluation (NMI, Précision).
- `matplotlib` & `tqdm` : Visualisation et barres de progression.

---
**Auteur :** Ethan Ducournau  
**Date :** Avril 2026  
**Lien Rapport :** Voir `answers.pdf`