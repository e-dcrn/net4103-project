import argparse
from data_loader import get_all_networks, load_network, get_largest_connected_component

import q2_analysis
import q3_assortativity
import q4_link_prediction
import q5_label_propagation
import q6_communities

def main():
    parser = argparse.ArgumentParser(description="Orchestrateur pour le devoir NET4103")
    parser.add_argument("--q2", action="store_true", help="Exécuter la Question 2")
    parser.add_argument("--q3", action="store_true", help="Exécuter la Question 3")
    parser.add_argument("--q4", action="store_true", help="Exécuter la Question 4")
    parser.add_argument("--q5", action="store_true", help="Exécuter la Question 5")
    parser.add_argument("--q6", action="store_true", help="Exécuter la Question 6")
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les questions")

    args = parser.parse_args()

    # --- Logique pour la Question 2  ---
    if args.q2 or args.all:
        networks_q2 = {
            "Caltech": "Caltech36",
            "MIT": "MIT8",
            "Johns Hopkins": "Johns Hopkins55"
        }
        
        graphs = {}
        for name, file in networks_q2.items():
            G_raw = load_network(file)
            graphs[name] = get_largest_connected_component(G_raw)
        
        q2_analysis.run_q2(graphs)

    if args.q3 or args.all:
        all_network_names = get_all_networks()
        q3_assortativity.run_q3(all_network_names, load_network)

    if args.q4 or args.all:
        graphs_to_test = {}
        
        # 1. On charge Caltech (petit, rapide pour tester ton code)
        print("Chargement des données de Caltech...")
        graphs_to_test["Caltech"] = get_largest_connected_component(load_network("Caltech36"))
        
        # 2. Pour tester le MIT
        print("Chargement des données du MIT...")
        graphs_to_test["MIT"] = get_largest_connected_component(load_network("MIT8"))
        
        q4_link_prediction.run_q4(graphs_to_test)

    if args.q5 or args.all:
        graphs_to_test = {}
        print("Chargement des données de Caltech pour la Q5...")
        graphs_to_test["Caltech"] = get_largest_connected_component(load_network("Caltech36"))
        
        print("Chargement des données du MIT...")
        graphs_to_test["MIT"] = get_largest_connected_component(load_network("MIT8"))

        q5_label_propagation.run_q5(graphs_to_test)

    # --- Logique pour la Question 6 ---
    if args.q6 or args.all:
        graphs_to_test = {}
        print("Chargement des données pour la Q6...")
        graphs_to_test["Caltech"] = get_largest_connected_component(load_network("Caltech36"))
        graphs_to_test["MIT"] = get_largest_connected_component(load_network("MIT8"))
        
        q6_communities.run_q6(graphs_to_test)

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()