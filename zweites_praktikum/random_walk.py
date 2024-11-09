import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scipy.stats import spearmanr
import pandas as pd

def connected_graph_with_n_p(n, p):
    g = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)
    return g

# def minimal_distance_statics(graph):
#     distances = dict(nx.all_pairs_shortest_path_length(graph))
#     distance_list = []
#     for source, target_distances in distances.items():
#         for target, distance in target_distances.items():
#             if source < target:  # Nur Distanzen in einer Richtung hinzufügen
#                 distance_list.append(distance)

#     mean_distance = np.mean(distance_list)
#     std_deviation = np.std(distance_list)
#     median_distance = np.median(distance_list)
#     quartiles = np.percentile(distance_list, [25, 50, 75])

#     return mean_distance, std_deviation, median_distance, quartiles

def random_walk(graph,s):
    n = choice(list(graph.nodes))

    #prefilled dict for later comparison as nx.*centrality dictionaries are complete
    visits = dict.fromkeys(range(0,len(graph.nodes),1),0)

    for i in range(s+1):
        if n not in visits.keys():
            visits[n] = 1
        else:
            visits[n] = visits[n]+1
        n = choice(list(graph.adj[n]))
    #list_of_visits = list(visits.items()) #reverse with dict(list_of_visits)
    return visits


def try_different_n_p_statics(n_values, p_values, s_formula):
    results = []
    for n in n_values:
        for p in p_values:

            graph = connected_graph_with_n_p(n, p)
            
            #s_values = list(range(n, s_formula(n), int( (s_formula(n)-n)/10 ) ) )
            #for s in s_values:
            
            s = s_formula(n)

            #mean_distance, std_deviation, median_distance, quartiles = minimal_distance_statics(graph)

            visits = random_walk(graph, s)
            visits_max = max(visits.values())

            # visits_normalized = {k: v/visits_max for k, v in visits.items()}
            visits_normalized = {k: v/s for k, v in visits.items()}

            degree = nx.degree_centrality(graph)
            #eigen = nx.eigenvector_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            betweenness = nx.betweenness_centrality(graph)
            
            # degree_comparison = 

            cor_spear_degree, p_value = calculate_spearman_correlation(visits_normalized, degree)
            cor_spear_closeness, p_value = calculate_spearman_correlation(visits_normalized, closeness)
            cor_spear_betweenness, p_value = calculate_spearman_correlation(visits_normalized, betweenness)
            
            

            results.append({
                'n': n,
                'p': p,
                's': s,
                'visits': visits,
                'visits_normalized': visits_normalized,
                'degree': degree,
                #'eigen': eigen,
                'closeness' : closeness,
                'betweenness' : betweenness,
                'cor_spear_degree': cor_spear_degree,
                'cor_spear_closeness': cor_spear_closeness,
                'cor_spear_betweenness': cor_spear_betweenness,

                # 'mean_distance': mean_distance,
                # 'std_deviation': std_deviation,
                # 'median_distance': median_distance,
                # '1st_quartile': quartiles[0],
                # '2nd_quartile': quartiles[1],
                # '3rd_quartile': quartiles[2]
            })
            print(
                #f"n: {n}, p: {p} -> Mittelwert der Distanzen: {mean_distance:.2f}, Standardabweichung: {std_deviation:.2f}, Median: {median_distance:.2f}")
                #f"n: {n}, p: {p} -> ")
                f"n: {n}, p: {p}, deg: {cor_spear_degree}, close: {cor_spear_closeness}, between: {cor_spear_betweenness}, ")

    return results

def plot_results(results, n_values, p_values):
    pass

# Quelle ChatGPT
def calculate_spearman_correlation(centrality_1, centrality_2):
    # Ensure both dictionaries have the same nodes
    if set(centrality_1.keys()) != set(centrality_2.keys()):
        raise ValueError("Both dictionaries must contain the same nodes.")
    
    # Sort dictionaries by node keys to ensure consistent order
    sorted_nodes = sorted(centrality_1.keys())
    values_1 = [centrality_1[node] for node in sorted_nodes]
    values_2 = [centrality_2[node] for node in sorted_nodes]
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(values_1, values_2)
    
    return correlation, p_value

# Quelle ChatGPT
def centrality_correlation_table(centrality_measures):
    """
    Computes a Spearman correlation table for multiple centrality measures.

    Args:
        centrality_measures (dict of dict): A dictionary where each key is the name
                                            of a centrality measure, and each value
                                            is a dictionary with node centrality scores.
                                            Example format:
                                            {
                                                'degree': {'A': 0.5, 'B': 0.7, ...},
                                                'betweenness': {'A': 0.2, 'B': 0.3, ...},
                                                ...
                                            }
    Returns:
        pd.DataFrame: A table (DataFrame) with Spearman correlation coefficients for each pair.
    """
    
    # Extract centrality measure names
    centrality_names = list(centrality_measures.keys())
    
    # Initialize an empty DataFrame to store correlation results
    correlation_table = pd.DataFrame(index=centrality_names, columns=centrality_names)

    # Compute Spearman correlation for each pair of centrality measures
    for i, measure_1 in enumerate(centrality_names):
        for j, measure_2 in enumerate(centrality_names):
            # Skip redundant calculations (fill with 1 for diagonal, i.e., self-correlation)
            if i == j:
                correlation_table.loc[measure_1, measure_2] = 1.0
            elif pd.isna(correlation_table.loc[measure_1, measure_2]):  # Avoid duplicate calculations
                # Ensure both centrality dictionaries have the same nodes
                if set(centrality_measures[measure_1].keys()) != set(centrality_measures[measure_2].keys()):
                    raise ValueError("All centrality measures must contain the same nodes.")
                
                # Sort nodes to ensure consistent order
                sorted_nodes = sorted(centrality_measures[measure_1].keys())
                values_1 = [centrality_measures[measure_1][node] for node in sorted_nodes]
                values_2 = [centrality_measures[measure_2][node] for node in sorted_nodes]
                
                # Calculate Spearman correlation
                correlation, _ = spearmanr(values_1, values_2)
                
                # Store the result in both (i, j) and (j, i) to avoid redundant calculations
                correlation_table.loc[measure_1, measure_2] = correlation
                correlation_table.loc[measure_2, measure_1] = correlation

    return correlation_table

# def plot_results(results, n_values, p_values):
#     # Extrahiere Mittelwerte und Standardabweichungen
#     mean_distances = []
#     std_devs = []

#     for p in p_values:
#         p_mean_distances = []
#         p_std_devs = []
#         for result in results:
#             if result['p'] == p:
#                 p_mean_distances.append(result['mean_distance'])
#                 p_std_devs.append(result['std_deviation'])
#         mean_distances.append(p_mean_distances)
#         std_devs.append(p_std_devs)

#     # Erstelle das Diagramm für Mittelwerte der Distanzen
#     plt.figure(figsize=(24, 12))
#     for i, p in enumerate(p_values):
#         plt.plot(n_values, mean_distances[i], label=f'p = {p:.1f}', marker='o')

#     plt.title('Mittelwerte der Distanzen für verschiedene p-Werte')
#     plt.xlabel('Anzahl der Knoten (n)')
#     plt.ylabel('Mittelwert der Distanzen')
#     plt.xticks(n_values)
#     plt.legend()
#     plt.grid()
#     plt.show()

#     # Erstelle das Diagramm für die Standardabweichungen
#     plt.figure(figsize=(24, 12))
#     for i, p in enumerate(p_values):
#         plt.plot(n_values, std_devs[i], label=f'p = {p:.1f}', marker='o')

#     plt.title('Standardabweichungen der Distanzen für verschiedene p-Werte')
#     plt.xlabel('Anzahl der Knoten (n)')
#     plt.ylabel('Standardabweichung der Distanzen')
#     plt.xticks(n_values)
#     plt.legend()
#     plt.grid()
#     plt.show()


# Definiere Werte für n und p
n_values = list(range(1000, 10000, 1000))  # Anzahl der Knoten von 50 bis 1000 in Schritten von 50
#n_values = [1000]

#p_values = [i /20.0 for i in range(1, 21)]  # Werte von 0.1 bis 1.0 in Schritten von 0.1
p_values = np.linspace(0.05, 0.15,10,endpoint=True).tolist()
#p_values = [0.02]


s_formula = lambda n: n * 10

# Führe die Statistiken für unterschiedliche Werte von n und p durch
results = try_different_n_p_statics(n_values, p_values, s_formula)

# Vergleiche die Zentralitätsmaße
# Idee: 


# Plotten der Ergebnisse
plot_results(results, n_values, p_values)