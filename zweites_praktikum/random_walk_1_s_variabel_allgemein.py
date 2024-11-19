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


def try_different_n_p_statics(n_values, p_values, s_values):
    results = []
    for n in n_values:
        for p in p_values:
            graph = connected_graph_with_n_p(n, p)
            for s in s_values:
            
                #s_values = list(range(n, s_formula(n), int( (s_formula(n)-n)/10 ) ) )
                #for s in s_values:
                
                #s = s_formula(n)

                #mean_distance, std_deviation, median_distance, quartiles = minimal_distance_statics(graph)

                visits = random_walk(graph, s)
                visits_max = max(visits.values())

                # visits_normalized = {k: v/visits_max for k, v in visits.items()}
                visits_normalized = {k: v/s for k, v in visits.items()}

                degree = nx.degree_centrality(graph)
                eigen = nx.eigenvector_centrality(graph)
                closeness = nx.closeness_centrality(graph)
                betweenness = nx.betweenness_centrality(graph)
                
                # degree_comparison = 

                cor_spear_degree, p_value = calculate_spearman_correlation(visits_normalized, degree)
                cor_spear_closeness, p_value = calculate_spearman_correlation(visits_normalized, closeness)
                cor_spear_betweenness, p_value = calculate_spearman_correlation(visits_normalized, betweenness)
                cor_spear_eigen, p_value = calculate_spearman_correlation(visits_normalized, eigen)
                
                #cor_kentau_degree = 

                results.append({
                    'n': n,
                    'p': p,
                    's': s,
                    's/n' : s/n,
                    'visits': visits,
                    'visits_normalized': visits_normalized,
                    'degree': degree,
                    #'eigen': eigen,
                     'cor_spear_closeness' : cor_spear_closeness,
                     'cor_spear_betweenness' : cor_spear_betweenness,
                    'cor_spear_degree': cor_spear_degree,
                    'cor_spear_eigen': cor_spear_eigen,


                })
                print(
                    #f"n: {n}, p: {p} -> Mittelwert der Distanzen: {mean_distance:.2f}, Standardabweichung: {std_deviation:.2f}, Median: {median_distance:.2f}")
                    #f"n: {n}, p: {p} -> ")
                    # f"n: {n}, p: {p}, s: {s}, walk_vs_deg: {cor_spear_degree}, walk_vs_close: {cor_spear_closeness}, between: {cor_spear_betweenness} ")
                    f"n: {n}, p: {p}, s: {s}, walk_vs_deg: {cor_spear_degree}")#, walk_vs_close: {cor_spear_closeness}, between: {cor_spear_betweenness} ")

    return results

def plot_results(results, n_values, p_values, s_values):
    n_values_full=[]
    cor_degree = []
    cor_between = []
    cor_closeness = []
    cor_eigen = []
    #s_values = []

    for s in s_values:
        for result in results:
            if result['s'] == s:
                cor_degree.append(result['cor_spear_degree'])
                #cor_between.append(result['cor_spear_betweenness'])
                #cor_closeness.append(result['cor_spear_closeness'])
                #cor_eigen.append(result['cor_spear_eigen'])
                
                
                
                # s_values_full.append(n)
                #s_values.append(result['s'])
        

    # Erstelle das Diagramm für Mittelwerte der Distanzen
    plt.figure()
    #for i, s in enumerate(s_values):
    plt.plot(s_values, cor_degree, label=f'Degree', marker='o')
    #plt.plot(s_values, cor_between, label=f'Closeness', marker='o')
    #plt.plot(s_values, cor_closeness, label=f'Betweenness', marker='o')
    #plt.plot(s_values, cor_eigen, label=f'Eigen', marker='o')

    plt.title('Korrelation Walk zu Degree für variables s von n*100 bis n*n')
    plt.xlabel('Walklength s')
    plt.ylabel('Korrelation Walk zu Centrality-Measure')
    plt.yscale('log')
    plt.xticks(s_values)
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("KorrelationWalkZuAllen_variable_s.png")
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




# Definiere Werte für n und p
n_values = list(range(1000, 1000, 1000))  # Anzahl der Knoten von 50 bis 1000 in Schritten von 50
n_values = np.linspace(100, 1000, 10,endpoint=True).tolist()
#n_values = [round(n) for n in n_values]
n_values = [500]

#p_values = [i /20.0 for i in range(1, 21)]  # Werte von 0.1 bis 1.0 in Schritten von 0.1
#p_values = np.linspace(0.05, 0.05, 1,endpoint=True).tolist()
p_values = [0.26]


#s_formula = lambda n: n * 10

s_values = np.linspace(100000, 1000000, 5,endpoint=True).tolist()
s_values = [round(s) for s in s_values]

# Führe die Statistiken für unterschiedliche Werte von n und p durch
results = try_different_n_p_statics(n_values, p_values, s_values)

# Vergleiche die Zentralitätsmaße
# Idee: 


# Plotten der Ergebnisse
plot_results(results, n_values, p_values, s_values)