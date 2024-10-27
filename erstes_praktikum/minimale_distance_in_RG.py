import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def connected_graph_with_n_p(n, p):
    g = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)
    return g


def minimal_distance_statics(graph):
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    distance_list = []
    for source, target_distances in distances.items():
        for target, distance in target_distances.items():
            if source < target:  # Nur Distanzen in einer Richtung hinzufügen
                distance_list.append(distance)

    mean_distance = np.mean(distance_list)
    std_deviation = np.std(distance_list)
    median_distance = np.median(distance_list)
    quartiles = np.percentile(distance_list, [25, 50, 75])

    return mean_distance, std_deviation, median_distance, quartiles


def try_different_n_p_statics(n_values, p_values):
    results = []
    for n in n_values:
        for p in p_values:
            graph = connected_graph_with_n_p(n, p)
            mean_distance, std_deviation, median_distance, quartiles = minimal_distance_statics(graph)
            results.append({
                'n': n,
                'p': p,
                'mean_distance': mean_distance,
                'std_deviation': std_deviation,
                'median_distance': median_distance,
                '1st_quartile': quartiles[0],
                '2nd_quartile': quartiles[1],
                '3rd_quartile': quartiles[2]
            })
            print(
                f"n: {n}, p: {p} -> Mittelwert der Distanzen: {mean_distance:.2f}, Standardabweichung: {std_deviation:.2f}, Median: {median_distance:.2f}")

    return results


def plot_results(results, n_values, p_values):
    # Extrahiere Mittelwerte und Standardabweichungen
    mean_distances = []
    std_devs = []

    for p in p_values:
        p_mean_distances = []
        p_std_devs = []
        for result in results:
            if result['p'] == p:
                p_mean_distances.append(result['mean_distance'])
                p_std_devs.append(result['std_deviation'])
        mean_distances.append(p_mean_distances)
        std_devs.append(p_std_devs)

    # Erstelle das Diagramm für Mittelwerte der Distanzen
    plt.figure(figsize=(12, 6))
    for i, p in enumerate(p_values):
        plt.plot(n_values, mean_distances[i], label=f'p = {p:.1f}', marker='o')

    plt.title('Mittelwerte der Distanzen für verschiedene p-Werte')
    plt.xlabel('Anzahl der Knoten (n)')
    plt.ylabel('Mittelwert der Distanzen')
    plt.xticks(n_values)
    plt.legend()
    plt.grid()
    plt.show()

    # Erstelle das Diagramm für die Standardabweichungen
    plt.figure(figsize=(12, 6))
    for i, p in enumerate(p_values):
        plt.plot(n_values, std_devs[i], label=f'p = {p:.1f}', marker='o')

    plt.title('Standardabweichungen der Distanzen für verschiedene p-Werte')
    plt.xlabel('Anzahl der Knoten (n)')
    plt.ylabel('Standardabweichung der Distanzen')
    plt.xticks(n_values)
    plt.legend()
    plt.grid()
    plt.show()


# Definiere Werte für n und p
n_values = list(range(5, 1000, 20))  # Anzahl der Knoten von 50 bis 1000 in Schritten von 50
p_values = [i / 10.0 for i in range(1, 11)]  # Werte von 0.1 bis 1.0 in Schritten von 0.1

# Führe die Statistiken für unterschiedliche Werte von n und p durch
results = try_different_n_p_statics(n_values, p_values)

# Plotten der Ergebnisse
plot_results(results, n_values, p_values)