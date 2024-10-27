import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

'''
Zufallsgraph mit n Knoten und die Knoten sind mit Wahrscheinlichkeit p miteinander verbunden
'''
def connected_graph_with_n_p(n,p):
    g=nx.erdos_renyi_graph(n, p)
    # auf zusammenhängend überprüfen
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)
    return g

def show_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()


def minimal_distance_statics(graph):
    # Berechne alle kürzesten Distanzen zwischen allen Knotenpaaren
    distances = dict(nx.all_pairs_shortest_path_length(graph))

    # Extrahiere alle Distanzen in einer Liste, ohne doppelte Einträge
    distance_list = []
    for source, target_distances in distances.items():
        for target, distance in target_distances.items():
            if source < target:  # Nur Distanzen in einer Richtung hinzufügen
                distance_list.append(distance)

    # Berechnung der Statistiken
    mean_distance = np.mean(distance_list)
    std_deviation = np.std(distance_list)
    median_distance = np.median(distance_list)
    quartiles = np.percentile(distance_list, [25, 75])  # 1. und 3. Quartil

    # Ausgabe der Ergebnisse
    print(f"Mittelwert der Distanzen: {mean_distance}")
    print(f"Standardabweichung: {std_deviation}")
    print(f"Median der Distanzen: {median_distance}")
    print(f"1. Quartil: {quartiles[0]}")
    print(f"3. Quartil: {quartiles[1]}")

