import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
import itertools
from datetime import datetime

def calculate_shortest_path_distribution(graph):
    """
    Berechnet die Verteilung der kürzesten Pfadlängen im Graphen.

    Parameter:
    - graph: NetworkX-Graph.

    Rückgabe:
    - path_distribution: Dictionary {Pfadlänge: Anzahl der Paare}.
    - total_pairs: Gesamtzahl der Knotenpaare.
    """
    shortest_paths = dict(nx.shortest_path_length(graph))
    path_lengths = []

    # Alle kürzesten Pfade sammeln
    #for node, lengths in shortest_paths.items():
    #    path_lengths.extend(lengths.values())

    # Häufigkeit der Pfadlängen berechnen
    #path_distribution = Counter(path_lengths)

    #node_pairs = list(itertools.combinations(graph.nodes(), 2))

    length_distribution = {}
    for source, target in itertools.combinations(graph.nodes(), 2):
        #print("s: ", source , " t: ", target)
        length = shortest_paths[source][target]

        if length in length_distribution:
            # If the length is already in the dictionary, increment its countif length in length_distribution:
            length_distribution[length] += 1
        else:
            # Otherwise, initialize the count for this length
            length_distribution[length] = 1
    #total_pairs = len(graph) * (len(graph) - 1) / 2  # Gesamtzahl der Knotenpaare
    total_pairs = len(graph) * (len(graph) - 1) / 2  # Gesamtzahl der Knotenpaare

    return length_distribution, total_pairs


def expected_path_length(n,k, p):
    """
    Berechnet die erwartete Pfadlänge E[L] für einen Watts-Strogatz-Graphen G(n, p).

    Parameter:
    - n: Anzahl der Knoten.
    - p: Wahrscheinlichkeit für das Neuverdrahten von Kanten.

    Rückgabe:
    - E[L]: Erwartete Pfadlänge.
    """
    graph = nx.watts_strogatz_graph(n, k, p)  # Graph erstellen // wie ist K zu wählen???
    path_distribution, total_pairs = calculate_shortest_path_distribution(graph)

    # Erwartete Pfadlänge berechnen
    E_L = sum(k * count / total_pairs for k, count in path_distribution.items())
    return E_L


def calculate_elp_for_n_values(n_values,k, p):
    """
    Berechnet E[L] für eine Liste von n-Werten.

    Parameter:
    - n_values: Liste der Knotenanzahlen (Stützstellen).
    - p: Wahrscheinlichkeit für das Neuverdrahten von Kanten.

    Rückgabe:
    - n_values: Liste der Knotenanzahlen.
    - elp_values: Liste der berechneten E[L]-Werte.
    """
    elp_values = []
    for n in n_values:
        print(str(datetime.now()) + " start   epl: " + str(n))
        elp = expected_path_length(n, k, p)
        elp_values.append(elp)
    return n_values, elp_values


def run_experiment_for_k_values(n_values, k_values, p):
    # Berechne E[L] für Stützstellen
    n_values, elp_values = calculate_elp_for_n_values(n_values, k, p)

    # Interpolation der Werte
    interpolation_function = interp1d(n_values, elp_values, kind='cubic',
                                      fill_value='extrapolate')  # Spline-Interpolation
    n_fine = np.linspace(min(n_values), max(n_values), 100)  # Feine Aufteilung der n-Werte
    elp_interpolated = interpolation_function(n_fine)

    # Plot der Ergebnisse
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, elp_values, 'o', label="Stützstellen (berechnete Werte)")
    plt.plot(n_fine, elp_interpolated, '-', label="Interpolierte Werte")
    plt.xlabel("n (Anzahl der Knoten)")
    plt.ylabel("E[L] (Erwartete Pfadlänge)")
    plt.title(f"Erwartete Pfadlänge E[L] für p = {p}, k = {k}")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"small_world_k{k}")


# Stützstellen für n und p-Wert
n_values = [100,300,500,1000,3000,5000,7000,10000,15000,20000]
#n_values = [100,200,300,400,600]#500,1000,3000,5000,7000,10000,15000,20000]
k_values = [20, 50, 100]
p = 0.05
k = 100

run_experiment_for_k_values(n_values, k_values, p)
