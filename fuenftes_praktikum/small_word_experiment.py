import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d


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
    for node, lengths in shortest_paths.items():
        path_lengths.extend(lengths.values())

    # Häufigkeit der Pfadlängen berechnen
    path_distribution = Counter(path_lengths)
    total_pairs = len(graph) * (len(graph) - 1) / 2  # Gesamtzahl der Knotenpaare

    return path_distribution, total_pairs


def expected_path_length(n, p):
    """
    Berechnet die erwartete Pfadlänge E[L] für einen Watts-Strogatz-Graphen G(n, p).

    Parameter:
    - n: Anzahl der Knoten.
    - p: Wahrscheinlichkeit für das Neuverdrahten von Kanten.

    Rückgabe:
    - E[L]: Erwartete Pfadlänge.
    """
    graph = nx.watts_strogatz_graph(n, 20, p)  # Graph erstellen // wie ist K zu wählen???
    path_distribution, total_pairs = calculate_shortest_path_distribution(graph)

    # Erwartete Pfadlänge berechnen
    E_L = sum(k * count / total_pairs for k, count in path_distribution.items())
    return E_L


def calculate_elp_for_n_values(n_values, p):
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
        elp = expected_path_length(n, p)
        elp_values.append(elp)
    return n_values, elp_values


# Stützstellen für n und p-Wert
n_values = [100,300,500,1000,3000,5000,7000,10000,15000,20000]
p = 0.05

# Berechne E[L] für Stützstellen
n_values, elp_values = calculate_elp_for_n_values(n_values, p)

# Interpolation der Werte
interpolation_function = interp1d(n_values, elp_values, kind='cubic')  # Spline-Interpolation
n_fine = np.linspace(min(n_values), max(n_values), 100)  # Feine Aufteilung der n-Werte
elp_interpolated = interpolation_function(n_fine)

# Plot der Ergebnisse
plt.figure(figsize=(10, 6))
plt.plot(n_values, elp_values, 'o', label="Stützstellen (berechnete Werte)")
plt.plot(n_fine, elp_interpolated, '-', label="Interpolierte Werte")
plt.xlabel("n (Anzahl der Knoten)")
plt.ylabel("E[L] (Erwartete Pfadlänge)")
plt.title(f"Erwartete Pfadlänge E[L] für p = {p}")
plt.legend()
plt.grid()
plt.show()
