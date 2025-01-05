import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
from datetime import datetime
import itertools

def calculate_shortest_path_distribution(graph):
    """
    Berechnet die Verteilung der kürzesten Pfadlängen im Graphen.

    Parameter:
    - graph: NetworkX-Graph.

    Rückgabe:
    - path_distribution: Dictionary {Pfadlänge: Anzahl der Paare}.
    - total_pairs: Gesamtzahl der Knotenpaare.
    """
    #nx.average_shortest_path_length(graph)

    node_pairs = list(itertools.combinations(graph.nodes(), 2))
    #print("node_pairs")
    #print(node_pairs)
    length_distribution = {}

    for source, target in node_pairs:
        #print("s: ", source , " t: ", target)
        length = nx.shortest_path_length(graph, source, target)

        if length in length_distribution:
            # If the length is already in the dictionary, increment its countif length in length_distribution:
            length_distribution[length] += 1
        else:
            # Otherwise, initialize the count for this length
            length_distribution[length] = 1
    #print(length_distribution)
    #print(len(node_pairs))
    return length_distribution, len(node_pairs)


def berechne_anteil_pfade_mit_laenge_k(n, k, length_distribution):
    if k not in length_distribution.keys(): #.keys()
        return 0
    number_of_paths_with_length_k = length_distribution[k]
    #print("number_of_paths_with_length_1")
    #print(number_of_paths_with_length_k)
    #total pairs
    #2 wg ungeordnet
    return 2 / (n * (n-1)) * number_of_paths_with_length_k



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

    length_distribution, total_pairs = calculate_shortest_path_distribution(graph)

    print("length_distribution")
    print(length_distribution)
    anteil_pfade_mit_laenge_k = {}

    k_threshold = n-1


    anteil_fuer_k7 = berechne_anteil_pfade_mit_laenge_k(n, 1, length_distribution)

    for k in range(k_threshold+1):
        anteil_pfade_mit_laenge_k[k] = berechne_anteil_pfade_mit_laenge_k(n, k, length_distribution)


    # Erwartete Pfadlänge berechnen
    E_L = sum(k * anteil for k, anteil in anteil_pfade_mit_laenge_k.items())
    return E_L, anteil_fuer_k7, length_distribution


def calculate_epl_for_n_values(n_values, p):
    """
    Berechnet E[L] für eine Liste von n-Werten.

    Parameter:
    - n_values: Liste der Knotenanzahlen (Stützstellen).
    - p: Wahrscheinlichkeit für das Neuverdrahten von Kanten.

    Rückgabe:
    - n_values: Liste der Knotenanzahlen.
    - elp_values: Liste der berechneten E[L]-Werte.
    """

    epl_k7 = []

    epl_values = []

    length_distributions = {}
    for n in n_values:
        print(str(datetime.now()) + " start   epl: " + str(n))
        epl, anteil_fuer_k7, length_distribution = expected_path_length(n, p)
        epl_values.append(epl)
        epl_k7.append(anteil_fuer_k7)
        length_distributions[n] = length_distribution
    return n_values, epl_values, epl_k7, length_distributions


# Stützstellen für n und p-Wert
#n_values = [100,300,500,1000,3000,5000,7000,10000,15000,20000]
n_values = [100,300,500,1000,3000,5000]
n_values = [100,200,300,400]#,1000]#,3000,5000]
p = 0.05

# Berechne E[L] für Stützstellen
n_values, epl_values, epl_k7, length_distribution = calculate_epl_for_n_values(n_values, p)
print(str(datetime.now()) + " endof elp: ")

# Interpolation der Werte
interpolation_function = interp1d(n_values, epl_values, kind='cubic')  # Spline-Interpolation
n_fine = np.linspace(min(n_values), max(n_values), 100)  # Feine Aufteilung der n-Werte
epl_interpolated = interpolation_function(n_fine)

print("epl values")
print(epl_values)
print("epl k7")
print(epl_k7)
#print("len distr")
#print(length_distribution)

# Plot der Ergebnisse
plt.figure(figsize=(10, 6))
plt.plot(n_values, epl_values, 'o', label="Stützstellen (berechnete Werte)")
plt.plot(n_fine, epl_interpolated, '-', label="Interpolierte Werte")
plt.xlabel("n (Anzahl der Knoten)")
plt.ylabel("E[L] (Erwartete Pfadlänge)")
plt.title(f"Erwartete Pfadlänge E[L] für p = {p}")
plt.legend()
plt.grid()
#plt.figure(figsize=(10, 6))
#plt.title(f"Erwartete Pfadlänge E[L] für p = {p}")
#plt.legend()
#plt.plot(n_values, epl_values, 'o', label="Stützstellen (berechnete Werte)")
#plt.plot(n_values, epl_k7, '-', label="k7")
plt.show()

for n in enumerate(n_values):
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, length_distribution[n])
    plt.xlabel("Pfadlänge")
    plt.ylabel("Häufigkeit")
    plt.title(f"Verteilung der Pfadlängen für n = {n}")
    plt.legend()
    plt.grid()
    plt.show()

plt.figure(figsize=(10, 6))
for i, n in enumerate(n_values):
    plt.hist(epl_k7[i], bins=30, alpha=0.7, label=f"n = {n}")
plt.xlabel("E[L] k7-Werte")
plt.ylabel("Häufigkeit")
plt.title("Verteilung der EPL k7-Werte für alle n")
plt.legend()
plt.grid()
plt.show()