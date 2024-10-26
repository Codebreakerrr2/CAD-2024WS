import networkx as nx
import matplotlib.pyplot as plt

'''
Zufallsgraph mit n Knoten und die Knoten sind mit Wahrscheinlichkeit p miteinander verbunden
'''
def graph_with_n_p(n,p):
    g=nx.erdos_renyi_graph(n,p)
    return g

def show_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()

