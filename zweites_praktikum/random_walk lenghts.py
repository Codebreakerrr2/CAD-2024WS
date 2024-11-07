
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def connected_graph_with_n_p(n, p):
    g = nx.erdos_renyi_graph(n, p)

    # takes very long for big n
    #while not nx.is_connected(g):
    #    g = nx.erdos_renyi_graph(n, p)
    return g

def random_walk(graph,s):
    n = choice(list(graph.nodes))

    #cc = nx.connected_components(graph)
    #largest_cc = max(cc, key=len)

    #while not len(graph.adj[n])>0:
    #    n = choice(list(graph.nodes))


    #prefilled dict for later comparison as nx.*centrality dictionaries are complete
    visits = dict.fromkeys(range(0,len(graph.nodes),1),0)

    # Upsert Logic nicht mehr notwendig
    for i in range(s+1):
        if n not in visits.keys():
            visits[n] = 1
        else:
            visits[n] = visits[n]+1

        # random walk in place if necessary
        if len(list(graph.adj[n]))==0:
            break
        else:
            n = choice(list(graph.adj[n]))
    #list_of_visits = list(visits.items()) #reverse with dict(list_of_visits)
    return visits


# https://stackoverflow.com/questions/71722781/creating-a-logarithm-scale-with-min-max-and-discrete-steps
def discrete_log_scale(min, max, steps):
    #const min = Math.min(data);
    #const max = Math.max(data);

    logmin = math.log(min)
    logmax = math.log(max)

    logrange = logmax - logmin;
    logstep = logrange / steps;
    # value = Math.exp(logmin + n * logstep);
    #k: v/visits_max for k, v in visits.items()
    
    logscale = [ math.ceil(math.exp(logmin + logstep * i)) for i in range(0,steps+1,1)]
    return logscale


def try_different_n_p_statics(n_values, p_values, s_formula):
    results = []
    for n in n_values:
        for p in p_values:
            graph = connected_graph_with_n_p(n, p)
            
            #s_values = list(range(n, s_formula(n), int( (s_formula(n)-n)/10 ) ) )
            
            #s_values = discrete_log_scale(n,n*10,10)
            s_values = s_formula(n)

            # suche nach geeigneten s für n
            # größe von n relevant?
            # für n1000, 100% für ca 10*n
            # n10k, 100% ab > 10*n
            # für große n benötigt man immer mehr
            # geeignet schon bei >1 visit für jeden knoten?

            
            for s in s_values:
                
                visits = random_walk(graph, s)
                #visits_max = max(visits.values())

                visited_only = [value for key, value in visits.items() if value > 0]

                visited_count = len(visited_only)

                visited_percent = visited_count/n
                
                results.append({
                    'n': n,
                    'p': p,
                    's': s,
                    #'visits': visits,
                    'visited_count': visited_count,
                    'visited_percent': visited_percent,
                    
                })
                print(
                    #f"n: {n}, p: {p} -> Mittelwert der Distanzen: {mean_distance:.2f}, Standardabweichung: {std_deviation:.2f}, Median: {median_distance:.2f}")
                    f"n: {n}, p: {p}, s: {s} -> %: {visited_percent}")

    return results

def plot_results(results, n_values, p_values, s_formula):


    #TODO: Handle multiple n_values
    n = n_values[0]

    #TODO: express steps in walk as a factor of n (fixed list maybe)

    sns.set_theme()
    df = pd.DataFrame(results)
    

    #columns_to_plot = ['p', 's', 'visited_percent']

    df_pivoted = pd.pivot_table(df, index='s', columns='p', values='visited_percent')
    
    f, ax = plt.subplots(figsize=(20, 10))
    sns_ax = sns.heatmap(df_pivoted, annot=True, fmt=".2f", linewidths=.5, ax=ax)
    sns_ax.set(xlabel ="probability", ylabel = "Steps in Walk", title =f"Random Walk - Node Coverage % - {str(n)} Nodes in Graph ")
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x) ))
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: np.format_float_positional(x/100, unique=False, precision=3) ))
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: np.round(x,3) ))
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: str(x) ) )
    
    plt.savefig('heatmap_unconnected.jpg')


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
n_values = list(range(50, 1000, 100))  # Anzahl der Knoten von 50 bis 1000 in Schritten von 50
n_values = [1000]

#p_values = [i /20.0 for i in range(1, 21)]  # Werte von 0.1 bis 1.0 in Schritten von 0.1
p_values = np.linspace(0,0.03,20,endpoint=True).tolist()
#p_values = np.arange(0.0, 0.05, (0.05-0)/20).tolist()

p_values = [round(x,3) for x in p_values] #rounding to fix ouput formatter

#s_formula = lambda n: n * 1000
s_formula = lambda n: discrete_log_scale(n,n*10,10)


# Führe die Statistiken für unterschiedliche Werte von n, p und s durch
results = try_different_n_p_statics(n_values, p_values, s_formula)

# Vergleiche die Zentralitätsmaße
# Idee: 


# Plotten der Ergebnisse
plot_results(results, n_values, p_values,s_formula)
