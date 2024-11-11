
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

""" def connected_graph_with_n_p(n, p):
    print("gen graph - start")
    g = nx.erdos_renyi_graph(n, p)
    print(f"Generated graph with {n} nodes and p = {p}. Is connected? {nx.is_connected(g)}")
    #g= nx.fast_gnp_random_graph(n,p)
    #print("gen graph - start")

    # takes very long for big n
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)
        print("connected graph generated")
    return g """
# List to store disconnected graphs' info
disconnected_graphs = []

def connected_graph_with_n_p(n, p, max_attempts=50):
    print(f"Attempting to generate a connected graph with {n} nodes and p = {p}")
    
    # Try to generate a connected graph, max_attempts times
    attempts = 0
    while attempts < max_attempts:
        g = nx.erdos_renyi_graph(n, p)
        attempts += 1
        if nx.is_connected(g):
            print(f"Graph is connected after {attempts} attempts.")
            return g
        else:
            print(f"Attempt {attempts}: Disconnected graph. Trying again...")
    
    # If the graph is still not connected after max_attempts
    print(f"Failed to generate a connected graph after {max_attempts} attempts.")
    
    # If not connected, add the details to the list of disconnected graphs
    disconnected_graphs.append((n, p, attempts))
    
    # Return the disconnected graph anyway
    return g

from collections import Counter

def random_walk(graph, s):
    print("random walk - start")
    
    # Initialize a random starting node
    n = choice(list(graph.nodes))
    
    # Initialize the visits dictionary with all nodes and zero visits
    visits = dict.fromkeys(range(0, len(graph.nodes), 1), 0)
    
    # Perform the random walk
    for i in range(s+1):
        # Increment the visit count for the current node
        visits[n] += 1

        # Check if the current node has neighbors (to avoid walking on isolated nodes)
        if len(list(graph.adj[n])) == 0:
            n = n  # No movement if the node has no neighbors
        else:
            # Randomly select the next node from its neighbors
            n = choice(list(graph.adj[n]))
    
    # Sort visits dictionary by number of visits in descending order
    sorted_visits = Counter(visits).most_common(1)
    
    print("Random walk - finish")
    print("Top 10 most visited nodes:")
    for node, count in sorted_visits:
        print(f"Node {node}: {count} visits")
    
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

def run_multiple_trials_and_return_averages(n_values, p_values, s_formula, num_trials=10):
    print("\nRunning trials to calculate average visited percentage:")
    
    # List to store the average results
    average_results = []

    for n in n_values:
        for p in p_values:
            avg_visited_percentages = []

            # Run multiple trials for each combination of n, p
            for trial in range(num_trials):
                graph = connected_graph_with_n_p(n, p)
                s_values = s_formula(n)

                for s in s_values:
                    visits = random_walk(graph, s)
                    visited_only = [value for key, value in visits.items() if value > 0]
                    visited_count = len(visited_only)
                    visited_percent = visited_count / n
                    avg_visited_percentages.append(visited_percent)

            # Compute the average visited percentage across all trials
            avg_visited_percent = np.mean(avg_visited_percentages)
            average_results.append({
                'n': n,
                'p': p,
                's': s_values[0],  # Assuming s values are uniform for each n
                'average_visited_percent': avg_visited_percent
            })
         
    
    # Return the list of averages
    return average_results

from collections import Counter

def calculate_average_top_n_visits(results, top_n=5):
    """
    Calculate the average visit count for each of the top N nodes (1st, 2nd, 3rd, etc.)
    across all trials for each graph size 'n'.
    """
    n_averages = {}  # Store averages by 'n'

    # Loop through each graph size 'n' in the results
    for n, trials in results.items():
        # Initialize lists to store the top N visits across all trials
        top_n_visits = {i: [] for i in range(1, top_n + 1)}

        # Loop through each trial for this particular 'n'
        for trial in trials:
            # Sort the visit counts in descending order and get the top N visits
            sorted_visits = sorted(trial.values(), reverse=True)[:top_n]
            
            # For each of the top N nodes, add the visit count to the corresponding list
            for i, visit_count in enumerate(sorted_visits, 1):
                top_n_visits[i].append(visit_count)

        # Calculate the average for each of the top N nodes
        avg_top_n = {i: sum(visits) / len(visits) for i, visits in top_n_visits.items()}

        # Store the result in n_averages
        n_averages[n] = avg_top_n

    return n_averages



def plot_results(results, n_values, p_values, s_formula):


    #n = n_values[0]
    
    for n in n_values:
        #TODO: Handle multiple n_values

        #TODO: express steps in walk as a factor of n (fixed list maybe)

        sns.set_theme()
        df = pd.DataFrame(results)

        #df = df.query(f"n = {n}")
        df = df[(df.n == n)]

        #columns_to_plot = ['p', 's', 'visited_percent']

        df_pivoted = pd.pivot_table(df, index='s', columns='p', values='visited_percent')
        
        f, ax = plt.subplots(figsize=(20, 10))
        sns_ax = sns.heatmap(df_pivoted, annot=True, fmt=".2f", linewidths=.5, ax=ax)
        sns_ax.set(xlabel ="probability", ylabel = "Steps in Walk", title =f"Random Walk - Node Coverage % - {str(n)} Nodes in Graph ")
        #https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-formatters.html
        # does not work
        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x) ))
        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: np.format_float_positional(x/100, unique=False, precision=3) ))
        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: np.round(x,3) ))
        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: str(x) ) )
        
        plt.savefig(f"0.25_heatmap_lengths_unconnected_{str(n)}.jpg")

import pandas as pd

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_averages_as_heatmap(average_results, num_trials):
    """
    Takes the average results (list of dictionaries) and visualizes them as a heatmap,
    including the number of trials performed.
    """
    # Convert the list of average results into a pandas DataFrame
    df_avg = pd.DataFrame(average_results)

    # Pivot the table so that the number of nodes (n) are the rows, and the probabilities (p) are the columns
    df_pivoted = pd.pivot_table(df_avg, index='n', columns='p', values='average_visited_percent', aggfunc='mean')

    # Set figure size for better readability
    plt.figure(figsize=(6, 4))

    # Create the heatmap
    sns.heatmap(df_pivoted, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Visited Nodes (%)'},
                xticklabels=df_pivoted.columns, yticklabels=df_pivoted.index, annot_kws={"size": 10}, vmin=0, vmax=1)

    # Add title and axis labels, including the number of trials
    plt.title(f"Average Node Coverage (%) over\n {num_trials} Trials with p=0.26 and s=n*10")
    plt.xlabel("Probability (p)")
    plt.ylabel("Number of Nodes (n)")

    # Display the plot
    plt.show()






'''
hier werden die heatmaps generiert, um das verhältnis zwischen den verschiedenen 
unabhängigen variablen n, p, und s - und deren resultierende Knoten-abdeckung 
innerhalb eines zusammenhängenden Graphen zu visualisieren 
'''

''' Hier werden die werte für n definiert - nach bedarf auskommentieren! '''
#n_values = list(range(50, 1000, 100))  # 1. Anzahl der Knoten von 50 bis 1000 in Schritten von 50
n_values = [100, 500, 1000] # 2. n als liste festgelegt 
#n_values = discrete_log_scale(100,1000,5) # 3. n als logarithmisch-zunehmender wert von min bis max in x schritte 

''' Hier werden die werte fur p definiert - nach bedarf auskommentieren! '''
#p_values = [0.01, 0.26, 0.51, 0.76, 1.0]  
p_values = [0.26]
#p_values = np.linspace(0,0.01,10,endpoint=True).tolist() # values between 0 and 0.01, with 10 steps 
#p_values = np.linspace(0.03, 0.05,3,endpoint=True).tolist() # values ranging from 0.03 to 0.05, with only 3 steps

p_values = [round(x,3) for x in p_values] # format p values to three decimal points 

''' Hier werden verschiedene s formula (von n abhängig) ausprobiert - nach bedarf auskommentieren!'''
#s_formula = lambda n: n * 1000
#s_formula = lambda n: discrete_log_scale(n,n*10,10) 
s_formula = lambda n: [n * 10]

''' Hier werden die Knoten abdeckung eines random walks für 
alle kombinationen von n, p, und s wertße berechnet 
und das Ergebniss als ein dictionary der größe n*p*s zurückgegeben '''
# Führe die Statistiken für unterschiedliche Werte von n, p und s durch
#results = try_different_n_p_statics(n_values, p_values, s_formula)

'''
Arithmetischer mittelwert über 10 durchläufe ausrechnen - ACHTUNG!! nur bei kleinen n benutzen
'''
plot_averages_as_heatmap(run_multiple_trials_and_return_averages(n_values, p_values, s_formula), 10)

# Vergleiche die Zentralitätsmaße
# Idee: 


'''  Plotten der Ergebnisse als heatmaps ''' 
#plot_results(results, n_values, p_values,s_formula)

print(disconnected_graphs)

avg_top_5_visits = calculate_average_top_n_visits(results, top_n=5)

# Print the result
for n, avg in avg_top_5_visits.items():
    print(f"Average visits for top 5 nodes for n={n}:")
    for i in range(1, 2):  # 1st to 5th most visited nodes
        print(f"  {i}-th node: {avg[i]:.2f} visits")

