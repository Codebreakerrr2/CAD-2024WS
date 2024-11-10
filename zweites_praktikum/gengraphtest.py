
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
    print("gen graph - start")
    #g = nx.erdos_renyi_graph(n, p)
    g= nx.fast_gnp_random_graph(n,p)
    print("gen graph - start")

    # takes very long for big n
    #while not nx.is_connected(g):
    #    g = nx.erdos_renyi_graph(n, p)
    return g

connected_graph_with_n_p(100000,0.05)
