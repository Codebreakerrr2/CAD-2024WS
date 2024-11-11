
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter



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

def convert(lst):
    res_dict = {}
    res_dict = {i+1: value for i, value in enumerate(lst)}
    return res_dict

a = [1,0,1,0,1,0]
b = [1,0,1,0,1,0]
c = [0,1,0,1,0,1]
d = [0.5,0,0.5,0,0.5,0,0.5]

a=convert(a)
b=convert(b)
c=convert(c)
d=convert(d)

cor_sp , pval = calculate_spearman_correlation(a,b)

print(cor_sp)
print(pval)
