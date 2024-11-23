import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx import NetworkXNoPath
import numpy as np


## nodeweight -> edgeweight
def convert_graph_from_nodebased_to_edgebased(g:nx.DiGraph) -> nx.DiGraph:
    ng = nx.DiGraph()
    for node, attr in g.nodes(data=True):
        node_in = f"{node}_i"
        node_out = f"{node}_o"
        weight = attr.get('weight', 0)
        ng.add_node(node_in)
        ng.add_node(node_out)
        ng.add_edge(node_in, node_out, weight=weight)
    for u, v, attr in g.edges(data=True):
        # Redirect the edges to connect the new nodes appropriately
        ng.add_edge(f"{u}_o", f"{v}_i",weight=1)
    return ng



layout_iterations = 300
layout_k = 0.5


def rautenGraph():
    # Create a graph with node weights
    G = nx.DiGraph()
    G.add_nodes_from([
        (1, {'weight': 10}), #A Start
        (2, {'weight': 20}), #B Weg 1
        (3, {'weight': 30}), #D Weg 2
        (4, {'weight': 10}), #C Ziel
    ])
    G.add_edges_from([(1, 2), (2, 4), (1, 3), (3,4)]) # Weg1 + Weg2
    return G

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def plot_original_graph(original_graph, options):
    plt.figure()  
    plt.title("original G", fontsize=16)
    pos = nx.spring_layout(original_graph, k=layout_k, iterations=layout_iterations)
    
    nx.draw_networkx(original_graph,pos=pos,with_labels=True, **options)

    extra_labels = {n: original_graph.nodes[n]['weight'] for n in original_graph.nodes}
    extra_label_offset = 0.1  # Adjust to control label distance from nodes
    extra_pos = {node: (x + extra_label_offset, y + extra_label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(original_graph,pos=extra_pos,labels=extra_labels)
    # plt.show()
    plt.savefig("original_graph.png")

def plot_transformed_graph(new_graph, options):
    plt.figure()  
    plt.title("transformed G", fontsize=16)
    pos = nx.spring_layout(new_graph, k=layout_k, iterations=layout_iterations)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'weight')
    print(labels)
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    # plt.show()
    plt.savefig("transformed_graph.png")

#cum
def plot_spcum_visits(new_graph, options):
    plt.figure()  
    plt.title("cumulative visits sp", fontsize=16)
    pos = nx.spring_layout(new_graph, k=layout_k, iterations=layout_iterations)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'spcum')
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.savefig("sp_cum_visits.png")
    #plt.show()

#aspcum
def plot_asp_cum_visits(new_graph, options):
    plt.figure()  
    plt.title("cumulative visits asp", fontsize=16)
    pos = nx.spring_layout(new_graph, k=layout_k, iterations=layout_iterations)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'aspcum')
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.savefig("asp_cum_visits.png")
    #plt.show()

def plot_oldgraph_with_any_attr_top_left(original_graph,title, attr_to_draw, options):
    plt.figure()  
    plt.title(title, fontsize=16)
    pos = nx.spring_layout(original_graph, k=layout_k, iterations=layout_iterations)
    
    nx.draw_networkx(original_graph,pos=pos,with_labels=True, **options)

    extra_labels_weight = {n: original_graph.nodes[n]['weight'] for n in original_graph.nodes}
    extra_label_offset_weight = 0.1  # Adjust to control label distance from nodes
    extra_pos_weight = {node: (x + extra_label_offset_weight, y + extra_label_offset_weight) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(original_graph,pos=extra_pos_weight,labels=extra_labels_weight)

    extra_labels_aspcum = {n: original_graph.nodes[n][attr_to_draw] for n in original_graph.nodes}
    extra_label_offset_aspcum = 0.1  # Adjust to control label distance from nodes
    extra_pos_aspcum = {node: (x - extra_label_offset_aspcum, y + extra_label_offset_aspcum) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(original_graph,pos=extra_pos_aspcum,labels=extra_labels_aspcum)

    plt.savefig(attr_to_draw + ".png")
    
    #plt.show()
    


def original_node_2_transformed_edge(nodeid) -> tuple:
    return (str(nodeid)+"_i",str(nodeid)+"_o")

#sum minpath for every pair
#mutates new_graph
def cumulative_visits_mutates_edgebased_graph(nodebased_graph,edgebased_graph):
    """adds spcum and aspcum attributes to edgebased graph"""

    def cartesian_product_nodebased_edgebased_mapping(nodebased_graph): # cartesian product - wrong?
        """knotenpaare als 4-tupel aus g in g^"""
        pairs = []
        for node_from in nodebased_graph.nodes:
            for node_to in nodebased_graph.nodes:
                pairs.append((node_from,node_to,str(node_from)+"_o",str(node_to)+"_i")) #sic! o->i
        return pairs
    
    possible_pairs = cartesian_product_nodebased_edgebased_mapping(nodebased_graph)

    for (fro,to,nfro,nto) in possible_pairs:
        try:
            spath = nx.shortest_path(edgebased_graph,nfro,nto) # None für nicht erreichbar
            aspath = nx.all_shortest_paths(edgebased_graph,nfro,nto)
        except NetworkXNoPath:
            continue
        if spath == None:
            continue
        window_size = 2
    #sp-cum
        for i in range(len(spath) - window_size + 1): #sliding window function
            attrname = "spcum"
            leg = spath[i: i + window_size]
            oldattr = nx.get_edge_attributes(edgebased_graph,name= attrname,default=0)
            edgecum = oldattr[(leg[0],leg[1])]
            nx.set_edge_attributes(edgebased_graph,{(leg[0],leg[1]):edgecum+1},attrname)
        #todo only between in+out > node_cum
    #asp-cum    
        # todo besuche, maxbesuche (via asp)
        for path in aspath:
            for i in range(len(spath) - window_size + 1): #sliding window function
                attrname = "aspcum"
                leg = spath[i: i + window_size]
                oldattr = nx.get_edge_attributes(edgebased_graph,name= attrname,default=0)
                edgecum = oldattr[(leg[0],leg[1])]
                nx.set_edge_attributes(edgebased_graph,{(leg[0],leg[1]):edgecum+1},attrname)



def reversetransform_attributes(nodebased_graph, edgebased_graph,attributename,defaultvalue) -> dict:
    new_attrs = nx.get_edge_attributes(edgebased_graph,attributename)
    #relevant_edges = []
    filtered_newgraph_attrs = {}
    filtered_oldgraph_attrs = {}
    for node in nodebased_graph.nodes:
        new_edge = original_node_2_transformed_edge(node)
        #relevant_edges.append(new_edge)
        if(new_edge in new_attrs):
            filtered_newgraph_attrs[new_edge] = new_attrs[new_edge]
        else:
            filtered_newgraph_attrs[new_edge] = defaultvalue
    for node in nodebased_graph.nodes:
        # if()
        attr = filtered_newgraph_attrs[original_node_2_transformed_edge(node)]
        filtered_oldgraph_attrs[node] = attr
    return filtered_oldgraph_attrs

def normalize_attrs(attribute_dict : dict) -> dict:
    #get max
    max_val = max(attribute_dict.values())

    new_dict = {}
    for key,value in attribute_dict.items():
        new_dict[key] = value/max_val
    return new_dict

def get_metrics(node_loads,nodebased_graph):
        """  node_loads,mean,variance,max,normalized_entropy, balance """
        ### Begin ChatGpt Copypasta
        loads = np.array(list(node_loads.values()))
        mean_load = np.mean(loads)
        var_load = np.var(loads)
        max_load = np.max(loads)
        
        # Entropie (normiert)
        probabilities = loads / np.sum(loads)  # Wahrscheinlichkeitsverteilung der Lasten
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Entropie ####1e-10 is added because small probabilities might get rounded to 0 and the log for 0 is undefined.
        max_entropy = np.log(len(nodebased_graph.nodes()))  # Maximale Entropie
        normalized_entropy = entropy / max_entropy
        
        # Balancemetrik
        balance_metric = max_load / mean_load
        
        # Ergebnisse
        metrics = {
            "node_loads": node_loads,
            "mean_load": mean_load,
            "var_load": var_load,
            "max_load": max_load,
            "normalized_entropy": normalized_entropy,
            "balance_metric": balance_metric
        }
        return metrics

def transform_edgebased_to_nodebased_attributes(nodebased_graph : dict,attribute_dict :dict) -> dict :
    nodebased_attribute_dict = {}
    for node in nodebased_graph.nodes:
        nfrom, nto = original_node_2_transformed_edge(node)
        nodebased_attribute_dict[node] = max(attribute_dict[nfrom],betweenness_edgebased_attrdict[nto])


def old_main():
    graph_nodebased = rautenGraph()

    graph_edgebased = convert_graph_from_nodebased_to_edgebased(graph_nodebased)

    plot_options = {
    'node_color': 'yellow',
    'node_size': 1000,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}
    
    plot_original_graph(graph_nodebased, plot_options)
    
    plot_transformed_graph(graph_edgebased, plot_options)
    
    # add spcum und aspcum attributes to new_graph
    cumulative_visits_mutates_edgebased_graph(graph_nodebased, graph_edgebased)

    plot_spcum_visits(graph_edgebased, plot_options)

    plot_asp_cum_visits(graph_edgebased, plot_options)

    spcum_edgebased = nx.get_edge_attributes(graph_edgebased, 'spcum') #shortest path visits
    #spcum format: {(1, 2): 4.5, (2, 3): 3.0}

    aspcum_edgebased = nx.get_edge_attributes(graph_edgebased, 'aspcum') #slightly better results than spcum maybe
    #spcum format: {(1, 2): 4.5, (2, 3): 3.0}

    #calculate betweenness on edgebased graph via weight attribute
    betweenness_edgebased_attrdict = nx.betweenness_centrality(graph_edgebased,weight="weight")#similar to spcum
    
    print("btwn" + str(betweenness_edgebased_attrdict) )
    
    

    btwn_transformed_nodebased = transform_edgebased_to_nodebased_attributes(graph_nodebased,betweenness_edgebased_attrdict)
    print(btwn_transformed_nodebased)
    
    #Setze betweenness im nodeweighted_graph mit attribute_dict
    nx.set_node_attributes(graph_nodebased,btwn_transformed_nodebased,"btwn") #betweenness ist knotenbezogen -- no fix.


    #reverse transformed dicts
    filtered_nodebased_attr_spcum = reversetransform_attributes(graph_nodebased, graph_edgebased, "spcum",0)
    normalized_nodebased_attr_spcum = normalize_attrs(filtered_nodebased_attr_spcum)
    filtered_nodebased_attr_aspcum = reversetransform_attributes(graph_nodebased, graph_edgebased, "aspcum",0)
    normalized_nodebased_attr_aspcum = normalize_attrs(filtered_nodebased_attr_aspcum)
    #filtered_oldgraph_attr_betweenness = reversetransform_attributes(original_graph, new_graph, "btwn",0)

    print(filtered_nodebased_attr_spcum)
    print(normalized_nodebased_attr_spcum)
    print(filtered_nodebased_attr_aspcum)
    print(normalized_nodebased_attr_aspcum)
    #print(filtered_oldgraph_attr_betweenness)
    #print(btwn)

    ## apply transformed normalized dicts to original graph for plotting
    nx.set_node_attributes(graph_nodebased,normalized_nodebased_attr_aspcum,"aspcum")

    plot_oldgraph_with_any_attr_top_left(graph_nodebased,"cumulative visits asp original graph", "aspcum", plot_options)

    plot_oldgraph_with_any_attr_top_left(graph_nodebased,"betweenness original graph", "btwn", plot_options)

    # SHOW PLOTS
    #plt.show()


    ###Calculate Metrics on loads (AllShortestPathVisits) 
    node_loads = aspcum_edgebased
    metrics = get_metrics(node_loads,graph_nodebased)
        # Format:
        # metrics = {
        #         "node_loads": node_loads,
        #         "mean_load": mean_load,
        #         "var_load": var_load,
        #         "max_load": max_load,
        #         "normalized_entropy": normalized_entropy,
        #         "balance_metric": balance_metric
        #     }
    
    print("remember the balance_metric for later comparison: ")
    print(metrics)


    
def apply_optimized_weights_on_attribute_dict_according_to_bussmeier(nodebased_betweenness_dict : dict, constant : int) -> dict:
    """VScode-DocStringTest"""
    # maybe also known as shortest-path-reweighting?
    #braucht betweenness, constant
    new_weights_attribute_dict = {}
    for k,v in nodebased_betweenness_dict:
        new_weight = constant * v
        new_weights_attribute_dict[k] = new_weight
    return new_weights_attribute_dict
    
    #apply_optimized_weights_on_attribute_dict_according_to_bussmeier(btwn_transformed_nodebased)



def do_single_experiment_iteration(original_graph_with_node_weights:nx.DiGraph):
    """ calculates load and metrics """
    graph_nodebased = original_graph_with_node_weights.copy()

    def calculate_load_and_get_metrics(graph_nodebased) -> tuple[dict, dict]:
        #convert nodebased (input graph) to edgebased (for calculations)
        graph_edgebased = convert_graph_from_nodebased_to_edgebased(graph_nodebased)
        
        #calculate cumulative visits as load similar to betweenness on edgebased graph
        #cumulative_visits_mutates_edgebased_graph(graph_nodebased, graph_edgebased)
        #aspcum_edgebased = nx.get_edge_attributes(graph_edgebased, 'aspcum') #slightly better results than spcum maybe
        
        #calculate betweenness centrality on edgebased
        betweenness_edgebased_attrdict = nx.betweenness_centrality(graph_edgebased,weight="weight")#similar to spcum
        
        # transform betweenness to nodebased
        betweenness_nodebased = transform_edgebased_to_nodebased_attributes(graph_nodebased,betweenness_edgebased_attrdict)
        
        #calculate metrics (i.e. balance)  using betweenness (not asp)
        node_loads = betweenness_edgebased_attrdict
        metrics_before_reweighting = get_metrics(node_loads,graph_nodebased)

        return betweenness_nodebased, metrics_before_reweighting

    betweenness_nodebased, metrics_before_reweighting = calculate_load_and_get_metrics(graph_nodebased)

    # apply bussmeier's reweighting formula
    new_weights_nodebased_attrdict = apply_optimized_weights_on_attribute_dict_according_to_bussmeier(betweenness_nodebased)
    
    # set new weights as weights on nodebased graph
    nx.set_node_attributes(graph_nodebased,new_weights_nodebased_attrdict,"weight")

    betweenness_nodebased, metrics_after_reweighting = calculate_load_and_get_metrics(graph_edgebased)

    return graph_nodebased, metrics_before_reweighting, metrics_after_reweighting


if __name__ == "__main__":
    directed_nodeweighted_graph = rautenGraph()
    altered_graph_nodebased, metrics_before_reweighting, metrics_after_reweighting = do_single_experiment_iteration(directed_nodeweighted_graph)
    print(metrics_before_reweighting)
    print(metrics_after_reweighting)
        


    
#rücktransformiert
# als dict
# von nx.betweenness() rücktransformieren als value:attr-paar-liste
# plot für die

# varianz (aus dict?)

#Vorschlag GPT für Evenness score des Graphs = weight1(1−Maß an Varianz)+weight2⋅Entropy (diese Idee mit gewichten finde ich gut)   

#Idealfall - alle gleicher evenness und damit vergleichen


#constant = 10
#newweight = []
#{(nodeid:betweenness-score-per-nodeid)}=nx.betweenness_centrality(g,weight="weight")
#foreach (node,score) in dict:
#    newweight.append(  (node,score * constant) )

#teil1 für newweight

# Formel w(new_graph_edge) = constant * betweenness(new_graph, weight="weight")
#                         gewicht hoch |  gewicht niedrig
# betweenness    hoch |    w=mittel    |     w= hoch
# betweenness niedrig |    w=niedrig   |     w=niedrig