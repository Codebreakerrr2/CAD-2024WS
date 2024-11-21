import networkx as nx
import matplotlib.pyplot as plt
from networkx import NetworkXNoPath
import numpy as np

##
def convertGraph(g:nx.DiGraph):
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
    #für alle knoten: 

##

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

def draw_original_graph(original_graph, options):
    plt.figure()  
    plt.title("original G", fontsize=16)
    pos = nx.spring_layout(original_graph, k=0.3, iterations=20)
    
    nx.draw_networkx(original_graph,pos=pos,with_labels=True, **options)

    extra_labels = {n: original_graph.nodes[n]['weight'] for n in original_graph.nodes}
    extra_label_offset = 0.1  # Adjust to control label distance from nodes
    extra_pos = {node: (x + extra_label_offset, y + extra_label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(original_graph,pos=extra_pos,labels=extra_labels)
    plt.show()

def plot_transformed_graph(new_graph, options):
    plt.figure()  
    plt.title("transformed G", fontsize=16)
    pos = nx.spring_layout(new_graph, k=0.3, iterations=20)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'weight')
    print(labels)
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.show()

#cum
def plot_cum_visits(new_graph, options):
    plt.figure()  
    plt.title("cumulative visits sp", fontsize=16)
    pos = nx.spring_layout(new_graph, k=0.3, iterations=20)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'cum')
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.show()

#aspcum
def plot_asp_cum_visits(new_graph, options):
    plt.figure()  
    plt.title("cumulative visits asp", fontsize=16)
    pos = nx.spring_layout(new_graph, k=0.3, iterations=20)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'aspcum')
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.show()

#sum minpath for every pair
#mutates new_graph
def cumulative_visits(new_graph, relevant_edges_4_tuple_list):
    for (fro,to,nfro,nto) in relevant_edges_4_tuple_list:
        try:
            spath = nx.shortest_path(new_graph,nfro,nto) # None für nicht erreichbar
            aspath = nx.all_shortest_paths(new_graph,nfro,nto)
        except NetworkXNoPath:
            continue
        if spath == None:
            continue
        window_size = 2
    #sp-cum
        for i in range(len(spath) - window_size + 1): #sliding window function
            attrname = "cum"
        # print("from " + )
            leg = spath[i: i + window_size]
            oldattr = nx.get_edge_attributes(new_graph,name= attrname,default=0)
            edgecum = oldattr[(leg[0],leg[1])]
            nx.set_edge_attributes(new_graph,{(leg[0],leg[1]):edgecum+1},attrname)
        #todo only between in+out > node_cum
    #asp-cum    
        # todo besuche, maxbesuche (via asp)
        for path in aspath:
            for i in range(len(spath) - window_size + 1): #sliding window function
                attrname = "aspcum"
            # print("from " + )
                leg = spath[i: i + window_size]
                oldattr = nx.get_edge_attributes(new_graph,name= attrname,default=0)
                edgecum = oldattr[(leg[0],leg[1])]
                nx.set_edge_attributes(new_graph,{(leg[0],leg[1]):edgecum+1},attrname)

#knotenpaare aus g in g^
def relevant_edges_mapping(original_graph):
    pairs = []
    for n1 in original_graph.nodes:
        for n2 in original_graph.nodes:
            pairs.append((n1,n2,str(n1)+"_o",str(n2)+"_i"))
    return pairs


if __name__ == "__main__":
    original_graph = rautenGraph()

    new_graph = convertGraph(original_graph)

    plot_options = {
    'node_color': 'yellow',
    'node_size': 1000,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}
    
    draw_original_graph(original_graph, plot_options)
    
    plot_transformed_graph(new_graph, plot_options)
    pairs = relevant_edges_mapping(original_graph)
    cumulative_visits(new_graph, pairs)
    plot_cum_visits(new_graph, plot_options)
    plot_asp_cum_visits(new_graph, plot_options)
        
        