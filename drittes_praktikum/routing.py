import networkx as nx
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    original_graph = rautenGraph()
    new_graph = convertGraph(original_graph)
    options = {
    'node_color': 'yellow',
    'node_size': 1000,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}
    print(original_graph)
    
    pos = nx.spring_layout(original_graph)
    
    nx.draw_networkx(original_graph,pos=pos,with_labels=True, **options)

    extra_labels = {n: original_graph.nodes[n]['weight'] for n in original_graph.nodes}
    extra_label_offset = 0.1  # Adjust to control label distance from nodes
    extra_pos = {node: (x + extra_label_offset, y + extra_label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(original_graph,pos=extra_pos,labels=extra_labels)
    plt.show()


    pos = nx.spring_layout(new_graph)
    nx.draw(new_graph,pos=pos,with_labels=True, **options)
    labels = nx.get_edge_attributes(new_graph,'weight')
    print(labels)
    nx.draw_networkx_edge_labels(new_graph,pos=pos,edge_labels=labels)
    plt.show()
    
#knotenpaare aus g in g^
pairs = []
for n1 in original_graph.nodes:
    for n2 in original_graph.nodes:
        pairs.append((n1,n2,str(n1)+"_o",str(n2)+"_i"))

#sum minpath for every pair

for (fro,to,nfro,nto) in pairs:
    spath = nx.shortest_path(new_graph,nfro,nto) # None für nicht erreichbar
    if spath == None:
        continue
    window_size = 2
    for i in range(len(spath) - window_size + 1): #sliding window function
        # print("from " + )
        leg = spath[i: i + window_size]
        old = nx.get_edge_attributes(new_graph,(leg[0],leg[1]),"cum")
        nx.set_edge_attributes(new_graph,(leg[0],leg[1]),"cum":old+1))
        #todo only between in+out > node_cum
        # todo besuche, maxbesuche (via asp)

        




    

"""
Erstmal Graphen erstellen - Bonbon (Raute)

Knotengewichte als Attribut vergeben

Funktion um Graph zu konvertieren

Betweenness-Centrality auswerten (per Hand oder Networkx)?



"""