# from msilib import add_data
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import networkx as nx
import scipy
import pandas as pd
import scipy.cluster.hierarchy as spc
from matplotlib import cm
# from tree_visualization import *
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np

def polar_score(adata):
    """
    Plot polar (mutually exclusive) scores for all gene pairs (MaxOVL method)
    """
    data = adata.uns['mule']['opposite score']
    scores = data['mutual_exclusive_score'].values

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + 1 * std_score  # MaxOVL 固定倍数

    plt.figure(figsize=(6,4))
    counts, bins, _ = plt.hist(scores, bins=50, color='skyblue', edgecolor='k')

    plt.axvline(threshold, color='r', linestyle='--', label=f"Mutually expressed threshold: {threshold:.2f}")
    plt.xlabel("Mutual exclusive score")
    plt.ylabel("Counts")
    plt.title("Gene pair mutual exclusivity histogram (MaxOVL)")
    plt.legend()
    plt.grid(False)
    plt.show()
    
'''

plot opposite gene pair graph based on networkx

'''

def oppo_graph(adata):

    print("plot the original graph")
    
    G = adata.uns['mule']["opposite graph"]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=30, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def graph_degree(adata, quantile_low=0.05, confidence=0.95):
    """
    Plot node degrees in opposite gene pair graph and estimate noise threshold
    """
    degrees = adata.uns['mule']["graph_degree"]['degree'].values
    n_nodes = len(degrees)

    # 只取高于低分位的节点来计算均值和标准差
    low_idx = int(n_nodes * quantile_low)
    degrees_sorted = np.sort(degrees)
    degree_m = np.mean(degrees_sorted[low_idx:])
    degree_s = np.std(degrees_sorted[low_idx:])

    # 计算阈值
    thre = stats.norm.ppf(confidence, degree_m, degree_s)
    thre = min(thre, degrees.max())

    # 绘图
    plt.figure(figsize=(6,4))
    counts, bins, _ = plt.hist(degrees, bins=30, color='skyblue', edgecolor='k')
    plt.axvline(thre, color='r', linestyle='--', label=f"Noise threshold: {thre:.2f}")
    plt.xlabel("Graph degree")
    plt.ylabel("Counts")
    plt.title("Graph node degree histogram")
    plt.legend()
    plt.grid(False)
    plt.show()

    return thre

'''
plot opposite gene pair graph after filter based on networkx

'''

def oppo_filter_graph(adata):
    
    print("plot the opposite filter graph")
    
    G = adata.uns['mule']["filter opposite subgraph"]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=30, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

'''

Show variance of merge gene sets to find a cut-off value for mode detection in opposite gene 
pair graph

'''

def cal_elbow(linkage):
    x = np.arange(linkage.shape[0])
    y = np.array(pd.DataFrame(linkage)[2])

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # 点到直线距离
    dis = np.abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    elbow_rank = np.argsort(dis)[::-1]
    return dis, elbow_rank

def merge_strategy(adata):
    G = adata.uns['mule']["filter opposite subgraph"]
    nn_m = pd.DataFrame(nx.to_numpy_array(G), index=G.nodes, columns=G.nodes)
    
    pdist = spc.distance.pdist(nn_m)
    linkage = spc.linkage(pdist, method='ward')
    adata.uns['mule']['linkage'] = linkage

    dis, elbow_rank = cal_elbow(linkage)

    y = pd.DataFrame(linkage)[2]
    plt.plot(y, marker='o', markersize=5, label='raw point')
    plt.scatter(elbow_rank[:10], y.iloc[elbow_rank[:10]], color='r', label='candidate elbow')
    plt.grid(False)
    plt.legend()
    plt.xlabel("Merge step")
    plt.ylabel("Merge gene set variance")
    plt.title("Elbow detection with ward metric")
    plt.xlim(max(linkage.shape[0]-20, 0), linkage.shape[0])
    plt.show()
    
    print("Linkage info (last 20 steps):")
    print(y.iloc[-20:])

'''

plot bipartite graph after we detect mode gene set.
We delete edges from genes belongs to the same mode

'''

def bipartite_graph(adata):

    print("Plot bipartite_graph")
    
    # draw the graph
    G_s = adata.uns['mule']["bipartite_graph"]
    pos = nx.spring_layout(G_s)

    community_list = adata.uns['mule']["bipartite_mode"]

    community_set = {}

    for i in range(len(community_list)):
        for j in range(len(community_list[i])):
            community_set[community_list[i][j]] = i
    partition = community_set 
            # color the nodes according to their partition

    pos = nx.spring_layout(G_s)
    pos = community_layout(G_s, community_set)

    partition = community_set
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values())+1)
    # cmap = cm.get_cmap('viridis',1)


    nx.draw_networkx_nodes(G_s, pos, partition.keys(), node_size=10,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_labels(G_s, pos=pos,font_size = 5)
    nx.draw_networkx_edges(G_s, pos, alpha=0.1)
    plt.grid(False)
    plt.show()

'''

plot Community based on networkx
The implementation reference from https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx

'''

def community_layout(g, partition):

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(self, g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos




def _position_communities( g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos



def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

'''

plot biparite embedding graph

'''

def mode_graph(adata):
    nx.draw(adata.uns['mule']["bipartite_embedding_graph"],with_labels = True)
    plt.title('Bipartite gene set graph')
    plt.show()

'''save taxonomy tree figure'''

def plot_taxonomy_tree(adata,save_path = "taxonomy_tree.png"):

    for i in adata.uns['mule']['taxonomy_tree_list']:
        graph = taxonomy_tree(i, adata.uns['mule']['mode info'])

        # show the figure and save
        Image(save_path)
        Image(graph.create_png())
    return graph

'''

Build Tree visulization function

'''

def taxonomy_tree(tree, dictionary, save_path = "taxonomy_tree.png"):
    '''
    Given a tree, and a dictionary that contains values corresponding to the tags of the tree nodes, generate a .png file that 
    shows the tree with the values on its nodes.
    
    Params
    -----------------------
    tree: treelib
        The tree to be visualized.
    dictionary: dictionary
        Dictionary with the tags of the tree nodes as keys, and their longer explanations as values.
    is_pdf: boolean, default(False)
        The file type to save.
         
        - False : Saves to .png file, default
        - True  : Saves to .pdf file.
    
    Returns
    -----------------------
    None.
    
    Saves
    -----------------------
    A detailed_tree.png(or .pdf) file, containing the visualized tree.
    '''
    import graphviz
    import subprocess
    import pydotplus
    import copy
    import os
    
    # copy the tree and the dict so that they stay the same
    tree_copy = copy.deepcopy(tree)
    dict_copy = copy.deepcopy(dictionary)
            
    # create a Dict, with node id as name and is_leaf/is_root as value
    positions = {}
    for node in tree_copy.all_nodes_itr():
        # 0 as root, 1 as leaf, 2 as node
        if node.is_leaf():
            positions[node.identifier] = 1
        elif node.is_root():
            positions[node.identifier] = 0
        else:
            positions[node.identifier] = 2
        
    # change the tag of the nodes to the values in the dict
    for node in tree_copy.all_nodes_itr():        
        
        # set node name
        string = node.tag + '\\n'
        
        # set node count of genes
        string += ('num of genes: '+ str(len(dict_copy[node.tag])) + '\\n')
        
        # if node is not root
        if node.is_root() == False:
            # set several genes in a row for better view, 1*2, 2*3, 3*4, etc.
            num_in_row = get_minimum_cover(len(dict_copy[node.tag]))
            for i in range(len(dict_copy[node.tag])):
                if i%num_in_row != 0:
                    string += ', '
                else:
                    string += '\\n'
                string += dict_copy[node.tag][i]
            
        # change the list to a string
        node.tag = string

    
    # Generate DOT code file
    tree_copy.to_graphviz("detailed_tree.dot", shape = 'Mrecord')
    
    # manipulate the dot file
    graph = pydotplus.graph_from_dot_file("detailed_tree.dot")
    nodes = graph.get_node_list()
    edges = graph.get_edge_list()
    
    # set the node styles
    for node in nodes:
        node.set_style("filled")
        node.set_fontname("Courier")
        node.set_penwidth(2)
        node.set_color("#CBE3C2")
        node.set_margin(0.12)
        
    # set the edge styles
    for edge in edges:
        edge.set_arrowsize(0.5)
        edge.set_color("#826B52")
    
    # set the special node styles
    for node in nodes:
        if positions[node.get_name()[1:len(node.get_name())-1]]==0:
            # node is root
            node.set_color("#6E4721")
        elif positions[node.get_name()[1:len(node.get_name())-1]]==1:
            # node is leaf
            node.set_color('#85A67A')
    
    # save the file
    if(save_path == False):
        Image(graph.create_png())
    else:
        graph.write_png(save_path)
    
    # remove the .dot file
        os.remove("detailed_tree.dot")

    return graph

def get_minimum_cover(count):
    for i in range(0, 20):
        if count < i*(i+2):
            return i
        
    # if count is too big(larger than 19*20=380), set counts in a row to be 20
    return 20
