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
# import cupy as cp
import time



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


def get_minimum_cover(count):
    for i in range(0, 20):
        if count < i*(i+2):
            return i
        
    # if count is too big(larger than 19*20=380), set counts in a row to be 20
    return 20

# def cal_elbow(linkage):
#     x = np.arange(linkage.shape[0])
#     y = np.array(pd.DataFrame(linkage)[2])

#     x1, y1 = x[0], y[0]
#     x2, y2 = x[-1], y[-1]

#     # 点到直线距离
#     dis = np.abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#     elbow_rank = np.argsort(dis)[::-1]
#     return dis, elbow_rank

# def merge_strategy_gpu(adata):
#     # 1. 构建邻接矩阵
#     G = adata.uns['mule']["filter opposite subgraph"]
#     nn_m = pd.DataFrame(nx.to_numpy_array(G), index=G.nodes, columns=G.nodes)
#     nn_m_values = nn_m.values  # CPU numpy array

#     # 2. GPU 计算距离矩阵
#     t0 = time.time()
#     nn_m_gpu = cp.asarray(nn_m_values)
#     # 计算 pdist (condensed distance)
#     diff = nn_m_gpu[:, None, :] - nn_m_gpu[None, :, :]
#     dist_matrix = cp.linalg.norm(diff, axis=2)
#     # 只取上三角矩阵
#     triu_indices = cp.triu_indices(nn_m_gpu.shape[0], k=1)
#     pdist_gpu = dist_matrix[triu_indices]
#     pdist = cp.asnumpy(pdist_gpu)  # 转回 CPU
#     t1 = time.time()
#     print(f"GPU pdist computation time: {t1 - t0:.4f} seconds")

#     # 3. CPU linkage
#     t0 = time.time()
#     linkage = spc.linkage(pdist, method='ward')
#     t1 = time.time()
#     print(f"CPU linkage computation time: {t1 - t0:.4f} seconds")

#     adata.uns['mule']['linkage'] = linkage

#     # 4. 计算 elbow
#     dis, elbow_rank = cal_elbow(linkage)

#     y = pd.DataFrame(linkage)[2]
#     plt.plot(y, marker='o', markersize=5, label='raw point')
#     plt.scatter(elbow_rank[:10], y.iloc[elbow_rank[:10]], color='r', label='candidate elbow')
#     plt.grid(False)
#     plt.legend()
#     plt.xlabel("Merge step")
#     plt.ylabel("Merge gene set variance")
#     plt.title("Elbow detection with Ward metric (GPU pdist)")
#     plt.xlim(max(linkage.shape[0]-20, 0), linkage.shape[0])
#     plt.show()

#     print("Linkage info (last 20 steps):")
#     print(y.iloc[-20:])

