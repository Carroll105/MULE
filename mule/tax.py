from ast import GeneratorExp
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager,cpu_count
import networkx as nx
import scipy
import scanpy as sc
import scipy.cluster.hierarchy as spc
import copy
import torch
from treelib import Tree
import random
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")


def mutually_exclusively_detect_CME(adata):
    adata.uns['mule'] = {}
    # data_df = adata.to_df()
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    X = adata.X.astype(np.int16).T
    res = pd.DataFrame(CME(X))
    res.index = adata.var_names
    res.columns = adata.var_names
    # res.index = data_df.columns
    # res.columns = data_df.columns
    adata.uns['mule']['opposite score df'] = res
    rows = res.index
    cols = res.columns
    upper_triangle_indices = np.triu_indices(len(res), k=1)
    gene_1, gene_2 = upper_triangle_indices
    coord_tuples = list(zip(adata.var_names[gene_1], adata.var_names[gene_2]))
    values = res.values[upper_triangle_indices]
    
    new_df = pd.DataFrame(values, index=coord_tuples)
    

    adata.uns['mule']['opposite score'] = new_df
    adata.uns['mule']['mutual exclusive method'] = 'CME'
    adata.uns['mule']['opposite score'].columns = ['mutual_exclusive_score']

from multiprocessing import Pool
from multiprocessing import Manager,cpu_count

def compute_min_expression(i, X, Y):
    min_exp = np.minimum(X[:, i][:, np.newaxis], Y)  
    return np.sum(min_exp, axis=0)  


@njit(parallel=True, fastmath=True, nopython=True)
def CME_numba(X):

    # _, gene_num = X.shape
    gene_num, _ = X.shape
    min_result = np.zeros((gene_num, gene_num), dtype=np.int64)
    (i_ind, j_ind) = np.triu_indices(gene_num)

    for k in prange(len(i_ind)):
        i = i_ind[k]
        j = j_ind[k]
        min_ary = np.minimum(X[i,:], X[j,:])
        min_result[j,i] = min_result[i,j] = sum(min_ary)      

    return min_result  

def CME(X):

    min_res = CME_numba(X)
    sum_x = np.sum(X, axis=1)
    ratio_x = min_res / sum_x[:, None]
    ratio_y = min_res / sum_x[None, :]
    result = 1 - np.maximum(ratio_x, ratio_y)

    return result.T


'''

Build opposite graph based on mutually exclusively pattern from gene pairs
Each edge from two nodes presents a mutually exclusively pattern

'''


def build_opposite_graph(adata, threshold=0.8):
    print("Build mutually exclusive gene graph")

    if 'mule' not in adata.uns or 'opposite score' not in adata.uns['mule']:
        raise ValueError("adata.uns['mule']['opposite score'] not found")
    
    df_angle = adata.uns['mule']['opposite score']
    if 'mutual_exclusive_score' not in df_angle.columns:
        raise ValueError("opposite score DataFrame must have 'mutual_exclusive_score' column")

    # 筛选 gene pair
    gene_pair_oppo = df_angle[df_angle['mutual_exclusive_score'] >= threshold][['mutual_exclusive_score']].copy()
    adata.uns['mule']['oppo gene pair test'] = gene_pair_oppo

    edge = list(gene_pair_oppo.index)
    G = nx.Graph()
    G.add_edges_from(edge)
    adata.uns['mule']['opposite graph'] = G

    print("Calculate original graph degree")
    df_de = pd.DataFrame(G.degree, columns=['gene','degree']).sort_values(by='degree', ascending=False)
    adata.uns['mule']['graph_degree'] = df_de


def get_subgraph(adata, threshold=10, relative=False):
    """
    Filter nodes with low degree to denoise.
    If relative=True, threshold is treated as a fraction (e.g., 0.01 means top 1%).
    """
    print("Filter low-degree nodes (denoise genes)")

    if 'mule' not in adata.uns or "graph_degree" not in adata.uns['mule']:
        raise ValueError("Run build_opposite_graph() before get_subgraph()")

    df_de = adata.uns['mule']["graph_degree"]

    if relative:
        cutoff = np.percentile(df_de['degree'], 100*(1-threshold))
    else:
        cutoff = threshold

    keep_nodes = df_de.loc[df_de['degree'] >= cutoff, 'gene']
    G_sub = adata.uns['mule']['opposite graph'].subgraph(keep_nodes).copy()

    adata.uns['mule']["filter opposite subgraph"] = G_sub
    # return G_sub

'''

For the gene pair graph, we want to estimate bipartite structure of graphs. We use adjcent matrix 
for hierarchical clustering to get different modes.

'''

from statsmodels.stats.multitest import multipletests

def bipartite_embedding_perm_fast(adata, merge_threshold, bipartite_alpha=0.05, n_perm=500, seed=42):
    """
    Fast permutation-based bipartite embedding using adjacency matrix vectorization
    """
    np.random.seed(seed)
    G = adata.uns['mule']["filter opposite subgraph"]

    # -----------------------
    # Step1: adjacency matrix and clustering
    # -----------------------
    nodes = list(G.nodes)
    node_idx_map = {n:i for i,n in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes)
    pdist = spc.distance.pdist(A)
    linkage = spc.linkage(pdist, method='ward')
    idx = spc.fcluster(linkage, merge_threshold, criterion='distance')
    bi_p = [np.array([i for i, cl in enumerate(idx) if cl == cl_val]) for cl_val in np.unique(idx)]

    # -----------------------
    # Step2: remove intra-mode edges
    # -----------------------
    H = G.copy()
    for cluster in bi_p:
        intra_edges = [(nodes[i], nodes[j]) for i in cluster for j in cluster if i < j and H.has_edge(nodes[i], nodes[j])]
        H.remove_edges_from(intra_edges)
    A_H = nx.to_numpy_array(H, nodelist=nodes)

    adata.uns['mule']["bipartite_mode"] = [[nodes[i] for i in cluster] for cluster in bi_p]
    adata.uns['mule']["bipartite_graph"] = H

    # -----------------------
    # Step3: vectorized permutation test
    # -----------------------
    n_nodes = len(nodes)
    mode_pairs = [(i,j) for i in range(len(bi_p)) for j in range(i+1,len(bi_p))]
    obs_edges_list = []

    # 计算每对 mode 的观测跨边数
    for i,j in mode_pairs:
        mask_i = bi_p[i][:,None]
        mask_j = bi_p[j][None,:]
        obs_edges = np.sum(A_H[mask_i, mask_j])
        obs_edges_list.append(obs_edges)

    # 生成 null 分布矩阵: n_perm x n_mode_pairs
    null_counts = np.zeros((n_perm, len(mode_pairs)), dtype=int)
    for p in range(n_perm):
        perm_idx = np.random.permutation(n_nodes)
        A_perm = A_H[np.ix_(perm_idx, perm_idx)]
        for k, (i,j) in enumerate(mode_pairs):
            mask_i = perm_idx[bi_p[i]]
            mask_j = perm_idx[bi_p[j]]
            null_counts[p,k] = np.sum(A_perm[np.ix_(mask_i, mask_j)])

    # 计算 p 值
    obs_array = np.array(obs_edges_list)
    pvals = (np.sum(null_counts >= obs_array[None,:], axis=0) + 1) / (n_perm + 1)

    # FDR 矫正
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=bipartite_alpha, method='fdr_bh')

    # 构建 mode-level bipartite graph
    G_e = nx.Graph()
    mode_info = {}
    for i, cluster in enumerate(bi_p):
        mode_info[f"mode {i}"] = [nodes[idx] for idx in cluster]

    for r, (i,j) in zip(reject, mode_pairs):
        if r:
            G_e.add_edge(f"mode {i}", f"mode {j}")

    adata.uns['mule']['mode info'] = mode_info
    adata.uns['mule']["bipartite_embedding_graph"] = G_e

    if len(G_e.nodes) == 0:
        print("There are no significant bipartite embeddings in the graph")

    # return adata


'''
Build taxonomy tree based on bipartite embedding graph

'''

def build_taxonomy_tree(adata):

    # bipartite_graph = adata.uns['mule']["bipartite_graph"]
    # bi_p = adata.uns['mule']['bipartite_mode']
    # graph = copy.deepcopy(bipartite_graph)
    # graph = nx.complement(graph)
    
    graph = adata.uns['mule']["bipartite_embedding_graph"]
    mode_info = adata.uns['mule']['mode info']
    noise_mode = detect_noise_mode(graph,mode_info)
    adata.uns['mule']['noise_mode'] = noise_mode
    graph = nx.complement(graph)
    if(len(graph.nodes)-1 > pd.DataFrame(graph.degree)[1].max()):
        print("Create a root marker set")
        graph.add_node("root set")
        mode_info['root set'] = list(adata.uns['mule']['filter opposite subgraph'].nodes)
        for i in graph.nodes:
            if(i!="root set"):
                graph.add_edge("root set",i)
    # nx.draw(graph,with_labels = True)
    tree_level = BFS_min_span(adata , graph)
    # print(tree_level)
    adata.uns['mule']['mode info'] = mode_info 


    ## give a multiple tree list
    # a, b = test_taxonomy_tree(tree_level,graph)
    # b.append(a)
    # new_tree_level_list = b

    taxonomy_tree_list = []

    taxonomy_tree = build_tree(tree_level,graph,adata)
    taxonomy_tree.show()

    for node in taxonomy_tree.all_nodes_itr():
        node.data = mode_info[node.tag]
    taxonomy_tree_list.append(taxonomy_tree)


    ## give a multiple tree list

    # if(len(new_tree_level_list) == 1):
    #     taxonomy_tree = build_tree(tree_level,graph)
    #     taxonomy_tree.show()
    #     taxonomy_tree_list.append(taxonomy_tree)

    # else:
    #     rank_num = []
    #     for i in new_tree_level_list:
    #         rank_num.append(len(i['2']))
    #     sort_num = np.argsort(np.array(rank_num))
    #     for i in range(len(new_tree_level_list)):
    #         taxonomy_tree_list.append(build_tree(new_tree_level_list[sort_num[i]],graph))
    adata.uns['mule']['taxonomy_tree_list'] = taxonomy_tree_list
    adata.uns['mule']['taxonomy_tree'] = taxonomy_tree
    adata.uns['mule']['tree_level'] = tree_level

    return 


def move_nodes_to_ancestor(adata, tree):

    corr_df = adata.to_df().T.loc[list(adata.uns['mule']['filter opposite subgraph'].nodes)].T.corr()
    mode_info = adata.uns['mule']['mode info']

    all_nodes_satisfy = False
    counts = 0

    while not all_nodes_satisfy and counts <=100:

        counts = counts + 1
        all_nodes_satisfy = True  # 假设所有节点已满足条件
        for node_id in tree.expand_tree(mode=Tree.DEPTH):
            current_node = tree.get_node(node_id)
    #         print(current_node.identifier)
            parent_id = tree.parent(current_node.identifier)

            if parent_id is None:
                continue  # 如果当前节点是根节点，跳过

            if(parent_id.identifier == 'root set'):
                continue

            parent_node = tree.get_node(parent_id.identifier)
            if len(tree.children(parent_node.identifier)) >= 2 and mutuall_express(tree,corr_df,mode_info,current_node):
                continue
            else:
    #             print("X")
                tree.move_node(current_node.identifier, tree.parent(parent_node.identifier).identifier)
                all_nodes_satisfy = False  # 当有节点移动时，说明还有节点不满足条件
                break  # 重新开始循环遍历
    
    return tree


def mutuall_express(tree, corr_df, mode_info, current_node):
    parent_id = tree.parent(current_node.identifier)
    children_nodes = tree.children(parent_id.identifier)
    tmp = True
    for i in range(len(children_nodes)):
        child = children_nodes[i]
        if(child == current_node):
            continue
        else:
            v = corr_df.loc[mode_info[current_node.identifier]].T.loc[mode_info[child.identifier]].mean().mean()
#             print(v)
            if(v > 0.05):
              tmp = False
    v2 = corr_df.loc[mode_info[current_node.identifier]].T.loc[mode_info[parent_id.identifier]].mean().mean()
    if(v2 < 0):
        tmp = False
    return tmp
    

def find_ancestor_with_multiple_children(tree, node_id):
    node = tree.get_node(node_id)
    while node:
        parent_id = node.predecessor(tree.identifier)
        if parent_id is None:
            return None
        parent_node = tree.get_node(parent_id)
        siblings = tree.children(parent_id)
        if len(siblings) >= 2:
            return parent_node
        node = parent_node

    return None

'''

Detect whether it has a noise mode in this bipartite embedding graph structure

'''


def detect_noise_mode(graph,mode):
    noise_mode = {}
    noise_key =  set(mode.keys()) - set(graph.nodes)
    for i in noise_key:
        noise_mode[i] = mode[i]
    return noise_mode


'''

BFS for modes assign in different tree level

'''

def BFS_min_span(adata, graph):

    looked = set()
    G = graph
    degree_df = pd.DataFrame(G.degree).sort_values(by = 1, ascending = False)
    degree_df.index = np.array(degree_df[0])
    before_degree = copy.deepcopy(degree_df)
    tree_level = {}
    i = 0
    root_nodes = degree_df.index[0]
    # print(degree_df)
    tree_level[str(i)] = [root_nodes]
    looked.add(root_nodes)
    new_G = copy.deepcopy(G)
    merge_info = {}
    while(len(looked) < len(G.nodes)):
        i = i + 1
        test_G = copy.deepcopy(G)
        test_G.remove_nodes_from(looked)
        S = [test_G.subgraph(c).copy() for c in nx.connected_components(test_G)]
        level_nodes = []
        for j in S:
            degree_df = pd.DataFrame(j.degree)
            degree_df.index = degree_df[0]
            node = degree_df.sort_values(by = 1, ascending= False).index[0]
            node_max = degree_df.sort_values(by = 1, ascending= False)[1].iloc[0]
            if(node_max == -1):
                node_list = list(degree_df[degree_df[1] == node_max].index)
                for g in node_list:
                    looked.add(g)
                if(before_degree.loc[node_list[0]] > before_degree.loc[node_list[1]]):
                    new_G.remove_node(node_list[0])
                    level_nodes.append(node_list[1])
                    tree_level[str(i)] = level_nodes
                    merge_info[node_list[1]] = node_list[0]
                else:
                    new_G.remove_node(node_list[1])
                    level_nodes.append(node_list[0])
                    tree_level[str(i)] = level_nodes
                    merge_info[node_list[1]] = node_list[1]
            if(node_max > 1):
                node_list = list(degree_df[degree_df[1] == node_max].index)
                for g in node_list:
                    looked.add(g)
                    level_nodes.append(g)
                tree_level[str(i)] = level_nodes
            else:
                if(node_max == 1):
                    node_list = list(degree_df[degree_df[1] == node_max].index)
                    if(before_degree.loc[node_list[0]][1] > before_degree.loc[node_list[1]][1]):
                        looked.add(node_list[0])
                        level_nodes.append(node_list[0])
                        tree_level[str(i)] = level_nodes
                    else:
                        looked.add(node_list[1])
                        level_nodes.append(node_list[1])
                        tree_level[str(i)] = level_nodes
                else:
                    looked.add(node)
                    level_nodes.append(node)
                    tree_level[str(i)] = level_nodes
    adata.uns['mule']['tree_level'] = tree_level
    return tree_level



def test_taxonomy_tree(tree_level, graph):
    tree_leaves = tree_level[str(len(tree_level)-1)]
    # looked = np.zeros(len(tree_level))
    # high_level = tree_level[str(len(tree_level)-2)]
    # new_high_level = []
    # while(len(np.where(looked == 1)) < len(looked)):
    #     tmp = high_level
    delete_level = []
    for i in range(1,len(tree_level)-1):
        tmp_list = tree_level[str(i)]
        if(len(tmp_list) > 1):
            continue
        else:
            for j in tree_leaves:
                if(graph.has_edge(tmp_list[0],j)):
                    delete_level.append(i)
    set_delete = set(np.unique(np.array(delete_level)).astype("str"))
    new_tree_level = {}
    tmp = 0
    for j in range(len(tree_level)):
        if(str(j) not in set_delete):
            new_tree_level[str(tmp)] = tree_level[str(j)]
            tmp = tmp + 1
    # find delete tree level correlation
    new_tree_level_list = []
    tmp = 0
    for j in set_delete:
        new_tree_level_list.append({})
        new_tree_level_list[tmp]['0'] = ['root set']
        new_tree_level_list[tmp]['1'] = [tree_level[str(j)][0]]
        new_tree_level_list[tmp]['2'] = []
        for i in tree_leaves:
            if(graph.has_edge(i,tree_level[str(j)][0])):
                new_tree_level_list[tmp]['2'].append(i)
            else:
                new_tree_level_list[tmp]['1'].append(i)
    return new_tree_level,new_tree_level_list

def build_tree(tree_level, graph, adata):
    G = graph
    tree = Tree()
    looked = set()
    for i in tree_level:
        if(i == '0'):
            tree.create_node(tree_level[i][0],tree_level[i][0])
            looked.add("root set")
        else:
            last_level_nodes = tree_level[str(int(i)-1)]
            now_level_nodes = tree_level[i]

            # if(len(now_level_nodes) != 1):

            #     for ii in now_level_nodes:
            #         for jj in last_level_nodes:
            #             if(G.has_edge(ii,jj) and (ii not in looked)):
            #                 tree.create_node(ii,ii, parent=jj)
            #                 looked.add(ii)

            if(len(now_level_nodes) != 1):
                for ii in now_level_nodes:
                    have_partent = False
                    for jj in last_level_nodes:
                        if(G.has_edge(ii,jj) and (ii not in looked)):
                            tree.create_node(ii,ii, parent=jj)
                            looked.add(ii)
                            have_partent = True
                    if(not have_partent):
                        tree.create_node(ii,ii, parent=tree_level['0'][0])
                        looked.add(ii)
            else:
                bool_flag = True
                for ii in now_level_nodes:
                    for jj in last_level_nodes:
                        if(G.has_edge(ii,jj) and (ii not in looked)):
                            tree.create_node(ii,ii, parent=jj)
                            looked.add(ii)
                            bool_flag = False
                if(bool_flag):
                    tree.create_node(ii,ii,parent=last_level_nodes[0])
                    looked.add(ii)

    tree = move_nodes_to_ancestor(adata,tree)

    return tree

def get_n_cores(max_cores=8):
    """

    """
    available_cores = cpu_count()
    return min(available_cores, max_cores)

# ------------------ permutation worker ------------------
def perm_cme_worker(seed, X, fdr_threshold):
    np.random.seed(seed)
    X_perm = X.copy()
    n_genes, n_cells = X_perm.shape
    for i in range(n_genes):
        np.random.shuffle(X_perm[i, :])
    score_perm = CME(X_perm)
    upper_tri_idx = np.triu_indices(n_genes, k=1)
    values_perm = score_perm[upper_tri_idx]
    return np.sum(values_perm > fdr_threshold)

# ------------------ scPurity 计算主函数 ------------------
def cal_scpurity_cme_parallel(adata, fdr_threshold=0.05, n_perm=1000, max_cores=8):
    """
    计算基于 CME 互斥分数的 scPurity（支持 permutation test 和多进程）
    
    adata: AnnData 对象，需先调用 mutually_exclusively_detect_CME 生成 CME 分数
    fdr_threshold: 显著互斥对阈值
    n_perm: permutation 次数
    max_cores: 最大使用 CPU 核心数
    """
    print("Calculate scPurity using CME scores with parallel permutation test")
    
    # 原始互斥分数
    df_score = adata.uns['mule']['opposite score']
    scores = df_score['mutual_exclusive_score'].values
    observed_count = np.sum(scores > fdr_threshold)
    total_pairs = len(scores)
    purity = observed_count / total_pairs if total_pairs > 0 else 1.0
    
    # permutation
    X = adata.X.astype(np.int16).T  # genes x cells
    n_cores = get_n_cores(max_cores)
    seeds = np.random.randint(0, 1e6, size=n_perm)
    
    with Pool(n_cores) as pool:
        perm_counts = pool.starmap(perm_cme_worker, [(s, X, fdr_threshold) for s in seeds])
    
    perm_counts = np.array(perm_counts)
    p_value = (np.sum(perm_counts >= observed_count) + 1) / (n_perm + 1)
    
    adata.uns['mule']['purity'] = {'value': purity, 'p_value': p_value}
    print("scPurity:", purity, "p_value:", p_value)

import numpy as np
import pandas as pd
from scipy import sparse

def mule_umap(adata, mode_list, mutual_degree=1.0, min_dist=0.5, spread=1.0):
    """
    Custom UMAP embedding based on mutually exclusive gene modes.
    adata: AnnData object
    mode_list: list of mode names in adata.uns['mule']['mode info']
    mutual_degree: value to set for mutually exclusive cells in neighbor graph
    """
    # 获取互斥基因集合
    mutual_gene_set = [adata.uns['mule']['mode info'][mode] for mode in mode_list]
    marker_gene_set = list(adata.uns['mule']['filter opposite subgraph'].nodes)

    # 构建临时 AnnData
    data = adata[:, marker_gene_set].X  # cells x genes
    adata_tmp = sc.AnnData(data.copy())
    sc.pp.normalize_total(adata_tmp, target_sum=np.sum(data))
    sc.pp.log1p(adata_tmp)
    sc.tl.pca(adata_tmp, svd_solver='arpack')

    # 构建邻接矩阵
    n_pcs = adata_tmp.obsm['X_pca'].shape[1]
    sc.pp.neighbors(adata_tmp, n_neighbors=20, n_pcs=n_pcs)
    connectivities = adata_tmp.obsp['connectivities'].tocoo(copy=True).tolil()

    # 构建细胞-互斥模式矩阵
    mutuall_df = np.zeros((adata_tmp.n_obs, len(mutual_gene_set)), dtype=int)
    for i, genes in enumerate(mutual_gene_set):
        gene_idx = [adata_tmp.var_names.get_loc(g) for g in genes if g in adata_tmp.var_names]
        if not gene_idx:
            continue
        expr = adata_tmp.X[:, gene_idx].toarray() if sparse.issparse(adata_tmp.X) else adata_tmp.X[:, gene_idx]
        cell_sum = expr.sum(axis=1)
        mean_val = cell_sum.mean()
        mutuall_df[:, i] = (cell_sum > mean_val).astype(int)

    # 修改邻接矩阵中互斥细胞的连接度
    for i in range(connectivities.shape[0]):
        neighbors = connectivities.rows[i]
        if not neighbors:
            continue
        mut_i = mutuall_df[i]
        for j in neighbors:
            mut_j = mutuall_df[j]
            if not np.array_equal(mut_i, mut_j):
                connectivities[i, j] = mutual_degree

    adata_tmp.obsp['connectivities'] = connectivities.tocsr()
    sc.tl.umap(adata_tmp, min_dist=min_dist, spread=spread)

    # 保存 UMAP 结果
    adata.obsm['X_umap_mule'] = adata_tmp.obsm['X_umap']
    adata.uns['mule']['mule_X_umap'] = adata_tmp.obsm['X_umap']
    # adata.uns['neighbors'] = adata_tmp.uns['neighbors']
    adata.obsp = adata_tmp.obsp

    # print("mule_umap finished, shape:", adata.obsm['X_umap_mule'].shape)



import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from pathlib import Path
import scanpy as sc
from treelib import Tree
import matplotlib.pyplot as plt


def get_labels_from_tree(adata, gene_tree, score_func, key='X_pca'):
    """
    从 gene module tree 给每个细胞打分并确定标签（稳健版）
    - 使用 node.data 作为模块基因列表
    - 与 adata.var_names 求交，避免 KeyError
    - 空模块打 0 分
    """
    nodes = sorted(gene_tree.all_nodes(), key=lambda n: n.identifier)
    modules = [(n.tag, n.data) for n in nodes]  # (模块名, 基因列表)

    # 2) 逐模块打分
    score_list = []
    module_names = []
    var_set = set(map(str, adata.var_names))  # 统一为字符串比较

    for tag, genes in modules:
        # 规范化 genes
        if genes is None:
            genes = []
        elif isinstance(genes, str):
            genes = [genes]
        else:
            genes = list(genes)

        # 与 var_names 求交（避免 KeyError）
        genes_in = [g for g in genes if str(g) in var_set]

        if len(genes_in) == 0:
            # 该模块在数据中无有效基因 → 记 0 分
            s = np.zeros(adata.n_obs, dtype=float)
        else:
            s = np.asarray(score_func(genes_in, adata)).ravel()
            # 防御：确保长度匹配
            if s.shape[0] != adata.n_obs:
                raise ValueError(f"score_func 返回长度 {s.shape[0]} 与细胞数 {adata.n_obs} 不一致")

        score_list.append(s)
        module_names.append(tag)

    scores = np.vstack(score_list)

    labels_idx = np.argmax(scores, axis=0)
    labels = np.array([module_names[i] for i in labels_idx])
    return labels

# =============================
# ② 样本池构建
# =============================
def build_pos_neg_pools(X, labels, n_neighbors=16):
    """
    构建正负样本池
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    knn_idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    pos_pool = []
    for i in range(X.shape[0]):
        same = [j for j in knn_idx[i] if labels[j] == labels[i]]
        pos_pool.append(same)

    neg_pool = defaultdict(list)
    unique_labels = np.unique(labels)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li != lj:
                neg_pool[i].append(j)
    return pos_pool, neg_pool


# =============================
# ③ Triplet 采样与 Loss
# =============================
def sample_triplets(pos_pool, neg_pool, batch_size, rng=np.random.default_rng()):
    anchors, pos, neg = [], [], []
    n_samples = len(pos_pool)
    while len(anchors) < batch_size:
        i = rng.integers(0, n_samples)
        if not pos_pool[i] or not neg_pool[i]:
            continue
        j = rng.choice(pos_pool[i])
        k = rng.choice(neg_pool[i])
        anchors.append(i); pos.append(j); neg.append(k)
    return torch.tensor(anchors), torch.tensor(pos), torch.tensor(neg)


def margin_vector(labels, a, n, strong_pairs, M_STRONG=3.0, M_WEAK=0.5, device='cpu'):
    mlist = []
    for ai, ni in zip(a.tolist(), n.tolist()):
        pair = {labels[ai], labels[ni]}
        if pair in strong_pairs:
            mlist.append(M_STRONG)
        else:
            mlist.append(M_WEAK)
    return torch.tensor(mlist, dtype=torch.float32, device=device)


def triplet_loss(Z, a, p, n, labels, strong_pairs, alpha=1.0, beta=0.3):
    d_ap = (Z[a] - Z[p]).norm(dim=1)
    d_an = (Z[a] - Z[n]).norm(dim=1)
    mvec = margin_vector(labels, a, n, strong_pairs, device=Z.device)
    trip = torch.relu(d_ap - d_an + mvec)
    compact = d_ap
    return (alpha * trip + beta * compact).mean()


# # =============================
# # ④ 主训练流程
# # =============================
# def train_triplet_embedding(
#     adata, label_source, output_dir,
#     epochs=600, batch_size=4096, lr=0.006, strong_pairs=None
# ):
#     """
#     Triplet-Loss 表达空间训练
#     参数：
#         label_source: 直接传 label array 或 gene_tree + score_func 组合
#         strong_pairs: 强排斥标签对（集合）
#     """
#     out_dir = Path(output_dir)
#     out_dir.mkdir(exist_ok=True)

#     if isinstance(label_source, np.ndarray):
#         labels = label_source
#     else:
#         gene_tree, score_func = label_source
#         labels = get_labels_from_tree(adata, gene_tree, score_func)
#     adata.obs['label'] = labels
#     X = adata.obsm['X_pca'].astype('float32')
#     pos_pool, neg_pool = build_pos_neg_pools(X, labels)
#     X_torch = torch.as_tensor(X)
#     n_samples, d = X.shape

#     W = torch.eye(d, d, requires_grad=True)
#     opt = torch.optim.AdamW([W], lr=lr)

#     rng = np.random.default_rng()

#     for ep in range(epochs + 1):
#         a, p, n = sample_triplets(pos_pool, neg_pool, batch_size, rng)
#         Z = X_torch @ W.T
#         loss = triplet_loss(Z, a, p, n, labels, strong_pairs)

#         opt.zero_grad(); loss.backward(); opt.step()

#         if ep % 60 == 0:
#             adata.obsm['X_custom'] = Z.detach().cpu().numpy()
#             sc.pp.neighbors(adata, use_rep='X_custom', n_neighbors=15)
#             sc.tl.umap(adata, random_state=42)
#             fig, ax = plt.subplots(figsize=(6.5, 6.5))
#             sc.pl.umap(adata, color='label', linewidth=1, s=166, ax=ax, show=False)
#             fig.savefig(out_dir / f"umap_ep{ep:03d}.png", dpi=300, bbox_inches='tight')
#             plt.close(fig)
#             adata.write(out_dir / f"adata_ep{ep:03d}.h5ad", compression='gzip')

#         if ep % 20 == 0:
#             print(f"Ep {ep:4d}/{epochs} | loss={loss.item():.4f}")

def score_func(module_genes, adata):
    return np.asarray(adata[:, module_genes].X.mean(axis=1)).ravel()

def train_triplet_embedding(
    adata, tree, 
    epochs=600, batch_size=4096, lr=0.006, strong_pairs=None,
    device=None,
    score_func = score_func
):
    # out_dir = Path(output_dir)
    # out_dir.mkdir(exist_ok=True)
    label_source = (tree,score_func)
    if isinstance(label_source, np.ndarray):
        labels = label_source
    else:
        gene_tree, score_func = label_source
        labels = get_labels_from_tree(adata, gene_tree, score_func)
    adata.obs['label'] = labels
    X = adata.obsm['X_pca'].astype('float32')
    pos_pool, neg_pool = build_pos_neg_pools(X, labels)
    
    # 设备选择：优先 GPU
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_torch = torch.as_tensor(X, device=device)   # 放到 device 上
    n_samples, d = X.shape

    # 初始化投影矩阵 W（可学习）
    W = torch.eye(d, d, dtype=torch.float32, device=device)
    W.requires_grad_(True)

    opt = torch.optim.AdamW([W], lr=lr)

    rng = np.random.default_rng()
    if strong_pairs is None:
        strong_pairs = set()

    for ep in range(epochs + 1):
        a, p, n = sample_triplets(pos_pool, neg_pool, batch_size, rng)
        # 确保索引 tensors 在同一 device（sample_triplets 返回 CPU tensors）
        a = a.to(device); p = p.to(device); n = n.to(device)

        Z = X_torch @ W.T   # (N, d) @ (d, d).T -> (N, d)
        loss = triplet_loss(Z, a, p, n, labels, strong_pairs)

        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 60 == 0:
            adata.obsm['X_custom'] = Z.detach().cpu().numpy()
            sc.pp.neighbors(adata, use_rep='X_custom', n_neighbors=15)
            sc.tl.umap(adata, random_state=42)
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            sc.pl.umap(adata, color='label', linewidth=1, s=166, ax=ax, show=False)
            # fig.savefig(out_dir / f"umap_ep{ep:03d}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            # adata.write(out_dir / f"adata_ep{ep:03d}.h5ad", compression='gzip')

        if ep % 20 == 0:
            print(f"Ep {ep:4d}/{epochs} | loss={loss.item():.4f}")

