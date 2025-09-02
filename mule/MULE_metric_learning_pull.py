"""
Gene-tree 驱动的度量学习（Triplet + Reconstruction）
输入：adata（含表达矩阵与 PCA）、pca_key、treelib.Tree
内部步骤：
  1) 由 gene tree 计算各节点 meta-gene 表达
  2) 给每个细胞分配 label
  3) 基于 tree 拓扑自动构建正/负样本池（同标签 KNN 为正；不同关系为负）
  4) 训练投影矩阵 W，联合 Triplet Loss 与 λ·Reconstruction Loss

主入口：
  train_triplet_model_from_tree(adata, tree, pca_key='X_pca', ...)

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from treelib import Tree
from typing import Dict, List, Tuple, Set, Optional


# ----------------------------
# 1) gene tree -> meta 表达 -> labels
# ----------------------------
def _nodes_with_genes(tree: Tree) -> List[str]:
    tags = []
    for node in tree.all_nodes_itr():
        if isinstance(node.data, list) and len(node.data) > 0:
            tags.append(node.tag)
    return tags


def _leaf_tags_with_genes(tree: Tree) -> List[str]:
    tags = []
    for node in tree.leaves():
        if isinstance(node.data, list) and len(node.data) > 0:
            tags.append(node.tag)
    return tags


def assign_labels_from_tree(
    adata,
    tree: Tree,
    use_leaves_only: bool = True,
    aggregator: str = "mean",
    min_valid_genes: int = 1
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    X_df: pd.DataFrame = adata.to_df()  # cells × genes
    nodes_considered = _leaf_tags_with_genes(tree) if use_leaves_only else _nodes_with_genes(tree)

    meta_expr = {}
    for node in tree.all_nodes_itr():
        tag = node.tag
        if tag not in nodes_considered:
            continue
        genes = node.data if isinstance(node.data, list) else []
        valid = [g for g in genes if g in X_df.columns]
        if len(valid) < min_valid_genes:
            continue
        if aggregator == "mean":
            meta_expr[tag] = X_df[valid].mean(axis=1)
        elif aggregator == "sum":
            meta_expr[tag] = X_df[valid].sum(axis=1)
        else:
            raise ValueError(f"未知 aggregator: {aggregator}")

    if not meta_expr:
        raise ValueError("有效 meta 表达为空，请检查 gene tree 的基因是否在 adata 基因中出现。")

    meta_df = pd.DataFrame(meta_expr, index=X_df.index)  # cells × nodes
    labels = meta_df.idxmax(axis=1).values
    nodes_kept = list(meta_df.columns)  # 过滤后真正参与打分的节点
    return labels, meta_df, nodes_kept


# ----------------------------
# 2) 基于 tree 的关系映射（祖先/同父/顶层分支等）
# ----------------------------
def build_tree_relations(tree: Tree, node_tags: List[str]) -> dict:
    # tag -> node_id
    tag2id = {}
    for n in tree.all_nodes_itr():
        tag2id[n.tag] = n.identifier

    def _parent_tag(tag: str) -> Optional[str]:
        nid = tag2id[tag]
        p = tree.parent(nid)
        return None if p is None else p.tag

    def _ancestors(tag: str) -> Set[str]:
        nid = tag2id[tag]
        return set([a.tag for a in tree.ancestors(nid)])

    def _siblings(tag: str) -> Set[str]:
        nid = tag2id[tag]
        sibs = tree.siblings(nid)
        return set([s.tag for s in sibs if s.tag in node_tags])

    root_id = tree.root

    def _top_branch(tag: str) -> str:
        nid = tag2id[tag]
        cur = nid
        parent = tree.parent(cur)
        if parent is None:
            return tree.get_node(cur).tag
        # 若 parent 已是根，则当前就是一级分支
        if parent.identifier == root_id:
            return tree.get_node(cur).tag
        # 否则向上直到 parent 是根
        while tree.parent(cur) is not None and tree.parent(cur).identifier != root_id:
            cur = tree.parent(cur).identifier
        return tree.get_node(cur).tag

    parent_map = {t: _parent_tag(t) for t in node_tags}
    ancestors_map = {t: _ancestors(t) for t in node_tags}
    siblings_map = {t: _siblings(t) for t in node_tags}
    top_branch_map = {t: _top_branch(t) for t in node_tags}
    all_labels = set(node_tags)

    # 预计算 descendants（通过 ancestors 反向汇总）
    descendants_map = {t: set() for t in node_tags}
    for t in node_tags:
        for u in node_tags:
            if t in ancestors_map[u]:
                descendants_map[t].add(u)

    return dict(
        parent_map=parent_map,
        ancestors_map=ancestors_map,
        siblings_map=siblings_map,
        top_branch_map=top_branch_map,
        descendants_map=descendants_map,
        all_labels=all_labels
    )


# ----------------------------
# 3) 正/负样本池（标签层面自动推导）
# ----------------------------
def build_label_to_indices(labels: np.ndarray) -> Dict[str, List[int]]:
    """label -> [indices]"""
    m = defaultdict(list)
    for i, lab in enumerate(labels):
        m[lab].append(i)
    return m


def build_pos_neg_pools(
    X_np: np.ndarray,
    labels: np.ndarray,
    relations: dict,
    n_neighbors: int = 16,
    neg_strategy: str = "relation_based"
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """
    正样本：同 label 的 KNN（若为空则退化为同 label 的全体）
    负样本：根据 tree 关系自动构建（两种策略）
      - 'all_other_labels': neg = 所有不同 label
      - 'relation_based' : neg = siblings ∪ 不同 top_branch ∪ 祖先 ∪ 子孙
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_np)
    knn_idx = nbrs.kneighbors(return_distance=False)[:, 1:]

    label_to_ids = build_label_to_indices(labels)
    all_labels: Set[str] = relations["all_labels"]
    siblings_map = relations["siblings_map"]
    top_branch_map = relations["top_branch_map"]
    ancestors_map = relations["ancestors_map"]
    descendants_map = relations["descendants_map"]

    # 预构建每个 label 对应的 neg label 集合
    if neg_strategy == "all_other_labels":
        neg_labels_map = {lab: (all_labels - {lab}) for lab in all_labels}
    elif neg_strategy == "relation_based":
        neg_labels_map = {}
        for lab in all_labels:
            # 不同顶层分支
            cross = set([x for x in all_labels if top_branch_map[x] != top_branch_map[lab]])
            # 同父兄弟
            sibs = set(siblings_map[lab])
            # 祖先与子孙
            anc = set(ancestors_map[lab])
            dec = set(descendants_map[lab])
            nl = (cross | sibs | anc | dec) - {lab}
            # 若过小，兜底为 all other
            neg_labels_map[lab] = nl if len(nl) > 0 else (all_labels - {lab})
    else:
        raise ValueError(f"未知 neg_strategy: {neg_strategy}")

    # 细胞层面的正/负池
    n = X_np.shape[0]
    pos_pool: List[List[int]] = [[] for _ in range(n)]
    neg_pool: Dict[int, List[int]] = defaultdict(list)

    for i in range(n):
        lab = labels[i]
        same_label_knn = [j for j in knn_idx[i] if labels[j] == lab]
        if len(same_label_knn) == 0:
            fallback = [j for j in label_to_ids[lab] if j != i]
            pos_pool[i] = fallback
        else:
            pos_pool[i] = same_label_knn

    for i in range(n):
        lab = labels[i]
        neg_labels = neg_labels_map[lab]
        ids = []
        for nl in neg_labels:
            ids.extend(label_to_ids[nl])
        # 避免负样本里包含自己
        neg_pool[i] = [j for j in ids if j != i]

    return pos_pool, neg_pool


# ----------------------------
# 4) Triplet & Reconstruction
# ----------------------------
def sample_triplets(
    n_samples: int,
    pos_pool: List[List[int]],
    neg_pool: Dict[int, List[int]],
    batch_size: int,
    rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors, pos, neg = [], [], []
    while len(anchors) < batch_size:
        i = rng.integers(0, n_samples)
        if len(pos_pool[i]) == 0 or len(neg_pool[i]) == 0:
            continue
        j = rng.choice(pos_pool[i])
        k = rng.choice(neg_pool[i])
        anchors.append(i); pos.append(j); neg.append(k)
    return torch.tensor(anchors), torch.tensor(pos), torch.tensor(neg)


def _margin_between_labels(
    la: str,
    lb: str,
    relations: dict,
    M_STRONG: float,
    M_DEFAULT: float,
    M_WEAK: float
) -> float:
    if la == lb:
        return 0.0
    parent_map = relations["parent_map"]
    ancestors_map = relations["ancestors_map"]
    top_branch_map = relations["top_branch_map"]

    if (la in ancestors_map[lb]) or (lb in ancestors_map[la]):
        return M_WEAK
    if parent_map.get(la, None) is not None and parent_map.get(la) == parent_map.get(lb):
        return M_STRONG
    if top_branch_map[la] != top_branch_map[lb]:
        return M_DEFAULT
    return M_DEFAULT


def margin_vector(
    idx_a: torch.Tensor,
    idx_n: torch.Tensor,
    label_of: List[str],
    relations: dict,
    M_STRONG: float = 3.66,
    M_DEFAULT: float = 1.66,
    M_WEAK: float = 0.10,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    mlist = []
    for ai, ni in zip(idx_a.tolist(), idx_n.tolist()):
        la, ln = label_of[ai], label_of[ni]
        m = _margin_between_labels(la, ln, relations, M_STRONG, M_DEFAULT, M_WEAK)
        mlist.append(m)
    return torch.tensor(mlist, dtype=torch.float32, device=device)


def triplet_loss(
    Z: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    n: torch.Tensor,
    label_of: List[str],
    relations: dict,
    margins: Tuple[float, float, float] = (3.66, 1.66, 0.10),
) -> torch.Tensor:
    d_ap = (Z[a] - Z[p]).norm(dim=1)
    d_an = (Z[a] - Z[n]).norm(dim=1)
    M_STRONG, M_DEFAULT, M_WEAK = margins
    mvec = margin_vector(a, n, label_of, relations, M_STRONG, M_DEFAULT, M_WEAK, device=Z.device)
    return torch.relu(d_ap - d_an + mvec).mean()


def reconstruction_loss(X_torch: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    Z = X_torch @ W.T
    X_recon = Z @ W
    return F.mse_loss(X_recon, X_torch)


# ----------------------------
# 5) 训练主入口
# ----------------------------
def train_triplet_model_from_tree(
    adata,
    tree: Tree,
    pca_key: str = "X_pca",
    use_leaves_only: bool = True,
    aggregator: str = "mean",
    min_valid_genes: int = 1,
    # 采样与负样本策略
    n_neighbors: int = 16,
    neg_strategy: str = "relation_based",  # or 'all_other_labels'
    # 训练超参
    batch_size: int = 4096,
    epochs: int = 400,
    lr: float = 6e-3,
    lam_recon: float = 1e-4,
    margins: Tuple[float, float, float] = (3.66, 1.66, 0.10),
    # 其他
    out_dir: str = "bifur_results",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    log_every: int = 20,
):
    if pca_key not in adata.obsm_keys():
        raise KeyError(f"adata.obsm['{pca_key}'] 不存在，请先准备 PCA 或更换 pca_key。")
    X_np = adata.obsm[pca_key].astype("float32")
    n_samples, d = X_np.shape

    # 1) gene tree -> labels
    labels, meta_df, nodes_considered = assign_labels_from_tree(
        adata, tree, use_leaves_only=use_leaves_only,
        aggregator=aggregator, min_valid_genes=min_valid_genes
    )

    # 2) 基于 tree 的关系映射
    relations = build_tree_relations(tree, nodes_considered)

    # 3) 正负样本池
    pos_pool, neg_pool = build_pos_neg_pools(
        X_np, labels, relations, n_neighbors=n_neighbors, neg_strategy=neg_strategy
    )

    # 4) 训练
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    X_torch = torch.as_tensor(X_np, dtype=torch.float32, device=dev)
    W = torch.eye(d, d, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([W], lr=lr)

    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    label_of = list(labels)

    for ep in range(epochs + 1):
        a, p, n = sample_triplets(n_samples, pos_pool, neg_pool, batch_size, rng)
        a = a.to(dev); p = p.to(dev); n = n.to(dev)
        Z = X_torch @ W.T

        loss_trip = triplet_loss(Z, a, p, n, label_of, relations, margins=margins)
        loss_recon = reconstruction_loss(X_torch, W)
        loss = loss_trip + lam_recon * loss_recon

        opt.zero_grad(); loss.backward(); opt.step()

        if (ep % log_every) == 0:
            print(f"Ep {ep:4d}/{epochs} | triplet={loss_trip.item():.4f} "
                  f"| recon={loss_recon.item():.4f} | total={loss.item():.4f}")

    return dict(W=W.detach().cpu(), labels=labels, meta_df=meta_df, relations=relations)


def project(X: np.ndarray, W: torch.Tensor) -> np.ndarray:
    Wt = W.float().numpy()
    return X.astype("float32") @ Wt.T
