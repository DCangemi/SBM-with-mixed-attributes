"""
SToC: Semantic-Topological Clustering for Attributed Graphs

Re-implementation of:
    A. Baroni, A. Conte, M. Patrignani, S. Ruggieri.
    "Efficiently Clustering Very Large Attributed Graphs."
    ASONAM 2017  /  arXiv:1703.08590

The algorithm:
  1. Compute a *semantic distance* dS between nodes using Euclidean
     distance for quantitative attributes (normalised to [0,1]) and
     Jaccard distance for categorical attributes.
  2. Compute a *topological distance* dT between nodes as the Jaccard
     distance of their l-neighbourhoods, approximated via bottom-k
     sketches for scalability.
  3. Combine them as  dST(u,v) = max(dS(u,v), dT(u,v)).
  4. For a threshold τ, iteratively pick random seeds and grow
     τ-close connected clusters via BFS (STo-Query).
  5. τ and l are auto-tuned from user-friendly "attraction ratio"
     parameters αS and αT.

Usage with the Simulazioni.ipynb data:
    from stoc import stoc
    result = stoc(G, Xg=Xg, Xc=Xc, alpha_s=0.3, alpha_t=0.3)
    labels = result['labels']
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import hashlib


# ────────────────────────────────────────────────────────────────
# Semantic distance
# ────────────────────────────────────────────────────────────────

def _build_semantic_vectors(
    n: int,
    Xg: Optional[pd.DataFrame],
    Xc: Optional[pd.DataFrame],
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Prepare normalised continuous matrix and categorical DataFrame.
    Continuous attributes are min-max normalised to [0, 1].
    """
    cont = None
    cat = None

    if Xg is not None and Xg.shape[1] > 0:
        arr = Xg.values[:n].astype(float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        cont = (arr - mins) / ranges  # (n, Q)

    if Xc is not None and Xc.shape[1] > 0:
        cat = Xc.iloc[:n].astype(str).reset_index(drop=True)

    return cont, cat


def _semantic_distance(
    cont: Optional[np.ndarray],
    cat: Optional[pd.DataFrame],
    i: int,
    j: int,
) -> float:
    """
    Eq. 2 from the paper:
        dS = ( sqrt(sum_quant (fi-fj)^2) * sqrt(Q)
              + sum_cat Jaccard(fi, fj) )  /  A

    For scalar categorical attributes, Jaccard(x, y) = 0 if x==y, 1 otherwise.
    """
    total = 0.0
    A = 0  # total number of attributes

    # quantitative part: Euclidean on normalised values, scaled by sqrt(Q)
    if cont is not None:
        Q = cont.shape[1]
        diff = cont[i] - cont[j]
        euc = np.sqrt(np.sum(diff ** 2))
        # scale so each quant attr contributes in [0,1] range
        total += euc / np.sqrt(Q) if Q > 0 else 0.0
        A += 1  # the quantitative block counts as 1 component

    # categorical part: fraction of mismatches (= average Jaccard for scalars)
    if cat is not None:
        n_cat = cat.shape[1]
        mismatches = sum(
            1.0 for c in cat.columns if cat.iloc[i][c] != cat.iloc[j][c]
        )
        total += mismatches / n_cat if n_cat > 0 else 0.0
        A += 1  # the categorical block counts as 1 component

    if A == 0:
        return 0.0

    return total / A


def _semantic_distance_matrix(
    cont: Optional[np.ndarray],
    cat: Optional[pd.DataFrame],
    n: int,
) -> np.ndarray:
    """Compute full n×n semantic distance matrix."""
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _semantic_distance(cont, cat, i, j)
            D[i, j] = d
            D[j, i] = d
    return D


# ────────────────────────────────────────────────────────────────
# Topological distance via bottom-k sketches
# ────────────────────────────────────────────────────────────────

def _compute_l_neighborhoods(
    adj: Dict[int, List[int]], n: int, l: int
) -> List[set]:
    """Compute the l-neighbourhood N_l(v) for each node v via BFS."""
    neighborhoods = []
    for v in range(n):
        visited = {v}
        frontier = {v}
        for _ in range(l):
            next_frontier = set()
            for u in frontier:
                for w in adj.get(u, []):
                    if w not in visited:
                        visited.add(w)
                        next_frontier.add(w)
            frontier = next_frontier
        neighborhoods.append(visited)
    return neighborhoods


def _build_sketches(
    neighborhoods: List[set], n: int, k: int, seed: int = 42
) -> List[np.ndarray]:
    """
    Build bottom-k sketches for each l-neighbourhood.
    We hash each node to a random value, then keep the k smallest.
    """
    rng = np.random.RandomState(seed)
    # assign a random hash to each node
    hash_vals = rng.random(n)

    sketches = []
    for neigh in neighborhoods:
        vals = sorted(hash_vals[list(neigh)])
        sketch = np.array(vals[:k]) if len(vals) >= k else np.array(vals)
        sketches.append(sketch)
    return sketches


def _jaccard_from_sketches(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Estimate Jaccard similarity from two bottom-k sketches.
    Merge and take the k smallest; count how many came from both.
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    k = max(len(s1), len(s2))
    merged = np.sort(np.union1d(s1, s2))[:k]
    s1_set = set(s1)
    s2_set = set(s2)
    intersection = sum(1 for v in merged if v in s1_set and v in s2_set)
    if len(merged) == 0:
        return 0.0
    sim = intersection / len(merged)
    return 1.0 - sim  # Jaccard distance


def _jaccard_exact(s1: set, s2: set) -> float:
    """Exact Jaccard distance between two sets."""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return 1.0 - inter / union if union > 0 else 0.0


def _topological_distance_matrix(
    adj: Dict[int, List[int]],
    n: int,
    l: int,
    use_sketches: bool = True,
    sketch_k: int = 20,
) -> np.ndarray:
    """Compute full n×n topological distance matrix."""
    neighborhoods = _compute_l_neighborhoods(adj, n, l)

    if use_sketches and n > 500:
        sketches = _build_sketches(neighborhoods, n, k=sketch_k)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = _jaccard_from_sketches(sketches[i], sketches[j])
                D[i, j] = d
                D[j, i] = d
    else:
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = _jaccard_exact(neighborhoods[i], neighborhoods[j])
                D[i, j] = d
                D[j, i] = d

    return D


# ────────────────────────────────────────────────────────────────
# Auto-tuning of τ and l
# ────────────────────────────────────────────────────────────────

def _auto_tune_tau(DS: np.ndarray, alpha_s: float) -> float:
    """
    Set τ as the αS-quantile of the semantic distance distribution.
    Since dST = max(dS, dT) and both must be ≤ τ, we use the
    semantic quantile as the threshold.
    """
    n = DS.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vals = DS[triu_idx]
    if len(vals) == 0:
        return 0.5
    tau = np.quantile(vals, alpha_s)
    return float(tau)


def _auto_tune_l(
    adj: Dict[int, List[int]],
    n: int,
    tau: float,
    alpha_t: float,
    max_l: int = 10,
) -> int:
    """
    Find l such that the fraction of node pairs with dT ≤ τ
    is closest to αT.

    For small graphs, l-neighborhoods grow quickly to cover the
    whole graph, so we pick the l that best matches the target
    attraction ratio.
    """
    best_l = 1
    best_diff = float("inf")

    n_samples = min(1000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    pairs = [(rng.randint(0, n), rng.randint(0, n)) for _ in range(n_samples)]
    pairs = [(i, j) for i, j in pairs if i != j]

    for l_try in range(1, max_l + 1):
        neighborhoods = _compute_l_neighborhoods(adj, n, l_try)

        count_close = 0
        for i, j in pairs:
            dt = _jaccard_exact(neighborhoods[i], neighborhoods[j])
            if dt <= tau:
                count_close += 1

        alpha_l = count_close / max(len(pairs), 1)
        diff = abs(alpha_l - alpha_t)

        if diff < best_diff:
            best_diff = diff
            best_l = l_try

        # if we've overshot (too many pairs close), and it's getting
        # worse, stop — but only after l ≥ 2 to avoid l=1 trap
        if l_try >= 2 and alpha_l > alpha_t and diff > best_diff:
            break

    return best_l


# ────────────────────────────────────────────────────────────────
# STo-Query: grow a τ-close cluster from a seed
# ────────────────────────────────────────────────────────────────

def _sto_query(
    seed: int,
    adj: Dict[int, List[int]],
    DS: np.ndarray,
    DT: np.ndarray,
    tau: float,
    active: set,
) -> set:
    """
    Algorithm 1 from the paper.
    BFS from seed, adding neighbours x if dST(seed, x) = max(dS, dT) ≤ τ
    and x is still active (not yet clustered).
    """
    cluster = {seed}
    queue = deque([seed])

    while queue:
        v = queue.popleft()
        for x in adj.get(v, []):
            if x not in active or x in cluster:
                continue
            d_st = max(DS[seed, x], DT[seed, x])
            if d_st <= tau:
                cluster.add(x)
                queue.append(x)

    return cluster


# ────────────────────────────────────────────────────────────────
# Main SToC algorithm
# ────────────────────────────────────────────────────────────────

def stoc(
    G: nx.Graph,
    Xg: Optional[pd.DataFrame] = None,
    Xc: Optional[pd.DataFrame] = None,
    alpha_s: float = 0.3,
    alpha_t: float = 0.3,
    tau: Optional[float] = None,
    l: Optional[int] = None,
    max_l: int = 10,
    verbose: bool = False,
) -> Dict:
    """
    SToC: Semantic-Topological Clustering.

    Parameters
    ----------
    G        : networkx Graph
    Xg       : (n, d_cont) DataFrame of continuous attributes (optional)
    Xc       : (n, d_cat) DataFrame of categorical attributes (optional)
    alpha_s  : semantic attraction ratio (0 to 1). Higher = larger clusters.
    alpha_t  : topological attraction ratio (0 to 1). Higher = larger clusters.
    tau      : distance threshold (if None, auto-tuned from alpha_s)
    l        : neighbourhood depth for topological distance (if None, auto-tuned from alpha_t)
    max_l    : maximum l to try during auto-tuning
    verbose  : print progress

    Returns
    -------
    dict with keys:
        'labels'    : np.ndarray (n,) cluster assignments
        'k'         : number of clusters found
        'tau'       : threshold used
        'l'         : neighbourhood depth used
        'clusters'  : list of sets of node indices
    """
    nodes = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)

    # build adjacency with integer indices
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i].append(j)
        adj[j].append(i)

    # ── semantic distance ──
    if verbose:
        print("Computing semantic distances ...")
    cont, cat = _build_semantic_vectors(n, Xg, Xc)
    DS = _semantic_distance_matrix(cont, cat, n)

    # ── auto-tune τ ──
    if tau is None:
        tau = _auto_tune_tau(DS, alpha_s)
    if verbose:
        print(f"  τ = {tau:.4f}")

    # ── auto-tune l ──
    if l is None:
        l = _auto_tune_l(adj, n, tau, alpha_t, max_l=max_l)
    if verbose:
        print(f"  l = {l}")

    # ── topological distance ──
    if verbose:
        print("Computing topological distances ...")
    DT = _topological_distance_matrix(adj, n, l, use_sketches=(n > 500))

    # ── SToC main loop (Algorithm 2) ──
    if verbose:
        print("Running SToC clustering ...")

    active = set(range(n))
    clusters: List[set] = []
    rng = np.random.RandomState(42)

    while active:
        # select random seed from active nodes
        seed = rng.choice(list(active))

        # grow τ-close cluster from seed
        cluster = _sto_query(seed, adj, DS, DT, tau, active)

        # remove cluster from active
        active -= cluster
        clusters.append(cluster)

    # ── build labels ──
    labels = np.zeros(n, dtype=int)
    for c_id, cluster in enumerate(clusters):
        for node in cluster:
            labels[node] = c_id

    if verbose:
        print(f"  Found {len(clusters)} clusters")

    return {
        "labels": labels,
        "k": len(clusters),
        "tau": tau,
        "l": l,
        "clusters": clusters,
    }


# ────────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # ── reproduce Scenario 1 ──
    random.seed(42)
    size_comm = [50, 50, 50]
    probs = [[0.6, 0.2, 0.05], [0.2, 0.5, 0.1], [0.05, 0.1, 0.5]]
    G = nx.stochastic_block_model(sizes=size_comm, p=probs, directed=False)

    np.random.seed(42)
    Xg = np.zeros((150, 2))
    Xg[:50, 0] = np.random.random_sample(50)
    Xg[50:100, 0] = np.random.random_sample(50) + 2
    Xg[100:, 0] = np.random.random_sample(50) - 2
    Xg[:50, 1] = np.random.random_sample(50)
    Xg[50:100, 1] = np.random.random_sample(50) + 2
    Xg[100:, 1] = np.random.random_sample(50) + 2
    Xg = pd.DataFrame(Xg)

    random.seed(42)
    Xc = pd.DataFrame(index=range(150), columns=["a1", "a2"])
    Xc.loc[:49, "a1"] = random.choices([1, 2, 3], k=50, weights=[0.8, 0.15, 0.05])
    Xc.loc[50:99, "a1"] = random.choices([1, 2, 3], k=50, weights=[0.15, 0.8, 0.05])
    Xc.loc[100:, "a1"] = random.choices([1, 2, 3], k=50, weights=[0.05, 0.15, 0.8])
    Xc.loc[:49, "a2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.7, 0.15, 0.05, 0.1])
    Xc.loc[50:99, "a2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.15, 0.7, 0.05, 0.1])
    Xc.loc[100:, "a2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.05, 0.15, 0.7, 0.1])

    labels_true = [n for n, s in enumerate(size_comm) for _ in range(s)]

    print("=== SToC parameter sweep ===")
    print(f"{'alpha':<12} {'k':>4}  {'ARI':>7}  {'NMI':>7}  {'tau':>6}  {'l':>2}")
    print("-" * 50)

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        result = stoc(G, Xg=Xg, Xc=Xc, alpha_s=alpha, alpha_t=alpha)
        ari = adjusted_rand_score(labels_true, result["labels"])
        nmi = normalized_mutual_info_score(labels_true, result["labels"])
        print(
            f"α={alpha:<10.1f} {result['k']:>4}  {ari:>7.4f}  {nmi:>7.4f}"
            f"  {result['tau']:>6.3f}  {result['l']:>2}"
        )
