"""
SA-Cluster: Graph Clustering Based on Structural/Attribute Similarities

Re-implementation of:
    Y. Zhou, H. Cheng, J. X. Yu.
    "Graph clustering based on structural/attribute similarities."
    VLDB 2009.

    Y. Zhou, H. Cheng, J. X. Yu.
    "Clustering large attributed graphs: An efficient incremental approach."
    ICDM 2010 / DMKD 2012.

The algorithm:
  1. Build an *attribute-augmented graph* by adding virtual "attribute
     nodes" — one per (attribute, value) pair — and connecting each
     real node to the attribute nodes corresponding to its values.
  2. Compute a *unified neighbourhood random-walk distance* matrix on
     the augmented graph.  This merges structural and attribute
     proximity into a single measure.
  3. Run K-Medoids clustering on the random-walk distance matrix.
  4. Iteratively re-weight each attribute's edges based on how well
     that attribute discriminates the current clusters (the "weight
     self-adjustment" step), then recompute distances and re-cluster.
  5. Repeat until the objective function converges.

This implementation accepts the same data format used in the
Simulazioni.ipynb notebook (networkx graph, Xg DataFrame for
continuous attributes, Xc DataFrame for categorical attributes).

Continuous attributes are discretised into `n_bins` equal-frequency
bins so that the augmented-graph formulation (which requires
discrete attribute values) can be applied.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _discretise_continuous(
    Xg: pd.DataFrame, n_bins: int = 10
) -> pd.DataFrame:
    """
    Discretise each continuous column into `n_bins` equal-frequency
    (quantile) bins.  Returns a DataFrame of string labels.
    """
    out = pd.DataFrame(index=Xg.index)
    for col in Xg.columns:
        out[col] = pd.qcut(
            Xg[col], q=n_bins, labels=False, duplicates="drop"
        ).astype(str)
    return out


def _build_attribute_table(
    n_nodes: int,
    Xg: Optional[pd.DataFrame],
    Xc: Optional[pd.DataFrame],
    n_bins: int,
) -> pd.DataFrame:
    """
    Merge continuous (after discretisation) and categorical
    attributes into a single DataFrame of string-valued columns.
    """
    parts = []
    if Xg is not None:
        disc = _discretise_continuous(Xg, n_bins=n_bins)
        disc.columns = [f"__cont_{c}" for c in disc.columns]
        parts.append(disc)
    if Xc is not None:
        cat = Xc.copy()
        cat.columns = [f"__cat_{c}" for c in cat.columns]
        cat = cat.astype(str)
        parts.append(cat)
    if not parts:
        raise ValueError("At least one of Xg, Xc must be provided")
    return pd.concat(parts, axis=1)


# ────────────────────────────────────────────────────────────────
# Core: augmented graph + random-walk distance
# ────────────────────────────────────────────────────────────────

def _build_augmented_transition(
    A: np.ndarray,
    attr_table: pd.DataFrame,
    omega: np.ndarray,
    omega0: float = 1.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Build the transition matrix P_A of the attribute-augmented graph.

    Parameters
    ----------
    A        : (N, N)  adjacency matrix of the original graph
    attr_table : (N, m) DataFrame of string attribute values
    omega    : (m,)  attribute edge weights
    omega0   : structure edge weight (fixed at 1.0)

    Returns
    -------
    PA       : (N+|Va|, N+|Va|) transition probability matrix
    N        : number of structure nodes
    n_attr   : total number of attribute nodes
    """
    N = A.shape[0]
    m = attr_table.shape[1]

    # Build attribute-value -> attribute-node-index mapping
    attr_node_offset = N
    attr_val_to_idx: Dict[Tuple[int, str], int] = {}
    idx = 0
    for j, col in enumerate(attr_table.columns):
        for val in sorted(attr_table[col].unique()):
            attr_val_to_idx[(j, val)] = attr_node_offset + idx
            idx += 1
    n_attr = idx
    total = N + n_attr

    # Build membership: for each attribute j, which attribute-node
    # does each structure-node connect to?
    # membership[j][i] = index of the attribute node that node i connects to
    membership: List[Dict[int, int]] = []
    for j, col in enumerate(attr_table.columns):
        d = {}
        for i in range(N):
            val = attr_table.iloc[i][col]
            d[i] = attr_val_to_idx[(j, val)]
        membership.append(d)

    # Count structural neighbours per node
    struct_deg = A.sum(axis=1)  # (N,)

    # Build transition matrix
    PA = np.zeros((total, total), dtype=float)

    for i in range(N):
        denom = struct_deg[i] * omega0 + sum(omega)
        if denom == 0:
            continue

        # structure-to-structure transitions  (Eq. 5)
        for j2 in range(N):
            if A[i, j2] > 0:
                PA[i, j2] = omega0 * A[i, j2] / denom

        # structure-to-attribute transitions  (Eq. 6)
        for j_attr in range(m):
            a_idx = membership[j_attr][i]
            PA[i, a_idx] = omega[j_attr] / denom

    # attribute-to-structure transitions  (Eq. 7)
    # Each attribute node connects back uniformly to all structure
    # nodes that share that value.
    attr_node_members: Dict[int, List[int]] = {}
    for j_attr in range(m):
        for i in range(N):
            a_idx = membership[j_attr][i]
            attr_node_members.setdefault(a_idx, []).append(i)

    for a_idx, members in attr_node_members.items():
        n_members = len(members)
        for i in members:
            PA[a_idx, i] = 1.0 / n_members

    # attribute-to-attribute = 0  (Eq. 8)  -- already zero

    return PA, N, n_attr


def _random_walk_distance(
    PA: np.ndarray, N: int, l: int = 20, c: float = 0.15
) -> np.ndarray:
    """
    Compute the unified neighbourhood random-walk distance matrix R
    on the top-left N×N block (structure nodes only).

    R^l = sum_{γ=1}^{l}  c * (1-c)^γ * P_A^γ     (Eq. 11)
    """
    total = PA.shape[0]
    R = np.zeros((total, total), dtype=float)
    Pg = np.eye(total, dtype=float)  # P^0 = I

    for gamma in range(1, l + 1):
        Pg = Pg @ PA                 # P^gamma
        coeff = c * ((1.0 - c) ** gamma)
        R += coeff * Pg

    # return only the N×N block for structure nodes
    return R[:N, :N]


# ────────────────────────────────────────────────────────────────
# K-Medoids
# ────────────────────────────────────────────────────────────────

def _init_centroids_density(
    R: np.ndarray, k: int, sigma: float
) -> np.ndarray:
    """
    Select k initial centroids with highest density values.
    density(v_i) = sum_j  (1 - exp(-R(i,j)^2 / (2*sigma^2)))
    (Eqs. 13-14)
    """
    N = R.shape[0]
    density = np.zeros(N)
    for i in range(N):
        density[i] = np.sum(1.0 - np.exp(-R[i, :] ** 2 / (2.0 * sigma ** 2)))

    # pick top-k, but ensure they are spread out
    order = np.argsort(-density)
    centroids = [order[0]]
    for idx in order[1:]:
        if len(centroids) >= k:
            break
        centroids.append(idx)
    return np.array(centroids[:k])


def _assign_clusters(R: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each node to its closest centroid (= largest random-walk
    distance, since RW distance = closeness).
    """
    # R[i, centroid] is "closeness"  -> argmax
    dists = R[:, centroids]       # (N, k)
    return np.argmax(dists, axis=1)


def _update_centroids(
    R: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """
    Update centroids: pick the node in each cluster whose RW-distance
    vector is closest to the cluster average.  (Eqs. 15-16)
    """
    N = R.shape[0]
    centroids = np.zeros(k, dtype=int)
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            centroids[c] = np.random.randint(N)
            continue
        avg_vec = R[members, :].mean(axis=0)               # (N,)
        diffs = np.linalg.norm(R[members, :] - avg_vec[None, :], axis=1)
        centroids[c] = members[np.argmin(diffs)]
    return centroids


# ────────────────────────────────────────────────────────────────
# Weight self-adjustment
# ────────────────────────────────────────────────────────────────

def _adjust_weights(
    attr_table: pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    omega: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Majority-vote weight update.  (Eqs. 23-25)

    For each attribute a_i, count how many cluster members share the
    centroid's value on a_i.  Attributes with higher agreement get
    higher weight.
    """
    m = attr_table.shape[1]
    votes = np.zeros(m, dtype=float)

    for j, col in enumerate(attr_table.columns):
        total_votes = 0.0
        for c_idx in range(k):
            members = np.where(labels == c_idx)[0]
            centroid_val = attr_table.iloc[centroids[c_idx]][col]
            total_votes += np.sum(
                attr_table.iloc[members][col].values == centroid_val
            )
        votes[j] = total_votes

    # normalise so that sum = m   (Eq. 24)
    vote_sum = votes.sum()
    if vote_sum == 0:
        delta = np.ones(m)
    else:
        delta = m * votes / vote_sum

    # smoothed update  (Eq. 22)
    omega_new = 0.5 * (omega + delta)
    return omega_new


# ────────────────────────────────────────────────────────────────
# Objective function
# ────────────────────────────────────────────────────────────────

def _objective(R: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    O = sum_c  mean pairwise RW-distance within cluster c.
    (Eqs. 17-18)
    """
    obj = 0.0
    for c in range(k):
        members = np.where(labels == c)[0]
        nc = len(members)
        if nc <= 1:
            continue
        sub = R[np.ix_(members, members)]
        obj += sub.sum() / (nc * nc)
    return obj


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────

def sa_cluster(
    G: nx.Graph,
    k: int,
    Xg: Optional[pd.DataFrame] = None,
    Xc: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
    l: int = 20,
    c: float = 0.15,
    sigma: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Dict:
    """
    SA-Cluster: attributed graph clustering via augmented-graph
    random walks with adaptive attribute weighting.

    Parameters
    ----------
    G       : networkx Graph
    k       : number of clusters
    Xg      : (n, d_cont) DataFrame of continuous attributes (optional)
    Xc      : (n, d_cat)  DataFrame of categorical attributes (optional)
    n_bins  : number of bins for discretising continuous attributes
    l       : random-walk length limit
    c       : restart probability
    sigma   : bandwidth for the density/influence function
    max_iter: maximum outer iterations
    tol     : convergence tolerance on the objective function
    verbose : print progress

    Returns
    -------
    dict with keys:
        'labels'    : np.ndarray (n,) cluster assignments
        'centroids' : np.ndarray (k,) centroid node indices
        'omega'     : np.ndarray (m,) final attribute weights
        'objective' : float, final objective value
        'history'   : list of objective values per iteration
    """
    # ── adjacency matrix ──
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # ── attribute table ──
    # Reindex Xg/Xc to match node order
    if Xg is not None:
        Xg = Xg.iloc[:N].reset_index(drop=True)
    if Xc is not None:
        Xc = Xc.iloc[:N].reset_index(drop=True)

    attr_table = _build_attribute_table(N, Xg, Xc, n_bins=n_bins)
    m = attr_table.shape[1]

    # ── initialise weights ──
    omega = np.ones(m, dtype=float)

    # ── first random-walk distance ──
    PA, _, _ = _build_augmented_transition(A, attr_table, omega)
    R = _random_walk_distance(PA, N, l=l, c=c)

    # ── initial centroids ──
    centroids = _init_centroids_density(R, k, sigma=sigma)

    # ── main loop ──
    history = []
    prev_obj = -np.inf

    for it in range(max_iter):
        # cluster assignment
        labels = _assign_clusters(R, centroids)

        # centroid update
        centroids = _update_centroids(R, labels, k)

        # re-assign after centroid update
        labels = _assign_clusters(R, centroids)

        # objective
        obj = _objective(R, labels, k)
        history.append(obj)

        if verbose:
            print(f"  iter {it+1:>3d}  obj={obj:.6f}  "
                  f"omega={np.round(omega, 3)}")

        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

        # weight self-adjustment
        omega = _adjust_weights(attr_table, labels, centroids, omega, k)

        # recompute random-walk distance with updated weights
        PA, _, _ = _build_augmented_transition(A, attr_table, omega)
        R = _random_walk_distance(PA, N, l=l, c=c)

    return {
        "labels": labels,
        "centroids": centroids,
        "omega": omega,
        "objective": obj,
        "history": history,
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
    Xc = pd.DataFrame(index=range(150), columns=["attr_cat1", "attr_cat2"])
    Xc.loc[:49, "attr_cat1"] = random.choices([1, 2, 3], k=50, weights=[0.8, 0.15, 0.05])
    Xc.loc[50:99, "attr_cat1"] = random.choices([1, 2, 3], k=50, weights=[0.15, 0.8, 0.05])
    Xc.loc[100:, "attr_cat1"] = random.choices([1, 2, 3], k=50, weights=[0.05, 0.15, 0.8])
    Xc.loc[:49, "attr_cat2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.7, 0.15, 0.05, 0.1])
    Xc.loc[50:99, "attr_cat2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.15, 0.7, 0.05, 0.1])
    Xc.loc[100:, "attr_cat2"] = random.choices([1, 2, 3, 4], k=50, weights=[0.05, 0.15, 0.7, 0.1])

    labels_true = []
    for n, i in enumerate(size_comm):
        labels_true += [n] * i

    print("Running SA-Cluster ...")
    result = sa_cluster(G, k=3, Xg=Xg, Xc=Xc, verbose=True)

    ari = adjusted_rand_score(labels_true, result["labels"])
    nmi = normalized_mutual_info_score(labels_true, result["labels"])
    print(f"\nSA-Cluster results:")
    print(f"  ARI : {ari:.4f}")
    print(f"  NMI : {nmi:.4f}")
    print(f"  Final omega: {result['omega']}")
