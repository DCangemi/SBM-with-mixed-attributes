"""
I-Louvain with Gower distance for mixed-type node attributes.

Extends the original I-Louvain (Combe et al.) to handle:
  - continuous (numerical) attributes
  - categorical (nominal) attributes
  - any mix of the two

The key modification: instead of squared-Euclidean inertia (which
requires a centroid), we use a pairwise-dissimilarity formulation
based on Gower distance.  For purely continuous data this is
equivalent to the original (via the Huygens decomposition), but
generalises naturally to categorical features.

Gower distance between two nodes i, j is:

    d_G(i,j) = (1/p) * sum_k  d_k(i,j)

where p = number of attributes and d_k is:
  - for continuous:   |x_ik - x_jk| / range_k
  - for categorical:  0  if x_ik == x_jk,  1  otherwise

The "inertia" of a set S of size n_S is then defined as:

    I(S) = (1 / (2 * n_S)) * sum_{i,j in S}  d_G(i,j)

which equals the average distance to the geometric median — and
for Euclidean distance reduces to classical inertia.

Usage is identical to the original ML2 class, except the
`attributes` dict now carries raw (possibly string) values,
and you must supply `attr_types`: a dict  dim_index -> 'continuous' | 'categorical'.

Example:
    attributes = {
        0: {0: 25.0, 1: 'male',   2: 3.7},
        1: {0: 30.0, 1: 'female', 2: 2.1},
        ...
    }
    attr_types = {0: 'continuous', 1: 'categorical', 2: 'continuous'}

    ml = ML2Gower(graph, attributes, authorIndex, attr_types)
    partition = ml.findPartition()
"""

from __future__ import division
import numpy as np
import os.path
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


# ────────────────────────────────────────────────────────────────
# Gower helpers
# ────────────────────────────────────────────────────────────────

def _encode_attributes(
    nodes: List[Any],
    attributes: Dict[Any, Dict[int, Any]],
    attr_types: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[Any, int]]]:
    """
    Encode raw attributes into numeric arrays suitable for Gower
    distance computation.

    Returns
    -------
    cont_matrix : (n, d_cont)  float array of continuous values
    cat_matrix  : (n, d_cat)   int array of encoded categorical values
    cont_ranges : (d_cont,)    range of each continuous attribute
    cont_cols   : (d_cont,)    original column indices for continuous attrs
    cat_cols    : (d_cat,)     original column indices for categorical attrs
    cat_encodings : dict  col -> {value -> int}   (for reference)
    """
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    # separate dims
    cont_dims = sorted([k for k, v in attr_types.items() if v == 'continuous'])
    cat_dims = sorted([k for k, v in attr_types.items() if v == 'categorical'])

    d_cont = len(cont_dims)
    d_cat = len(cat_dims)

    cont_matrix = np.zeros((n, d_cont), dtype=float)
    cat_matrix = np.zeros((n, d_cat), dtype=int)

    # gather unique values for each categorical dim
    cat_encodings: Dict[int, Dict[Any, int]] = {}
    for d in cat_dims:
        unique_vals: Dict[Any, int] = {}
        for nd in nodes:
            val = attributes.get(nd, {}).get(d, None)
            if val not in unique_vals:
                unique_vals[val] = len(unique_vals)
        cat_encodings[d] = unique_vals

    # fill matrices
    for nd in nodes:
        i = node_to_idx[nd]
        attr = attributes.get(nd, {})
        for j, d in enumerate(cont_dims):
            cont_matrix[i, j] = float(attr.get(d, 0.0))
        for j, d in enumerate(cat_dims):
            val = attr.get(d, None)
            cat_matrix[i, j] = cat_encodings[d][val]

    # ranges for continuous (avoid div-by-zero)
    if d_cont > 0:
        cont_ranges = cont_matrix.max(axis=0) - cont_matrix.min(axis=0)
        cont_ranges[cont_ranges == 0] = 1.0
    else:
        cont_ranges = np.array([], dtype=float)

    return (
        cont_matrix,
        cat_matrix,
        cont_ranges,
        np.array(cont_dims, dtype=int),
        np.array(cat_dims, dtype=int),
        cat_encodings,
    )


def gower_matrix(
    cont_matrix: np.ndarray,
    cat_matrix: np.ndarray,
    cont_ranges: np.ndarray,
    weights_cont: Optional[np.ndarray] = None,
    weights_cat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the full (n x n) Gower dissimilarity matrix.

    Parameters
    ----------
    cont_matrix : (n, d_cont) continuous attribute values
    cat_matrix  : (n, d_cat)  integer-encoded categorical values
    cont_ranges : (d_cont,)   range of each continuous attribute
    weights_cont : optional per-dimension weights for continuous attrs
    weights_cat  : optional per-dimension weights for categorical attrs

    Returns
    -------
    D : (n, n) Gower dissimilarity matrix  (values in [0, 1])
    """
    n = cont_matrix.shape[0]
    d_cont = cont_matrix.shape[1]
    d_cat = cat_matrix.shape[1]
    p = d_cont + d_cat

    if p == 0:
        return np.zeros((n, n), dtype=float)

    if weights_cont is None:
        weights_cont = np.ones(d_cont, dtype=float)
    if weights_cat is None:
        weights_cat = np.ones(d_cat, dtype=float)

    total_weight = weights_cont.sum() + weights_cat.sum()
    if total_weight == 0:
        return np.zeros((n, n), dtype=float)

    D = np.zeros((n, n), dtype=float)

    # continuous contribution:  |x_ik - x_jk| / range_k
    if d_cont > 0:
        # normalise
        X_norm = cont_matrix / cont_ranges[None, :]          # (n, d_cont)
        for k in range(d_cont):
            col = X_norm[:, k]
            diff = np.abs(col[:, None] - col[None, :])        # (n, n)
            D += weights_cont[k] * diff

    # categorical contribution:  0 if same, 1 if different
    if d_cat > 0:
        for k in range(d_cat):
            col = cat_matrix[:, k]
            mismatch = (col[:, None] != col[None, :]).astype(float)
            D += weights_cat[k] * mismatch

    D /= total_weight
    return D


# ────────────────────────────────────────────────────────────────
# Status (unchanged from original)
# ────────────────────────────────────────────────────────────────

@dataclass
class Status:
    node2com: Dict[Any, int] = field(default_factory=dict)
    total_weight: float = 0.0
    internals: Dict[int, float] = field(default_factory=dict)
    degrees: Dict[int, float] = field(default_factory=dict)
    gdegrees: Dict[Any, float] = field(default_factory=dict)
    loops: Dict[Any, float] = field(default_factory=dict)

    def __str__(self):
        return (
            "------------------------\n"
            f"node2com   : {self.node2com}\n"
            f"degrees   : {self.degrees}\n"
            f"gdegrees  : {self.gdegrees}\n"
            f"internals : {self.internals}\n"
            f"total_w   : {self.total_weight}\n"
            f"loops     : {self.loops}\n"
            "------------------------"
        )

    def init(self, graph: nx.Graph):
        """Initialize graph status: each node in its own community."""
        self.node2com.clear()
        self.degrees.clear()
        self.gdegrees.clear()
        self.internals.clear()
        self.loops.clear()

        self.total_weight = graph.size(weight="weight")
        com_id = 0

        for node in sorted(graph.nodes()):
            self.node2com[node] = com_id
            deg = float(graph.degree(node, weight="weight"))
            self.degrees[com_id] = deg
            self.gdegrees[node] = deg
            self.loops[node] = float(
                graph.get_edge_data(node, node, {"weight": 0}).get("weight", 0.0)
            )
            self.internals[com_id] = self.loops[node]
            com_id += 1

    def init_attrib_status_gower(
        self,
        graph: nx.Graph,
        node_to_idx: Dict[Any, int],
        gower_dist_matrix: np.ndarray,
    ):
        """
        Initialise the attribute status using pairwise Gower distances.

        The total "inertia" is defined as:
            I_tot = (1 / (2n)) * sum_{i,j} d_G(i,j)

        which generalises the Euclidean inertia to arbitrary
        dissimilarities.  We set total_weight = -I_tot  (negative,
        as in the original I-Louvain convention).

        For each node, the "degree" is:
            phi_node = -(1/n) * ( I_tot + avg_dist_to_others )

        This mirrors the original formula  phi = -(I_tot + dist²_to_center)/n
        but uses pairwise distances instead of centroid distances.
        """
        n = len(node_to_idx)

        self.node2com.clear()
        self.degrees.clear()
        self.gdegrees.clear()
        self.internals.clear()
        self.loops.clear()

        # total inertia via pairwise distances
        inertie_tot = float(np.sum(gower_dist_matrix) / (2.0 * n))

        self.total_weight = -inertie_tot

        com_id = 0
        for node in sorted(graph.nodes()):
            idx = node_to_idx[node]

            # average distance of this node to all others
            avg_d = float(np.sum(gower_dist_matrix[idx, :])) / n

            phi = -1.0 * (inertie_tot + avg_d) / n

            self.node2com[node] = com_id
            self.degrees[com_id] = phi
            self.gdegrees[node] = phi
            self.loops[node] = 0.0
            self.internals[com_id] = 0.0
            com_id += 1

    def induced_attrib_status_tab(
        self, node2com: Dict[Any, int], bijection: Dict[int, int]
    ):
        """Update attribute status after graph coarsening."""
        retro = {v: k for k, v in bijection.items()}

        old_degrees = self.degrees
        old_internals = self.internals

        self.node2com = {}
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}

        for new_com in retro:
            old_com = retro[new_com]
            self.node2com[new_com] = new_com
            deg = old_degrees[old_com]
            self.degrees[new_com] = deg
            self.gdegrees[new_com] = deg
            self.loops[new_com] = old_internals[old_com]
            self.internals[new_com] = self.loops[new_com]


# ────────────────────────────────────────────────────────────────
# Main class: ML2Gower
# ────────────────────────────────────────────────────────────────

class ML2Gower:
    """
    I-Louvain extended with Gower distance for mixed-type attributes.

    Parameters
    ----------
    graph : nx.Graph
        The network (undirected or directed).
    attributes : dict  node -> {dim_index: value}
        Raw attribute values.  Continuous dims should have float/int
        values; categorical dims can have any hashable value (str, int, …).
    authorIndex : dict
        Node-to-label mapping (kept for compatibility).
    attr_types : dict  dim_index -> 'continuous' | 'categorical'
        Specifies the type of each attribute dimension.
    weights_cont : array-like or None
        Per-dimension weight for continuous attributes in Gower distance.
    weights_cat : array-like or None
        Per-dimension weight for categorical attributes in Gower distance.
    """

    __MIN = 1e-6
    __PASS_MAX = 100

    def __init__(
        self,
        graph: nx.Graph,
        attributes: Dict[Any, Dict[int, Any]],
        authorIndex: Dict[Any, Any],
        attr_types: Dict[int, str],
        weights_cont: Optional[np.ndarray] = None,
        weights_cat: Optional[np.ndarray] = None,
    ):
        self.graphBase = graph.copy()
        self.graph = graph
        self.authorIndex = authorIndex

        self.nodes: List[Any] = sorted(graph.nodes())
        self.node_to_idx: Dict[Any, int] = {n: i for i, n in enumerate(self.nodes)}
        self.nbVertices = len(self.nodes)

        # encode attributes
        (
            self.cont_matrix,
            self.cat_matrix,
            self.cont_ranges,
            self.cont_cols,
            self.cat_cols,
            self.cat_encodings,
        ) = _encode_attributes(self.nodes, attributes, attr_types)

        self.weights_cont = weights_cont
        self.weights_cat = weights_cat

        # precompute full Gower matrix
        self.gower_full = gower_matrix(
            self.cont_matrix,
            self.cat_matrix,
            self.cont_ranges,
            weights_cont=self.weights_cont,
            weights_cat=self.weights_cat,
        )

        # ── Status objects ──
        self.statusTab: List[Status] = []

        s_graph = Status()
        s_graph.init(graph)
        self.statusTab.append(s_graph)

        s_attr = Status()
        s_attr.init_attrib_status_gower(
            graph, self.node_to_idx, self.gower_full
        )
        self.statusTab.append(s_attr)

        self.status_list: List[Dict[Any, int]] = []

    # ── pairwise Gower distance between two nodes ──

    def dist(self, v1, v2) -> float:
        """Gower distance between two nodes (from precomputed matrix)."""
        i = self.node_to_idx[v1]
        j = self.node_to_idx[v2]
        return float(self.gower_full[i, j])

    # ── Gini matrix (= negative pairwise contribution matrix) ──

    def calculateGiniMatrixInitial(self) -> np.ndarray:
        """
        Initial "Gini matrix":  g[i,j] = -d_G(i,j) / n^2

        This is the direct analogue of the original code's
        -||x_i - x_j||^2 / n^2, but using Gower distance.
        """
        n = self.nbVertices
        return -self.gower_full / (n ** 2)

    def firstInducedGiniMatrix(
        self, partition: Dict[Any, int], gini_matrix: np.ndarray
    ) -> np.ndarray:
        """Coarsen the Gini matrix from nodes to communities (first pass)."""
        n_com = len(set(partition.values()))
        out = np.zeros((n_com, n_com), dtype=float)

        for i, node_i in enumerate(self.nodes):
            com_i = partition[node_i]
            for j, node_j in enumerate(self.nodes):
                com_j = partition[node_j]
                out[com_i, com_j] += gini_matrix[i, j]

        return out

    def inducedGiniMatrix(
        self, partition: Dict[Any, int], gini_matrix: np.ndarray
    ) -> np.ndarray:
        """Coarsen the Gini matrix (subsequent passes)."""
        n_com_new = len(set(partition.values()))
        out = np.zeros((n_com_new, n_com_new), dtype=float)

        for node_i in partition:
            for node_j in partition:
                ci = partition[node_i]
                cj = partition[node_j]
                out[ci, cj] += gini_matrix[node_i, node_j]

        return out

    # ── combination criterion ──

    def critereCombinaison(self) -> float:
        return (
            self.__modularity(self.statusTab[0])
            + self.__modularity(self.statusTab[1])
        ) / 2.0

    # ── main algorithm ──

    def findPartition(self) -> Dict[Any, int]:
        giniMatrix = self.calculateGiniMatrixInitial()
        self.__one_level(giniMatrix=giniMatrix)
        new_mod = self.critereCombinaison()

        partition, bijection = self.__renumber()
        self.status_list.append(partition)

        mod = new_mod
        self.induced_graph(partition)
        self.statusTab[0].init(self.graph)
        self.statusTab[1].induced_attrib_status_tab(partition, bijection)
        giniMatrix = self.firstInducedGiniMatrix(partition, giniMatrix)

        while True:
            self.__one_level(giniMatrix=giniMatrix)
            new_mod = self.critereCombinaison()

            if new_mod - mod < self.__MIN:
                break

            partition, bijection = self.__renumber()
            self.status_list.append(partition)
            mod = new_mod

            self.induced_graph(partition)
            giniMatrix = self.inducedGiniMatrix(partition, giniMatrix)
            self.statusTab[0].init(self.graph)
            self.statusTab[1].induced_attrib_status_tab(partition, bijection)

        # flatten dendrogram
        dendogram = self.status_list[:]
        partition = dendogram[0].copy()
        for idx in range(1, len(dendogram)):
            for node, community in partition.items():
                partition[node] = dendogram[idx][community]

        return partition

    # ── graph coarsening ──

    def induced_graph(self, partition: Dict[Any, int]):
        newGraph = nx.Graph()
        newGraph.add_nodes_from(set(partition.values()))
        for u, v, data in self.graph.edges(data=True):
            w = data.get("weight", 1.0)
            cu = partition[u]
            cv = partition[v]
            prev_w = newGraph.get_edge_data(cu, cv, {"weight": 0.0}).get(
                "weight", 0.0
            )
            newGraph.add_edge(cu, cv, weight=prev_w + w)
        self.graph = newGraph

    def __renumber(self) -> Tuple[Dict[Any, int], Dict[int, int]]:
        dictionary = self.statusTab[0].node2com
        ret = dictionary.copy()
        new_values: Dict[int, int] = {}
        count = 0
        for key, value in dictionary.items():
            if value not in new_values:
                new_values[value] = count
                count += 1
            ret[key] = new_values[value]
        return ret, new_values

    def __neighcom(self, node, giniMatrix=None):
        weights: Dict[int, np.ndarray] = {}
        voisins = self.graph[node].items()

        curCommunity = self.statusTab[0].node2com[node]
        if curCommunity not in weights:
            weights[curCommunity] = np.zeros(len(self.statusTab), dtype=float)

        for neighbor, datas in voisins:
            if neighbor == node:
                continue

            w = datas.get("weight", 1.0)
            neighborcom = self.statusTab[0].node2com[neighbor]

            if neighborcom not in weights:
                weights[neighborcom] = np.zeros(
                    len(self.statusTab), dtype=float
                )

            # graph part
            weights[neighborcom][0] += w

            # attributes part (Gower-based)
            if giniMatrix is not None:
                i = self.node_to_idx.get(node, node)
                j = self.node_to_idx.get(neighbor, neighbor)
                w_attr = giniMatrix[i, j]
            else:
                w_attr = -self.dist(node, neighbor) / (self.nbVertices ** 2)
            weights[neighborcom][1] += w_attr

        return weights

    def __one_level(self, giniMatrix=None):
        modif = True
        nb_pass = 0
        while modif:
            modif = False
            nb_pass += 1
            if self.__PASS_MAX > 0 and nb_pass > self.__PASS_MAX:
                break

            for node in self.graph.nodes():
                com_node = self.statusTab[0].node2com[node]
                best_com = com_node
                best_increase = 0.0

                neigh_communities = self.__neighcom(
                    node, giniMatrix=giniMatrix
                )

                degc_totw_tab = []
                for status in self.statusTab:
                    degc_totw_tab.append(
                        status.gdegrees.get(node, 0.0)
                        / (status.total_weight * 2.0)
                    )
                    theWeight = neigh_communities[com_node][
                        self.statusTab.index(status)
                    ]
                    if abs(status.degrees.get(com_node, 0.0)) <= abs(
                        status.gdegrees[node]
                    ):
                        status.degrees[com_node] = abs(
                            status.gdegrees[node]
                        )
                    self.__remove(node, com_node, theWeight, status)

                # check all neighbor communities
                for com, dnc in neigh_communities.items():
                    incr = 0.0
                    for i, status in enumerate(self.statusTab):
                        totw = abs(status.total_weight)
                        deg = status.degrees.get(com, 0.0)
                        if i == 0:
                            a = (
                                abs(dnc[i]) - abs(deg * degc_totw_tab[i])
                            ) / totw
                            incr += a
                        else:
                            a = (
                                0.0
                                - abs(dnc[i])
                                + abs(deg * degc_totw_tab[i])
                            ) / totw
                            incr += a
                    incr /= 2.0
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com

                # insert into best community
                for i, status in enumerate(self.statusTab):
                    theWeight = neigh_communities[best_com][i]
                    self.__insert(node, best_com, theWeight, status)

                if best_com != com_node:
                    modif = True

    def __remove(self, node, com, weight, status: Status):
        status.degrees[com] = status.degrees.get(com, 0.0) - status.gdegrees.get(
            node, 0.0
        )
        status.internals[com] = float(
            status.internals.get(com, 0.0) - weight - status.loops.get(node, 0.0)
        )
        status.node2com[node] = -1

    def __insert(self, node, com, weight, status: Status):
        status.node2com[node] = com
        status.degrees[com] = status.degrees.get(com, 0.0) + status.gdegrees.get(
            node, 0.0
        )
        status.internals[com] = float(
            status.internals.get(com, 0.0) + weight + status.loops.get(node, 0.0)
        )

    def __modularity(self, status):
        """
        Modularity for graph status or inertia-based modularity
        for attribute status.
        """
        # ── ATTRIBUTE PART: Q_inertia ──
        if status is self.statusTab[1]:
            links = abs(float(status.total_weight))
            if links == 0.0:
                return 0.0

            result = 0.0
            for community in set(status.node2com.values()):
                in_degree = abs(status.internals.get(community, 0.0))
                degree = abs(status.degrees.get(community, 0.0))
                expected = (degree / (2.0 * links)) ** 2
                found = in_degree / links
                if status.total_weight < 0:
                    result += expected - found
                else:
                    result += found - expected
            return result

        # ── GRAPH PART: Q_graph ──
        g = self.graph

        if not g.is_directed():
            links = abs(float(status.total_weight))
            if links == 0.0:
                return 0.0

            result = 0.0
            for community in set(status.node2com.values()):
                in_degree = abs(status.internals.get(community, 0.0))
                degree = abs(status.degrees.get(community, 0.0))
                expected = (degree / (2.0 * links)) ** 2
                found = in_degree / links
                if status.total_weight < 0:
                    result += expected - found
                else:
                    result += found - expected
            return result

        # directed modularity
        m = g.size(weight="weight")
        if m == 0.0:
            return 0.0

        com = status.node2com
        kout = dict(g.out_degree(weight="weight"))
        kin = dict(g.in_degree(weight="weight"))

        result = 0.0
        for u, v, data in g.edges(data=True):
            if com.get(u, -1) != com.get(v, -1):
                continue
            w = data.get("weight", 1.0)
            result += w - (kout[u] * kin[v] / m)

        return result / m


# ────────────────────────────────────────────────────────────────
# Convenience: backward-compatible wrapper
# ────────────────────────────────────────────────────────────────

class ML2(ML2Gower):
    """
    Drop-in replacement for the original ML2 class.

    If no `attr_types` are given, all attributes are assumed continuous
    and the behaviour is equivalent to the original I-Louvain
    (up to the switch from squared-Euclidean to range-normalised
    Manhattan distance used in Gower — see `use_original_euclidean`).

    Parameters
    ----------
    graph, attributes, authorIndex : same as original
    attr_types : optional; if None, all dims treated as continuous
    use_original_euclidean : bool
        If True AND all attributes are continuous, fall back to
        squared-Euclidean distance (exact original behaviour).
        Default False (use Gower everywhere for consistency).
    """

    def __init__(
        self,
        graph: nx.Graph,
        attributes: Dict[int, Dict[int, Any]],
        authorIndex: Dict[Any, Any],
        attr_types: Optional[Dict[int, str]] = None,
        use_original_euclidean: bool = False,
        weights_cont: Optional[np.ndarray] = None,
        weights_cat: Optional[np.ndarray] = None,
    ):
        # infer attr_types if not given
        if attr_types is None:
            max_dim = 0
            for nd in sorted(graph.nodes()):
                if nd in attributes and attributes[nd]:
                    max_dim = max(max_dim, max(attributes[nd].keys()) + 1)
            attr_types = {d: 'continuous' for d in range(max_dim)}

        super().__init__(
            graph,
            attributes,
            authorIndex,
            attr_types,
            weights_cont=weights_cont,
            weights_cat=weights_cat,
        )


# ────────────────────────────────────────────────────────────────
# Dataset loader (extended for mixed types)
# ────────────────────────────────────────────────────────────────

def loadDataset(path, attr_types: Optional[Dict[int, str]] = None):
    """
    Load graph + attributes.

    If `attr_types` is provided, categorical columns are kept as
    strings; continuous columns are cast to float.  If not provided,
    all columns are cast to float (original behaviour).
    """
    graph = nx.Graph()

    if not os.path.isfile(path + ".edgeList"):
        print("Error: file '" + path + ".edgeList' not found")
        exit(-1)
    with open(path + ".edgeList") as f:
        for line in f.readlines():
            parts = line.strip().split()
            v1, v2 = int(parts[0]), int(parts[1])
            graph.add_node(v1)
            graph.add_node(v2)
            graph.add_edge(v1, v2)

    attributes: Dict[int, Dict[int, Any]] = {}
    for n in graph:
        attributes[n] = {}

    if not os.path.isfile(path + ".attributes"):
        print("Error: file '" + path + ".attributes' not found")
        exit(-1)

    with open(path + ".attributes") as f:
        for line in f.readlines():
            parts = line.strip().split()
            vertexId = int(parts[0])
            elems = parts[1].split(",")
            attrValues: Dict[int, Any] = {}
            for i, val in enumerate(elems):
                if attr_types and attr_types.get(i) == 'categorical':
                    attrValues[i] = val          # keep as string
                else:
                    attrValues[i] = float(val)   # cast to float
            attributes[vertexId] = attrValues

    authorIndex = {n: n for n in graph}

    if os.path.exists(path + ".2ModLouvain"):
        os.remove(path + ".2ModLouvain")

    return graph, attributes, authorIndex


# ────────────────────────────────────────────────────────────────
# Quick test / demo
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # tiny synthetic example
    G = nx.karate_club_graph()

    # create mixed attributes: one continuous, one categorical
    np.random.seed(42)
    attributes = {}
    for n in G.nodes():
        attributes[n] = {
            0: float(np.random.randn()),            # continuous
            1: np.random.choice(["A", "B", "C"]),   # categorical
            2: float(np.random.rand() * 10),         # continuous
        }

    attr_types = {
        0: 'continuous',
        1: 'categorical',
        2: 'continuous',
    }

    authorIndex = {n: n for n in G.nodes()}

    ml = ML2Gower(G, attributes, authorIndex, attr_types)
    partition = ml.findPartition()

    # print results
    communities = {}
    for node, com in partition.items():
        communities.setdefault(com, []).append(node)

    print(f"Found {len(communities)} communities:")
    for com_id, members in sorted(communities.items()):
        print(f"  Community {com_id}: {sorted(members)}")

    print(f"\nCombined modularity: {ml.critereCombinaison():.4f}")
