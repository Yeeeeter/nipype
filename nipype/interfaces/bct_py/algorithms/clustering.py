from __future__ import division, print_function
import numpy as np
from .modularity import modularity_louvain_und_sign
from bct.utils import cuberoot, BCTParamError, dummyvar, binarize, get_rng
from .distance import breadthdist, distance_wei_floyd, retrieve_shortest_path
from ..due import due, BibTeX
from ..citations import (
    WATTS1998, ONNELA2005, FAGIOLO2007, LANCICHINETTI2012,
    RUBINOV2010, HUMPHRIES2008, COSTANTINI2014, ZHANG2005
)

from bct.algorithms.clustering import *

from nipype import BaseInterface
import os

from ..base import NipyBaseInterface
from ...base import (
    TraitedSpec,
    traits,
    Tuple,
    File,
    OutputMultiPath,
    BaseInterfaceInputSpec,
    isdefined,
)


class AgreementInputSpec(BaseInterfaceInputSpec):
    degeneration_file = File(exists=True,
                       mandatory=True,
                       desc="Set of M different partitions of nodes")
    algo_type = traits.Enum(
        'normal',
        'weights',
        usedefault=True,
        mandatory=True
    )
    buffsz = traits.Int(
        default_value=1000,
        usedefault=True
    )
    relative_weights = File(
        exists=True,
        desc="relative weights for weighted agreement algorithm"
    )

class AgreementOutputSpec(TraitedSpec):
    agreement_mat_file = File(
        exists=True,
        desc="output agreement matrix file"
    )

class Agreement(BaseInterface):
    input_spec = AgreementInputSpec
    output_spec = AgreementOutputSpec

    def _run_interface(self, runtime):
        degeneration_file = self.inputs.degeneration_file
        algo_type = self.inputs.algo_type

        deg_mat = np.load(degeneration_file)

        if algo_type == 'normal':
            error_vals = [val for val in (None, 0,) if self.inputs.algo_type is val]
            if error_vals:
                raise ValueError(f"Agreement Algorithm (type normal) needs an approriate buffsz not {error_vals}")
            buffsz = self.inputs.buffsz

            agreement_mat = agreement(deg_mat, buffsz)

        if algo_type == 'weights':
            if self.inputs.weights is None:
                raise ValueError("Weights are not specified")

            weights = np.load(self.inputs.weights)
            agreement_mat = agreement_weighted(deg_mat, weights)

        np.save(os.path.abspath("agreement_mat"), agreement_mat)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["agreement_mat_file"] = os.path.abspath("agreement_mat")


@due.dcite(BibTeX(WATTS1998), description="Unweighted directed clustering coefficient")
@due.dcite(BibTeX(ONNELA2005), description="Unweighted directed clustering coefficient")
@due.dcite(BibTeX(FAGIOLO2007), description="Unweighted directed clustering coefficient")
@due.dcite(BibTeX(WATTS1998), description="Unweighted undirected clustering coefficient")
@due.dcite(BibTeX(ONNELA2005), description="Unweighted undirected clustering coefficient")
@due.dcite(BibTeX(FAGIOLO2007), description="Unweighted undirected clustering coefficient")
@due.dcite(BibTeX(WATTS1998), description="Weighted directed clustering coefficient")
@due.dcite(BibTeX(ONNELA2005), description="Weighted directed clustering coefficient")
@due.dcite(BibTeX(FAGIOLO2007), description="Weighted directed clustering coefficient")
@due.dcite(BibTeX(WATTS1998), description="Weighted undirected clustering coefficient")
@due.dcite(BibTeX(ONNELA2005), description="Weighted undirected clustering coefficient")
@due.dcite(BibTeX(FAGIOLO2007), description="Weighted undirected clustering coefficient")
@due.dcite(
    BibTeX(ONNELA2005),
    description="Weighted undirected signed clustering coefficient (Onnela)",
    conditions={(1, "coef_type"): {"DC_DEFAULT", "default"}}
)
@due.dcite(
    BibTeX(COSTANTINI2014),
    description="Weighted undirected signed clustering coefficient (Costatini)",
    conditions={(1, "coef_type"): {"costantini"}}
)
@due.dcite(
    BibTeX(ZHANG2005),
    description="Weighted undirected signed clustering coefficient (Zhang)",
    conditions={(1, "coef_type"): {"zhang"}}
)

class ClusteringCoefficientInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="Numpy matrix file",
        mandatory=True
        )

    algorithm_type = traits.Enum(
        'weighted_undirected',
        'weighted_undirected_signed',
        'weighted_directed',
        'binary_undirected',
        'binary_directed',
        mandatory=True
    )

class ClusteringCoefficientOutputSpec(TraitedSpec):
    clustering_coefficient = File(
        exists=True,
        desc="clustering coefficient file"
    )
    clustering_coefficient_positive = File(
        exists=True,
        desc="clustering coefficient positive file"
    )
    clustering_coefficient_negative = File(
        exists=True,
        desc="clustering coefficient negative file"
    )

class ClusteringCoefficient(BaseInterface):
    input_spec = ClusteringCoefficientInputSpec
    output_spec = ClusteringCoefficientOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        np_mat = np.load(np_mat_file)

        algo_type = self.inputs.algo_type

        if np_mat.ndim != 2:
            raise ValueError(f"Input matrix file has wrong dimension: {np_mat.ndim}. Actually needs 2")
        if np_mat.shape[0] != np_mat.shape[1]:
            raise ValueError(f"Input matrix file matrix is not equal in dimension: {np_mat.shape[0]} and {np_mat.shape[1]}")

        match algo_type:
            case "weighted_undirected":
                if not np.equal(np_mat, np_mat.T):
                    raise ValueError("Weighted undirected clustering coefficient algorithm needs a symmetric np.mat")

                clustering_coef = clustering_coef_wu(np_mat)

                np.save(os.path.abspath("clustering_coef_wu"), clustering_coef)

            case "weighted_undirected_signed":
                if not np.equal(np_mat, np_mat.T):
                    raise ValueError ("Weighted undirected signed clustering coefficient algorithm needs a symmetric np.mat")

                clustering_coef = clustering_coef_wu_sign(np_mat)
                np.save(os.path.abspath("clustering_coef_wu_sign"), clustering_coef)

            case "weighted_directed":

                clustering_coef = clustering_coef_wd(np_mat)
                np.save(os.path.abspath("clustering_coef_wd"), clustering_coef)

            case "binary_undirected":
                if not np.equal(np_mat, np_mat.T):
                    raise ValueError ("Binary undirected signed clustering coefficient algorithm needs a symmetric np.mat")

                #TODO: Find out how to check for binary matrix

                clustering_coef = clustering_coef_bu(np_mat)
                np.save(os.path.abspath("clustering_coef_bu"), clustering_coef)

            case "binary_directed":
                # TODO: Find out how to check for binary matrix

                clustering_coef = clustering_coef_bd(np_mat)
                np.save(os.path.abspath("clustering_coef_bd"), clustering_coef)

            case _:
                raise ValueError(f"Unknown algorithm type: {algo_type} or None was provided")


    def _list_outputs(self):
        outputs = self._outputs().get()
        algo_type = self.inputs.algo_type

        match algo_type:
            case "weighted_undirected":
                outputs["clustering_coef"] = os.path.abspath("clustering_coef_wu")
            case "weighted_undirected_signed":
                outputs["clustering_coef"] = os.path.abspath("clustering_coef_wu_sign")
            case "weighted_directed":
                outputs["clustering_coef"] = os.path.abspath("clustering_coef_wd")
            case "binary_undirected":
                outputs["clustering_coef"] = os.path.abspath("clustering_coef_bu")
            case "binary_directed":
                outputs["clustering_coef"] = os.path.abspath("clustering_coef_bd")


#TODO: continue from here


@due.dcite(BibTeX(LANCICHINETTI2012), description="Undirected consensus partitioning")
def consensus_und(D, tau, reps=1000, seed=None):
    '''
    This algorithm seeks a consensus partition of the
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.

    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.

    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.

    Parameters
    ----------
    D : NxN np.ndarray
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.

    Returns
    -------
    ciu : Nx1 np.ndarray
        consensus partition
    '''
    rng = get_rng(seed)
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology

        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci

        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
            # count+=1
            # print count,c,dup
            # if count>10:
            #	class QualitativeError(): pass
            #	raise QualitativeError()
        return np.transpose(ciu)

    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                cis[:, i], _ = modularity_louvain_und_sign(dt, seed=rng)
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = agreement(cis) / reps

    return np.squeeze(ciu + 1)


def get_components(A, no_depend=False):
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : Any
        Does nothing, included for backwards compatibility

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    matlab code, although the component topology is.

    Many thanks to Nick Cullen for providing this implementation
    '''

    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')
    
    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)

    edge_map = [{u,v} for u in range(n) for v in range(n) if A[u,v] == 1]
    union_sets = []
    for item in edge_map:
        temp = []
        for s in union_sets:

            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        union_sets = temp

    comps = np.array([i+1 for v in range(n) for i in 
        range(len(union_sets)) if v in union_sets[i]])
    comp_sizes = np.array([len(s) for s in union_sets])

    return comps, comp_sizes


def get_components_old(A, no_depend=False):
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    adj : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : bool
        If true, doesn't import networkx to do the calculation. Default value
        is false.

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    although the component topology is.

    Note: networkx is used to do the computation efficiently. If networkx is
    not available a breadth-first search that does not depend on networkx is
    used instead, but this is less efficient. The corresponding BCT function
    does the computation by computing the Dulmage-Mendelsohn decomposition. I
    don't know what a Dulmage-Mendelsohn decomposition is and there doesn't
    appear to be a python equivalent. If you think of a way to implement this
    better, let me know.
        '''
    # nonsquare matrices cannot be symmetric; no need to check

    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)

    try:
        if no_depend:
            raise ImportError()
        else:
            import networkx as nx
        net = nx.from_numpy_matrix(A)
        cpts = list(nx.connected_components(net))

        cptvec = np.zeros((n,))
        cptsizes = np.zeros(len(cpts))
        for i, cpt in enumerate(cpts):
            cptsizes[i] = len(cpt)
            for node in cpt:
                cptvec[node] = i + 1

    except ImportError:
        # if networkx is not available use less efficient breadth first search
        cptvec = np.zeros((n,))
        r, _ = breadthdist(A)
        for node, reach in enumerate(r):
            if cptvec[node] > 0:
                continue
            else:
                cptvec[np.where(reach)] = np.max(cptvec) + 1

        cptsizes = np.zeros(np.max(cptvec))
        for i in np.arange(np.max(cptvec)):
            cptsizes[i] = np.size(np.where(cptvec == i + 1))

    return cptvec, cptsizes


def number_of_components(A):
    _, csizes = get_components(A)
    return len(csizes)


@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, unweighted directed")
def transitivity_bd(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary directed connection matrix

    Returns
    -------
    T : float
        transitivity scalar

    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8
    triangles (2*2*2 edges). The number of existing triangles is the main

    diagonal of S^3/2. The number of all (in or out) neighbour pairs is
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of
    false pairs is the main diagonal of A^2. Thus the maximum possible
    number of triangles = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
                        = 2 * (K(K-1)/2 - diag(A^2))
                        = K(K-1) - 2(diag(A^2))
    '''
    S = A + A.T  # symmetrized input graph
    K = np.sum(S, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))  # number of all possible 3-cycles
    return np.sum(cyc3) / np.sum(CYC3)


@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, unweighted undirected")
def transitivity_bu(A):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    T : float
        transitivity scalar
    '''
    tri3 = np.trace(np.dot(A, np.dot(A, A)))
    tri2 = np.sum(np.dot(A, A)) - np.trace(np.dot(A, A))
    return tri3 / tri2


@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, weighted directed")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, weighted directed")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, weighted directed")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, weighted directed")
def transitivity_wd(W):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    T : int
        transitivity scalar

    Methodological note (also see note for clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make T=0
    # number of all possible 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    return np.sum(cyc3) / np.sum(CYC3)  # transitivity


@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, weighted undirected")
def transitivity_wu(W):
    '''
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    T : int
        transitivity scalar
    '''
    K = np.sum(np.logical_not(W == 0), axis=1)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    return np.sum(cyc3, axis=0) / np.sum(K * (K - 1), axis=0)

def path_transitivity(W, transform=None):
    '''
    This function computes the density of local detours (triangles) that
    are available along the shortest-paths between all pairs of nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted or unweighted undirected connection weight or length matrix
    transform : None or enum
        if the input is a connection length matrix, no transform is needed
        if the input is a connection weight matrix, use 'log' for
        log transform l_ij = -log(w_ij)
        or 'inv' for inversion l_ij = 1/w_ij
        The default value is None

    Returns
    -------
    T : NxN
        matrix of pairwise path transitivity
    '''
    n = len(W)
    m = np.zeros((n, n))
    T = np.zeros((n, n))

    for i in range(n-1):
        for j in range(i+1, n):
            x = 0
            y = 0
            z = 0
            
            for k in range(n):
                if W[i, k] != 0 and W[j, k] != 0 and k not in (i, j):
                    x += W[i, k] + W[j, k]
                if k != j:
                    y += W[i, k]
                if k != i:
                    z += W[j, k]

            m[i,j] = x/(y+z)

    m = m + m.T

    _, hops, pmat = distance_wei_floyd(W, transform=transform)

    for i in range(n-1):
        for j in range(i+1, n):

            x = 0
            path = retrieve_shortest_path(i, j, hops, pmat)
            k = len(path)
            print(path)
            print(k)

            for t in range(k-1):
                for l in range(t+1, k):
                    x += m[path[t], path[l]]

            T[i, j] = 2 * x / (k * (k - 1))

    T = T + T.T
    return T
