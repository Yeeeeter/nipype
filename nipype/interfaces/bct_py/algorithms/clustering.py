from __future__ import division, print_function

import logging
from unittest import case

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


@due.dcite(BibTeX(LANCICHINETTI2012), description="Undirected consensus partitioning")
class ConsensusInputSpec(BaseInterfaceInputSpec):
    agreement_mat_file = File(
        exists=True,
        mandatory=True
    )
    tau = traits.Float()
    reps = traits.Int(
        default_value = 1000
    )
    seed = traits.Int(
    )

class ConsensusOutputSpec(TraitedSpec):
    consensus_part = File(exists=True)


class Consensus(BaseInterface):
    input_spec = ConsensusInputSpec
    output_spec = ConsensusOutputSpec

    def _run_interface(self, runtime):
        agreement_mat_file = self.inputs.agreement_mat_file
        tau = self.inputs.tau
        reps = self.inputs.reps
        seed = self.inputs.seed

        agreement_mat = np.load(agreement_mat_file)

        if not (agreement_mat.ndim == 2 or (agreement_mat.shape[0] == agreement_mat.shape[1])):
            raise ValueError("Agreement matrix must be square")

        if not tau < 0:
            raise ValueError("tau must be positive")

        # TODO: check how to better tie in the logger of nipype!!!
        if tau == 0:
            logging.warning("No tau given, assuming 0")

        if seed == 0:
            logging.warning("seed is 0, assuming no seed was given")
            consensus_part = consensus_und(agreement_mat, tau, reps)
        else:
            consensus_part = consensus_und(agreement_mat, tau, reps, seed)

        np.save(os.path.abspath("consensus_part"), consensus_part)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["consensus_part"] = os.path.abspath("consensus_part")


class GetComponentInputSpec(BaseInterfaceInputSpec):
    adjacency_mat_file = File(
        exists=True,
        mandatory=True
    )

class GetComponentOutputSpec(TraitedSpec):
    component_assignments = File(exists=True)
    component_sizes = File(exists=True)

class GetComponent(BaseInterface):
    input_spec = GetComponentInputSpec
    output_spec = GetComponentOutputSpec

    def _run_interface(self, runtime):
        adjacency_mat_file = self.inputs.adjecency_mat_file

        adjacency_mat = np.load(adjacency_mat_file)

        if not adjacency_mat.ndim == 2 or (adjacency_mat.shape[0] == adjacency_mat.shape[1]):
            raise ValueError("Adjacency matrix must be square")

        if not np.equal(adjacency_mat, adjacency_mat.T):
            raise ValueError("GetComponent requires undirected adjacency matrix")

        # TODO research way to check for binary matrix

        comps, comp_sizes = get_components(adjacency_mat)

        np.save(os.path.abspath("component_assignments"), comps)
        np.save(os.path.abspath("component_sizes"), comp_sizes)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["component_assignments"] = os.path.abspath("component_assignments")
        outputs["component_sizes"] = os.path.abspath("component_sizes")


@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, unweighted directed")
@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, unweighted undirected")
@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, weighted directed")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, weighted directed")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, weighted directed")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, weighted directed")
@due.dcite(BibTeX(RUBINOV2010), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(ONNELA2005), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(FAGIOLO2007), description="Transitivity, weighted undirected")
@due.dcite(BibTeX(HUMPHRIES2008), description="Transitivity, weighted undirected")
class TransitivityInputSpec(BaseInterfaceInputSpec):
    algorithm_type = traits.Enum(
        "None",
        "weighted_undirected",
        "weighted_directed",
        "binary_undirected",
        "binary_directed"
    )
    np_mat_file = File(
        exists=True,
        mandatory=True
    )

class TransitivityOutputSpec(TraitedSpec):
    transitivity = traits.Float()


class Transitivity(BaseInterface):
    input_spec = TransitivityInputSpec
    output_spec = TransitivityOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        algorithm = self.inputs.algorithm_type

        np_mat = np.load(np_mat_file)

        if not (np_mat.ndim == 2 or (np_mat.shape[0] == np_mat.shape[1])):
            raise ValueError("Matrix must be square")

        match algorithm:
            case "None":
                raise Exception("Transitivity: No algorithm specified!")

            case "weighted_undirected":
                if not np.equal(np_mat, np_mat.T):
                    raise ValueError("Transitivity: Matrix must be square")
                else:
                    transitivity = transitivity_wu(np_mat)

            case "weighted_directed":
                transitivity = transitivity_wd(np_mat)

            case "binary_undirected":
                if not np.equal(np_mat, np_mat.T):
                    raise ValueError("Transitivity: Matrix must be square")
                else:
                    transitivity = transitivity_bu(np_mat)

            case "binary_directed":
                transitivity = transitivity_bd(np_mat)

            case _:
                raise Exception("Transitivity: Unknown algorithm parameter provided!")


        self.transitivity = transitivity

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["transitivity"] = self.transitivity


# TODO decide if this function should get an interface
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
