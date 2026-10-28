from __future__ import division, print_function
import numpy as np
from zipfile import BadZipFile
from scipy import sparse

from nipype import BaseInterface
from pipeline.engine.tests.test_nodes import FailCommandLine
from .core import kcore_bd, kcore_bu
from .distance import reachdist
from bct.utils import invert, BCTParamError
from ..due import due, BibTeX
from ..citations import (
    BRANDES2001, KINTALI2008, SHANNON1948, RUBINOV2011, NEWMAN2016, HONEY2007,
    HAGMANN2008, GUIMERA2005, MORRISON2005, BOLDI2009, ESTRADA2005, ESTRADA2010
)

from bct import *
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

@due.dcite(BibTeX(BRANDES2001), description="Unweighted betweenness centrality")
@due.dcite(BibTeX(KINTALI2008), description="Unweighted betweenness centrality")
@due.dcite(BibTeX(BRANDES2001), description="Weighted betweenness centrality")
@due.dcite(BibTeX(KINTALI2008), description="Weighted betweenness centrality")
@due.dcite(BibTeX(BRANDES2001), description="Unweighted edge betweenness centrality")
@due.dcite(BibTeX(BRANDES2001), description="Weighted edge betweenness centrality")
class BetweennessCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    is_weighted = traits.Bool(
        default_value=False,
        usedefault=True,
        desc="whether to apply the weighted betweenness centrality"
    )
    include_edges = traits.Bool(
        default_value=False,
        usedefault=True,
    )

class BetweennessCentralityOutputSpec(TraitedSpec):
    betweenness = File(
        exists=True,
        desc="output vector file with betweenness values"
    )

    edge_betweenness = File(
        exists=True,
        desc="output vector file with edge betweenness values"
    )



class BetweennessCentrality(BaseInterface):
    input_spec = BetweennessCentralityInputSpec
    output_spec = BetweennessCentralityOutputSpec

    def _run_interface(self, runtime):
        # TODO: Find out how to check matrix to be a of connection length

        np_mat_file = self.inputs.np_mat_file
        is_weighted = self.inputs.is_weighted
        include_edges = self.inputs.include_edges

        np_mat = np.load(np_mat_file)

        # check the matrix to be symmetric:

        # if not (np.equal(np_mat, np_mat.T)):
        #     raise ValueError("BetweennessCentrality requires matrix to be symmetric")

        if is_weighted:
            if include_edges:
                edge_bet, betweenness = edge_betweenness_wei(np_mat)
            else:
                betweenness = betweenness_wei(np_mat)
        else:
            if include_edges:
                edge_bet, betweenness = edge_betweenness_bin(np_mat)
            else:
                betweenness = betweenness_bin(np_mat)

        if include_edges:
            np.save(os.path.abspath("edge_bet.npy"), edge_bet)
        np.save(os.path.abspath("betweenness.npy"), betweenness)
        return runtime

    def _list_outputs(self):
        include_edges = self.inputs.include_edges
        outputs = self.output_spec().get()
        outputs["betweenness"] = os.path.abspath("betweenness.npy")
        if include_edges:
            outputs["edge_betweenness"] = np.load(os.path.abspath("edge_bet.npy"))

        return outputs


@due.dcite(BibTeX(RUBINOV2011), description="Entropy-based diversity coefficient")
@due.dcite(BibTeX(SHANNON1948), description="Entropy-based diversity coefficient")
class DiversityCoefficientInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    community_affiliation = File(
        exists=True,
        desc="community affiliation",
        mandatory=True
    )

class DiversityCoefficientOutputSpec(TraitedSpec):
    diversity_coef_hpos = File(exists=True, desc="output vector file")
    diversity_coef_hneg = File(exists=True, desc="output vector file")

class DiversityCoefficient(BaseInterface):
    input_spec = DiversityCoefficientInputSpec
    output_spec = DiversityCoefficientOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        community_affiliation = self.inputs.community_affiliation

        np_mat = np.load(np_mat_file)
        comm_aff = np.load(community_affiliation)

        hpos, hneg = diversity_coef_sign(np_mat, comm_aff)
        np.save(os.path.abspath("diversity_coef_hpos.npy"), hpos)
        np.save(os.path.abspath("diversity_coef_hneg.npy"), hneg)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["diversity_coef_hpos"] = os.path.abspath("diversity_coef_hpos.npy")
        outputs["diversity_coef_hneg"] = os.path.abspath("diversity_coef_hneg.npy")
        return outputs


@due.dcite(BibTeX(NEWMAN2016), description="Eigenvector centrality")
class EigenvectorCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )

class EigenvectorCentralityOutputSpec(TraitedSpec):
    highest_eigenvector = File(exists=True, desc="output vector file")

class EigenvectorCentrality(BaseInterface):
    input_spec = EigenvectorCentralityInputSpec
    output_spec = EigenvectorCentralityOutputSpec
    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        np_mat = np.load(np_mat_file)

        if not np.equal(np_mat, np_mat.T):
            raise ValueError("EigenvectorCentrality (undirected) requires adjacency "
                             "matrix")

        high_eig_vec = eigenvector_centrality_und(np_mat)
        np.save(os.path.abspath("highest_eigenvector.npy"), high_eig_vec)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["highest_eigenvector"] = np.load(os.path.abspath("highest_eigenvector.npy"))

        return outputs


class ERangeInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )

class ERangeOutputSpec(TraitedSpec):
    erange = File(exists=True, desc="range of each edge")
    eta = traits.Float(desc="average range for the entire graph")
    eshort = File(exists=True, desc="shortcut edges")
    frac = traits.Float(desc="fraction of shortcuts in the graph")

class ERange(BaseInterface):
    input_spec = ERangeInputSpec
    output_spec = ERangeOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        np_mat = np.load(np_mat_file)

        erange, eta, eshort, fs = erange(np_mat)
        np.save(os.path.abspath("erange.npy"), erange)
        np.save(os.path.abspath("eshort.npy"), eshort)
        self._eta = eta
        self._frac = fs

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["erange"] = np.load(os.path.abspath("erange.npy"))
        outputs["eshort"] = np.load(os.path.abspath("eshort.npy"))
        outputs["fs"] = self._eta
        outputs["frac"] = self._frac

        return outputs


@due.dcite(BibTeX(HONEY2007), description="Flow coefficient")
class FlowCoefficientInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )

class FlowCoefficientOutputSpec(TraitedSpec):
    flow_coefficient_file = File(exists=True, desc="flow coefficient file")
    average_flow_coefficient = traits.Float(desc="average flow coefficient")
    total_flow_coefficient = traits.Int(desc="total flow coefficient")

class FlowCoefficientCentrality(BaseInterface):
    input_spec = FlowCoefficientInputSpec
    output_spec = FlowCoefficientOutputSpec
    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        np_mat = np.load(np_mat_file)

        fc, FC, total_flo = flow_coef_bd(np_mat)
        self._average_flow_coefficient = fc
        self._total_flow_coefficient = total_flo
        np.save(os.path.abspath("flow_coefficient.npy"), fc)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["flow_coefficients"] = np.load(os.path.abspath("flow_coefficient.npy"))
        outputs["average_flow_coefficient"] = self._average_flow_coefficient
        outputs["total_flow_coefficient"] = self._total_flow_coefficient



class GatewayCoefficientInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    community_affiliation = File(
        exists=True,
        desc="community affiliation",
        mandatory=True
    )
    centrality_type = traits.Enum(
        'degree', 'betweenness'
    )

class GatewayCoefficientOutputSpec(TraitedSpec):
    gateway_coefficient_pos = File(desc="gateway coefficient file")
    gateway_coefficient_neg = File(desc="gateway coefficient file")

class GatewayCoefficientCentrality(BaseInterface):
    input_spec = GatewayCoefficientInputSpec
    output_spec = GatewayCoefficientOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        np_mat = np.load(np_mat_file)
        community_affiliation = self.inputs.community_affiliation
        centrality_type = self.inputs.centrality_type

        gpos, gneg = gateway_coef_sign(np_mat, community_affiliation, centrality_type)

        np.save(os.path.abspath("gateway_coefficient_pos.npy"), gpos)
        np.save(os.path.abspath("gateway_coefficient_neg.npy"), gneg)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["gateway_coefficient_pos"] = np.load(
            os.path.abspath("gateway_coefficient_pos.npy")
        )
        outputs["gateway_coefficient_neg"] = np.load(
            os.path.abspath("gateway_coefficient_neg.npy")
        )
        return outputs


@due.dcite(BibTeX(HAGMANN2008), description="Unweighted directed k-coreness centrality")
@due.dcite(BibTeX(HAGMANN2008), description="Unweighted undirected k-coreness centrality")
class KCorenessCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    is_directed = traits.Bool(
        default=False,
        usedefault=True,
    )

class KCorenessCentralityOutputSpec(TraitedSpec):
    coreness = File(exists=True, desc="corness value")
    k_core_size = File(exists=True, desc="k-core size")


class KCorenessCentrality(BaseInterface):
    input_spec = KCorenessCentralityInputSpec
    output_spec = KCorenessCentralityOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        is_directed = self.inputs.is_directed
        np_mat = np.load(np_mat_file)

        symmetric = np.equal(np_mat, np_mat.T)

        if is_directed:
            #TODO: Find out how to raise Warning if matrix is symmetric
            #if symmetric:
            coreness, kn = kcoreness_centrality_bd(np_mat)

        else:
            if not symmetric:
                raise ValueError("KCorenessCentrality needs undirected matrix if is_directed=False")

            coreness, kn = kcoreness_centrality_bu(np_mat)

        np.save(os.path.abspath("coreness.npy"), coreness)
        np.save(os.path.abspath("k_core_size.npy"), kn)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["coreness"] = np.load(os.path.abspath("coreness.npy"))
        outputs["k_core_size"] = np.load(os.path.abspath("k_core_size.npy"))
        return outputs


@due.dcite(BibTeX(GUIMERA2005), description="Within-module degree z-score")
class ModuleDegreeCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    community_affiliation = File(
        exists=True,
        desc="community affiliation vector",
        mandatory=True
    )
    graph_type = traits.Enum(
        "undirected",
        "directed_out_degree",
        "directed_in_degree",
        "directed_in_out_degree",
        usedefault=True,
        mandatory=True,
    )

class ModuleDegreeCentralityOutputSpec(TraitedSpec):
    module_degree = File(exists=True, desc="module degree matrix")

class ModuleDegreeCentrality(BaseInterface):
    input_spec = ModuleDegreeCentralityInputSpec
    output_spec = ModuleDegreeCentralityOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        community_affiliation = self.inputs.community_affiliation
        graph_type = self.inputs.graph_type

        np_mat = np.load(np_mat_file)

        if not (np_mat.ndim == 2 and np_mat.shape[0] == np_mat.shape[1]):
            raise ValueError("ModuleDegreeCentrality needs of equal size")


        flag = 0
        match graph_type:
            case "undirected":
                flag = 0
            case "directed_out_degree":
                flag = 1
            case "directed_in_degree":
                flag = 2
            case "directed_in_out_degree":
                flag = 3
            case _:
                raise ValueError("ModuleDegreeCentrality: Unknown graph type")

        module_degree = module_degree_zscore(np_mat, community_affiliation, flag)
        np.save(os.path.abspath("module_degree.npy"), module_degree)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["module_degree"] = np.load(os.path.abspath("module_degree.npy"))

        return outputs



@due.dcite(BibTeX(MORRISON2005), description="PageRank centrality")
@due.dcite(BibTeX(BOLDI2009), description="PageRank centrality")
class PageRankCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="numpy matrix to apply the algorithm to",
        mandatory=True
    )
    damping_factor = traits.Float(
        desc="damping factor"
    )
    pagrank_probability = File(
        exists=True,
        desc="initial pagrank probability",
        mandatory=True
    )

class PageRankCentralityOutputSpec(TraitedSpec):
    pagerank_vector = File(exists=True, desc="PageRank vector")

class PageRankCentrality(BaseInterface):
    input_spec = PageRankCentralityInputSpec
    output_spec = PageRankCentralityOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file
        damping_factor = self.inputs.damping_factor
        pagrank_probability = self.inputs.pagrank_probability
        np_mat = np.load(np_mat_file)

        if not (np_mat.ndim == 2 and np_mat.shape[0] == np_mat.shape[1]):
            raise ValueError("ModuleDegreeCentrality needs of equal size")

        pagerank_vector = pagerank_centrality(np_mat, damping_factor, pagrank_probability)

        np.save(os.path.abspath("pagerank_vector.npy"), pagerank_vector)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["pagerank_vector"] = np.load(os.path.abspath("pagerank_vector.npy"))


@due.dcite(BibTeX(GUIMERA2005), description="Participation coefficient")
class ParticipationCoefficientInputSpec(BaseInterfaceInputSpec):
    mat_file = File(
        exists=True,
        desc="matrix file to apply the algorithm to",
        mandatory=True
    )
    community_affiliation = File(
        exists=True,
        desc="community affiliation",
        mandatory=True
    )
    algorithm = traits.Enum(
        "normal",
        "sparse",
        "signed",
        use_default=True,
        mandatory=True
    )
    degree = traits.Enum(
        "undirected",
        "directed_in_degree",
        "directed_out_degree",
        mandatory=False
    )

class ParticipationCoefficientOutputSpec(TraitedSpec):
    participation_coefficient = File(exists=False, desc="Participation vector")
    participation_coefficient_pos = File(exists=False, desc="Participation vector pos")
    participation_coefficient_neg = File(exists=False, desc="Participation vector neg")

class ParticipationCoefficientCentrality(BaseInterface):
    input_spec = ParticipationCoefficientInputSpec
    output_spec = ParticipationCoefficientOutputSpec
    def _run_interface(self, runtime):
        mat_file = self.inputs.np_mat_file
        community_affiliation = self.inputs.community_affiliation
        algorithm = self.inputs.algorithm
        degree = self.inputs.degree

        if algorithm == "sparse":
            try:
                mat = sparse.load_npz(mat_file)
            except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    EOFError,
                    BadZipFile,
            ) as exc:
                raise ValueError(
                    f"Could not load {str(mat_file)!r} as a SciPy sparse .npz file: {exc}"
                ) from exc
            # Reject other sparse formats rather than converting them.
            if mat.format != "csr":
                raise ValueError(
                    f"Expected CSR data in {str(mat_file)!r}; "
                    f"found sparse format {mat.format!r}."
                )

            try:
                mat.check_format(full_check=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid CSR structure in {str(mat_file)!r}: {exc}"
                ) from exc

        else:
            mat = np.load(mat_file)
            if not (mat.ndim == 2 and mat.shape[0] == mat.shape[1]):
                raise ValueError("ParticipationCoefficientCentrality needs of equal size")

        match algorithm:
            case "normal":
                missing = [
                    name
                    for name in ("degree",)
                    if not isdefined(getattr(self.inputs, name))
                ]

                #check if all required values exist
                if missing:
                    raise ValueError(f"ParticipationCoefficientCentrality (normal version is missing: {missing}")

                participation_c = participation_coef(mat, community_affiliation, degree)

                np.save(os.path.abspath("participation_coefficient"), participation_c)

            case "sparse":

                missing = [
                    name
                    for name in ("degree",)
                    if not isdefined(getattr(self.inputs, name))
                ]
                if missing:
                    raise ValueError(f"ParticipationCoefficientCentrality (sparse version) is missing: {missing}")

                participation_c = participation_coef_sparse(mat, community_affiliation, degree)

                np.save(os.path.abspath("participation_coefficient"), participation_c)

            case "signed":
                if not np.equal(mat == mat.T):
                    raise ValueError("ParticipationCoefficientCentrality (signed version) needs a symmetric/undirected matrix")
                participation_c_pos, participation_c_neg = participation_coef_sign(mat, community_affiliation)
                np.save(os.path.abspath("participation_coefficient_pos"), participation_c_pos)
                np.save(os.path.abspath("participation_coefficient_neg"), participation_c_neg)
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.inputs.algorithm == "signed":
            outputs["participation_coefficient_pos"] = np.load(os.path.abspath("participation_coefficient_pos"))
            outputs["participation_coefficient_neg"] = np.load(os.path.abspath("participation_coefficient_neg"))
        else:
            outputs["participation_coefficient"] = np.load(os.path.abspath("participation_coefficient"))
        
        return outputs


@due.dcite(BibTeX(ESTRADA2005), description="Subgraph centrality")
@due.dcite(BibTeX(ESTRADA2010), description="Subgraph centrality")
class SubgraphCentralityInputSpec(BaseInterfaceInputSpec):
    np_mat_file = File(
        exists=True,
        desc="Numpy adjacency matrix",
        mandatory=True
    )

class SubgraphCentralityOutputSpec(TraitedSpec):
    subgraph_centrality = File(exists=True, desc="Subgraph centrality")

class SubgraphCentrality(BaseInterface):
    input_spec = SubgraphCentralityInputSpec
    output_spec = SubgraphCentralityOutputSpec

    def _run_interface(self, runtime):
        np_mat_file = self.inputs.np_mat_file

        np_mat = sparse.load_npz(np_mat_file)

        Cs = subgraph_centrality(np_mat)

        np.save(os.path.abspath("subgraph_centrality"), Cs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subgraph_centrality"] = np.load(os.path.abspath("subgraph_centrality"))
        return outputs
