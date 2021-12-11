import torch as tch
import numpy as np

from .network import Network
from .markovchain import AbsorbingMC

from ..utils import AgentId

ProbsEmbs = list[tuple[tch.Tensor, tch.Tensor, tch.Tensor]]


class EDTSolver:
    """Calculates the expected delivery time between two nodes"""

    @staticmethod
    def get_q_vals_prob(q_vals:       tch.Tensor,
                        temp:         float,
                        smooth_param: float) -> tch.Tensor:
        """Get the smoothed probability of routing to the first
        neighbor
        """

        prob = (q_vals / temp).softmax(dim=0)[0]

        return EDTSolver.smooth_prob(prob, smooth_param)

    @staticmethod
    def smooth_prob(prob, param):
        return (1 - param) * prob + param / 2

    def __init__(self,
                 chain:        AbsorbingMC,
                 onode:        AgentId,  # source node
                 network:      Network,
                 temp:         float,  # sigmoid temperature
                 # Probability smoothing parameter
                 smooth_param: float) -> None:
        self.onode = onode
        self.network = network
        self.temp = temp
        self.smooth_param = smooth_param

        self.calc_edt, nontriv_dvtrs = chain.get_edt_func(onode)

        # Pack the embeddings to calculate the routing probability for
        # each nontrivial diverter
        self.probs_embs: ProbsEmbs = []
        iemb = tch.FloatTensor(network.get_node_emb(chain.sink))
        for node in nontriv_dvtrs:
            demb = tch.FloatTensor(network.get_node_emb(node))
            nembs = tch.FloatTensor(np.array(network.get_nebr_embs(node)))

            self.probs_embs.append((demb, nembs, iemb))

    @property
    def n_dvtrs(self):
        """Number of nontrivial diverters"""

        return len(self.probs_embs)

    def get_edt_val(self, probs_embs: ProbsEmbs = None) -> tch.Tensor:
        if probs_embs is None:
            probs_embs = self.probs_embs

        probs = []
        for demb, nembs, iemb in probs_embs:
            dembs = demb.repeat(2, 1)
            iembs = iemb.repeat(2, 1)

            q_vals = self.network.calc_q_vals(dembs, nembs, iembs)
            prob = EDTSolver.get_q_vals_prob(q_vals,
                                             self.temp,
                                             self.smooth_param)
            probs.append(prob)
        print('probs')
        print(probs)
        return self.calc_edt(*probs)
