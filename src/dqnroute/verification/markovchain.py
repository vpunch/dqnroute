from typing import *

import sympy as sym
from sympy.matrices import eye, zeros, ones
from sympy.solvers.solveset import linsolve
import networkx as nx
import itertools

from .network import Network

from ..utils import AgentId


class AbsorbingMC:
    """The class to model the delivery process as a discrete Markov
    chain, where the states correspond to the nodes of the network and
    the transitions correspond to conveyor sections. The transitions are
    weighted with conveyor section lengths.
    """

    def __init__(self,
                 network: Network,
                 sink:    AgentId,
                 is_spc:  bool,  # simple path cost
                 verbose: bool) -> None:
        self.sink = sink
        self.is_spc = is_spc
        self.verbose = verbose

        self.__create_absorbing_chain(network)

        self.__find_edt_sol()

    def __create_absorbing_chain(self, network: Network) -> None:
        chain = network.graph.copy()
        # A sink must be an absorbing state
        chain.remove_edges_from(list(chain.edges(self.sink)))

        # We are interested in the delivery of a bag only to our stock,
        # so we need to remove the rest of the stocks
        # We also need to remove nodes from which we cannot get into an
        # absorbing state (absorbing chain property)
        for node in network.graph:
            if chain.has_node(node) and node != self.sink:
                ds = list(nx.descendants(chain, node))
                if self.sink in ds:
                    continue

                # If we cannot get into the sink from the node, then we
                # cannot get there from the descendants of the node
                chain.remove_nodes_from(ds + [node])

        for f, s in chain.edges:
            chain[f][s]['length'] = network.get_section_len(f, s)

        #chain.add_edge(self.sink, self.sink)

        if self.verbose:
            print(f'Absorbing Markov chain for node {self.sink} is created')
            network.print(f'amc-{self.sink}', chain)

        self.chain = chain

    def __find_edt_sol(self):
        """Find a solution to the problem of finding the expected
        delivery time of a bag
        """

        nodes = list(self.chain)
        self.node_idx = dict(zip(nodes, itertools.count()))

        # We can use P instead of Q for convenience

        # h = (I - P)^-1 * 1
        # let h = x, 1 = b, then (I - P)x = b

        P = zeros(self.chain.number_of_nodes())
        b = ones(P.rows, 1)
        I = eye(P.rows)

        for node in self.chain:
            idx = self.node_idx[node]

            if node == self.sink:
                b[idx] = 0
                continue

            nebrs = list(self.chain.successors(node))
            nebr_idxs = [self.node_idx[nebr] for nebr in nebrs]

            if len(nebrs) == 2:
                p = sym.symbols(f'p{node[1]}')

                P[idx, nebr_idxs[0]] = p
                P[idx, nebr_idxs[1]] = 1 - p

                if not self.is_spc:
                    f, s = [self.chain[node][nebr]['length'] for nebr in nebrs]
                    b[idx] = f * p + s * (1 - p)
            elif len(nebrs) == 1:
                P[idx, nebr_idxs[0]] = 1

                if not self.is_spc:
                    b[idx] = self.chain[node][nebrs[0]]['length']
            else:
                assert False

        self.edt_sol, = linsolve(((I - P), b))

        if self.verbose:
            print(f'Expected delivery time solution for {self.sink}')
            print(self.edt_sol)

    def get_edt_func(self, source: AgentId) -> tuple[Callable, list[AgentId]]:
        """Get the expected delivery time as a function of routing
        probabilities
        """

        sol = sym.simplify(self.edt_sol[self.node_idx[source]])

        params = []
        nontriv_dvtrs = []
        for s in sol.free_symbols:
            params.append(s)
            nontriv_dvtrs.append(('diverter', int(s.name[1:])))

        calc_edt = sym.lambdify(params, sol)

        if self.verbose:
            print(f'E({source} -> {self.sink}) = {sol}')

        return calc_edt, nontriv_dvtrs
