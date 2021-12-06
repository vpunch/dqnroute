from typing import *

import sympy as sym
from sympy.matrices import eye, zeros, ones
from sympy.solvers.solveset import linsolve
import networkx as nx
import itertools

from .router_graph import RouterGraph
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
                 is_spc:  bool,
                 verbose: bool):
        self.sink = sink
        self.is_spc = is_spc
        self.verbose = verbose

        self.__create_absorbing_chain(network)

        self.__find_edt_sol()

    def __create_absorbing_chain(self, network: RouterGraph):
        # TODO копировать с оставлением длины ребер, длину переименовать в
        # label
        chain = network.graph.copy()
        # A sink must be an absorbing state
        chain.remove_edges_from(list(chain.edges(self.sink)))

        # TODO здесь  надо переписать с использованием
        # матрицы достижимости

        # We are interested in the delivery of a bag only to our stock,
        # so we need to remove the rest of the stocks
        # We also need to remove nodes from which we cannot get into an
        # absorbing state
        for node in network.graph:
            if chain.has_node(node) and node != self.sink:
                d = list(nx.descendants(chain, node))
                if self.sink in d:
                    continue

                # If we cannot get into the sink from the node, then we
                # cannot get there from the descendants of the node
                chain.remove_nodes_from(d + [node])

        for f, s in chain.edges:
            chain[f][s]['label'] = network.get_edge_length(f, s)

        chain.add_edge(self.sink, self.sink)

        if self.verbose:
            A = nx.drawing.nx_agraph.to_agraph(chain)
            A.node_attr['shape'] = 'box'
            A.draw(f'amc-{self.sink}.png', prog='dot')

            print(f'Absorbing Markov chain for node {self.sink} is created')
            print(A)

        self.chain = chain

    def __find_edt_sol(self):
        """Find a solution to the problem of finding the expected
        delivery time of a bag
        """

        nodes = list(self.chain)

        self.node_to_id = dict(zip(nodes, itertools.count()))

        # We can use P instead of Q for convenience

        # h = (I - P)^-1 * 1
        # let h = x, 1 = b, then (I - P)x = b

        P = zeros(self.chain.number_of_nodes())
        b = ones(P.rows, 1)
        I = eye(P.rows)

        for node in self.chain:
            nid = self.node_to_id[node]

            if node == self.sink:
                b[nid] = 0
                continue

            nbrs = list(self.chain.successors(node))
            nbr_ids = [self.node_to_id[nbr] for nbr in nbrs]

            if len(nbrs) == 2:
                p = sym.symbols(f'p{node[1]}')

                P[nid, nbr_ids[0]] = p
                P[nid, nbr_ids[1]] = 1 - p

                if not self.is_spc:
                    f, s = [self.chain[node][nbr]['label'] for nbr in nbrs]
                    b[nid] = f * p + s * (1 - p)
            elif len(nbrs) == 1:
                P[nid, nbr_ids[0]] = 1

                if not self.is_spc:
                    b[nid] = self.chain[node][nbrs[0]]['label']
            else:
                assert False

        self.edt_sol, = linsolve(((I - P), b))

        if self.verbose:
            print(f'Expected delivery time solution for {self.sink}')
            print(self.edt_sol)

    def get_edt_sol(self, src_node) -> tuple[sym.Expr, Callable]:
        """Get the expected delivery time as a function of routing
        probabilities.
        """

        src_sol = sym.simplify(self.edt_sol[self.node_to_id[src_node]])

        self.params = []
        self.nontrivial_dvtrs = []
        for s in src_sol.free_symbols:
            self.params.append(s)
            self.nontrivial_dvtrs.append(('diverter', int(s.name[1:])))

        calc_edt = sym.lambdify(self.params, src_sol)

        if self.verbose:
            print(f"E({src_node} -> {self.sink}) = {src_sol}")

        return src_sol, calc_edt
