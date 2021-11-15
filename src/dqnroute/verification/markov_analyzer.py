from typing import *

import sympy as sym
from sympy.matrices import eye, zeros, ones
from sympy.solvers.solveset import linsolve
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
import matplotlib.pyplot as plt

from .router_graph import RouterGraph
from ..utils import AgentId

class MarkovAnalyzer:
    """The class to model the delivery process as a discrete Markov
    chain, where the states correspond to the nodes of the network and
    the transitions correspond to conveyor sections. The transitions are
    weighted with conveyor section lengths.
    """
    
    def __init__(self,
                 network: nx.DiGraph,
                 sink: AgentId,
                 simple_path_cost: bool,
                 verbose: bool):
        self.sink = sink
        self.simple_path_cost = simple_path_cost

        self.__create_absorbing_chain(network, verbose)

        self.__find_edt_sol()

    def __create_absorbing_chain(self, network: nx.DiGraph, verbose):
        chain = network.copy()
        # A sink must be an absorbing state
        chain.remove_edges_from(list(chain.edges(self.sink)))

        # We are interested in the delivery of the bag only to our stock, so we
        # need to remove the rest of the stocks
        # We also need to remove nodes from which we cannot get into an
        # absorbing state
        for node in list(network):
            if chain.has_node(node) and node != self.sink:
                d = list(nx.descendants(chain, node))
                if self.sink in d:
                    continue

                #if we cannot get into the drain from the node, then we cannot
                #get there from the children of the node 
                chain.remove_nodes_from(d + [node])

        # нам нужно удалить узлы, в которые мы не попадаем из истока, чтобы
        # убрать лишние дивертеры? Может достаточно прикрепить список
        # параметров к формуле?
        # тут код

        #chain.remove_nodes_from([('source', 0), ('diverter', 0)])

        #if True:
        #    nx.draw(chain, with_labels=True)
        #    plt.savefig("filename.png")
        if True:#verbose:
            fig = plt.figure()
            pos = graphviz_layout(chain, prog="dot")

            edge_labels = nx.get_edge_attributes(chain, 'length')
            nx.draw_networkx_edge_labels(chain, pos, edge_labels)

            nx.draw(chain, pos, with_labels=True)
            fig.savefig(f"filename{self.sink}.png")
            fig.show()

        self.chain = chain

    def __find_edt_sol(self):
        """Find a solution to the problem of finding the expected delivery time
        of a bag 
        """
        ## drop absorbing state

        #chain.remove_node(sink)
#

        import itertools

        nodes = list(self.chain)
        #nodes = [(0, key) for key in nodes if key[0] == "source"] \
        #               + [(2, key) for key in nodes if key[0] == "sink"] \
        #               + [(1, key) for key in nodes if key[0] not in ["source", "sink"]]
        #nodes = [x[1] for x in sorted(nodes)]

        self.node_to_id = dict(zip(nodes, itertools.count()))

        # find nontrivial diverters

        # we can use P insead Q для удобства

        # h = (I - P)^-1 * 1
        # let h = x, 1 = b, then (I - P)x = b

        P = zeros(self.chain.number_of_nodes())
        b = ones(P.rows, 1)
        I = eye(P.rows)
        self.params = []
        self.nontrivial_diverters = []

        for node in self.chain:
            nid = self.node_to_id[node]
            nbrs = list(self.chain.successors(node))
            nbr_ids = [self.node_to_id[nbr] for nbr in nbrs]

            if len(nbrs) == 2:
                self.nontrivial_diverters.append(node)

                p = sym.symbols(f'p{node[1]}')
                self.params.append(p)

                P[nid, nbr_ids[0]] = p
                P[nid, nbr_ids[1]] = 1 - p

                if not self.simple_path_cost:
                    f, s = [self.chain[node][nbr]['length'] for nbr in nbrs]
                    b[nid] = f * p + s * (1 - p)
            elif len(nbrs) == 1:
                P[nid, nbr_ids[0]] = 1

                if not self.simple_path_cost:
                    b[nid] = self.chain[node][nbrs[0]]['length']
            elif len(nbrs) == 0:
                # поглощающее состояние
                b[nid] = 0
                assert node == self.sink
            else:
                # cann't be more then 2 neighbors
                assert False

        #print(I - P)
        #print(b)
        #self.solution = (I - P).inv() @ b
        self.solution, = linsolve(((I - P), b))
        #print(self.solution)

    def get_edt_sol(self, source) -> Tuple[sym.Expr, Callable]:
        """
        Computes the expected delivery cost as a function of routing probabilities.
        :return (objective as SymPy expression, objective as Callable).
        """
        source_index = self.node_to_id[source]

        #source_index = self.reachable_nodes_to_indices[self.source]
        symbolic_objective = sym.simplify(self.solution[source_index])
        print(f"  E(delivery cost from {source} to {self.sink}) = {symbolic_objective}")
        objective = sym.lambdify(self.params, symbolic_objective)
        return symbolic_objective, objective
