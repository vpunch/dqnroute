import numpy as np
import torch as tch
import networkx as nx
import pygraphviz

from ..utils import AgentId
from ..simulation.conveyors import ConveyorsEnvironment


class Network:
    """Wrap of the conveyor network to make the interface more stable"""

    def __init__(self, world: ConveyorsEnvironment, verbose: bool) -> None:
        self.world = world

        self.__check_embeddings()

        # Store clear graph
        graph = nx.DiGraph()
        graph.add_edges_from(world.topology_graph.edges)
        self.graph = graph

        # Store mapping source/sink/junction/diverter id -> router id

        abct_router = world.handlers.items()[0][1]  # abstract router

        self.map_router_id = abct_router.node_mapping

        # Store neural network and node encoding method

        router = abct_router.routers.items()[0][1]
        # We need to increase analysis precision
        self.q_net = router.brain.double()
        self._idx_enc_method = router._nodeRepr

        self.reachable = self.__get_reachability_matrix()

    def get_node_num(node: AgentId) -> int:
        """Get absolute index for the node (node number)"""

        return self.map_router_id(node)[1]

    def get_section_len(f: AgentId, s: AgentId) -> int:
        return self.world.topology_graph[f][s]['length']

    def get_conv_idx(f: AgentId, s: AgentId) -> int:
        return self.world.topology_graph[f][s]['conveyor']

    def get_neighbors(self, node: AgentId) -> list[AgentId]:
        """Get the nodes that follow the specified node. For a diverter,
        the node that belongs to the same conveyor will be returned
        first.
        """

        out_edges = list(self.graph.out_edges(node))
        assert len(out_edges) <= 2,\
               'A node cannot have more then two neighbors'

        # If the node represents a diverter
        if len(out_edges) == 2:
            in_edges = list(self.graph.in_edges(node))
            assert len(in_edges) == 1, 'Diverter cannot be junction'

            conv_idx = self.get_conv_idx(*in_edges[0])
            out_edges.sort(
                    key=lambda edge: abs(self.get_conv_idx(*edge) - conv_idx))

        return [edge[1] for edge in out_edges]

    def get_node_emb(self, node: AgentId) -> np.ndarray:
        return self._idx_enc_method(self.get_node_num(node))

    def get_nebr_embs(self, node: AgentId) -> list[np.ndarray]:
        """Get embeddings of all neighbors of the specified node"""

        nebrs = self.get_neighbors(node)
        return [self.get_node_emb(node) for node in nebrs]

    def get_nodes_with_type(self, ntype: str) -> list[AgentId]:
        """Return identifiers of nodes that have the specified type"""

        return [node for node in list(self.graph) if node[0] == ntype]

    def print(self, name, graph=None) -> pygraphviz.AGraph:
        """Save network image to file"""

        if graph is None:
            graph = self.graph

        agraph = nx.drawing.nx_agraph.to_agraph(graph)

        A.node_attr.update({'shape':     'box',
                            'style':     'filled',
                            'fixedsize': True,
                            'width':     0.9,
                            'height':    0.7})

        node_bg = {'source':   '#8888FF',
                   'sink':     '#88FF88',
                   'diverter': '#FF9999',
                   'junction': '#EEEEEE'}

        for node in graph:
            anode = agraph.get_node(n)
            anode.attr.update({'fillcolor': node_bg[node[0]],
                               'label':     f'{node[0]} {node[1]}'})

        for f, s in graph.edges:
            aedge = agraph.get_edge(f, s)
            aedge.attr['label'] = f'{self.get_section_len(f, s)}, '\
                                  f'c{self.get_conv_idx(f, s)}'

        if self.verbose:
            print(agraph)

        agraph.draw(f'{name}.png', prog='dot')

    def calc_q_vals(self,
                    dembs: tch.Tensor,  # diverter embeddings
                    nembs: tch.Tensor,  # neighbours
                    # Sink
                    iembs: tch.Tensor) -> tch.Tensor:
        return self.q_net.forward(dembs, iembs, nembs)

    def __check_embeddings(self):
        """Check that all routers return the same embeddings for the
        corresponding nodes
        """

        embs = []
        for _, abct_router in self.world.handlers.items():
            for _, router in abct_router.routers.items():
                emb = np.concatenate([router._nodeRepr(i)
                                      for i in range(len(self.graph))])
                embs.append(emb)

        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                assert np.abs(emb[i] - emb[j]).max() == 0,\
                       'The embeddings of the same node are different. This '\
                       'may be caused by the nondeterminism in embeddings '\
                       'computing.'

    def __get_reachability_matrix(self):
        nodes = sorted(self.graph)
        reachable = {(f, s): f == s for f in nodes for s in nodes}

        for node in nodes:
            for d in list(nx.descendants(self.graph, node)):
                reachable[node, d] = True

        if self.verbose:
            for f in nodes:
                for s in nodes:
                    print(int(reachable[f, s]), end=" ")

                print(f)

        return reachable
