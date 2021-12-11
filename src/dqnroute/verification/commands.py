import time
from argparse import ArgumentParser

import torch as tch
import networkx as nx

from .markovchain import AbsorbingMC
from .network import Network
from .edtsolver import EDTSolver
from .adversary import PGDAdversary
from .nnet_verifier import NNetVerifier, VerificationResult

from ..simulation.conveyors import ConveyorsEnvironment


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--edt_bound', type=float,
            help='')

    parser.add_argument('--simple_path_cost', action='store_true',
            help='')

    parser.add_argument('--euclid_norm', action='store_true',
            help='')

    parser.add_argument('--max_perturbation_norm', type=float,
            help='')

    parser.add_argument('--sink', type=int,
            help='')

    parser.add_argument('--source', type=int,
            help='')

    parser.add_argument('--marabou_path', type=str,
            help='')

    parser.add_argument('--marabou_memory_limit', type=int,
            help='')

    parser.add_argument('--verify_dynamic_graph', action='store_true',
            help='')

    parser.add_argument('--verbose', action='store_true',
            help='')


def compute_edt(world:        ConveyorsEnvironment,
                sink:         int,
                source:       int,
                temp:         float,
                smooth_param: float,
                is_spc:       bool = True,
                verbose:      bool = False) -> float:
    network = Network(world, verbose)

    chain = AbsorbingMC(network, ('sink', sink), is_spc, verbose)

    edt_solver = EDTSolver(chain,
                           ('source', source),
                           network,
                           temp,
                           smooth_param)
    return edt_solver.get_edt_val().item()


def find_advers_emb(world:        ConveyorsEnvironment,
                    sink:         int,
                    source:       int,
                    temp:         float,
                    smooth_param: float,
                    max_pertb_l:  float,
                    is_spc:       bool = True,
                    is_euclid_l:  bool = False,
                    verbose:      bool = False) -> tuple[tch.Tensor, float]:
    network = Network(world, verbose)

    chain = AbsorbingMC(network, ('sink', sink), is_spc, verbose)

    edt_solver = EDTSolver(chain,
                           ('source', source),
                           network,
                           temp,
                           smooth_param)

    advers = PGDAdversary(max_pertb_l, 10, True, is_euclid_l, 100, 100000,
                          False, 0.02, verbose)
    return advers.perturb(edt_solver)


def verif_edt_bound_wrt_emb(world:           ConveyorsEnvironment,
                             sink:            int,
                             source:          int,
                             temp:            float,
                             smooth_param:    float,
                             max_pertb_l:     float,
                             marabou_path:    str,
                             edt_bound:       float,
                             is_spc:          bool = True,
                             is_dyn_net:      bool = False,
                             marabou_mem_lim: int = 0,
                             verbose:         bool = False)\
        -> VerificationResult:
    network = Network(world, verbose)
    if is_dyn_net:
        pass
    #network.maximize_load()

    chain = AbsorbingMC(network, ('sink', sink), is_spc, verbose)

    edt_solver = EDTSolver(chain,
                           ('source', source),
                           network,
                           temp,
                           smooth_param)

    verifier = NNetVerifier(network,
                               marabou_path,
                               '../network.nnet',
                               '../property.txt',
                               smooth_param,
                               temp,
                               10,
                               marabou_mem_lim)
    return verifier.verify_delivery_cost_bound(
                    ('source', source), ('sink', sink),
                    chain,
                    max_pertb_l, edt_bound)


#def verif_edt_robustness_wrt_emb(world:           ConveyorsEnvironment,
#                                 temp:            float,
#                                 smooth_param:    float,
#                                 max_pertb_l:     float,
#                                 marabou_path:    str,
#                                 is_spc:          bool = True,
#                                 marabou_mem_lim: int = 0,
#                                 verbose:         bool = False)\
#        -> VerificationResult:
#    network = Network(world)
#    get_pairs(network.graph)
#
#    grr = Network(gr)
#
#    log = []
#
#    counter = 0
#    for pair in get_pair(grr.graph):
#
#        print('')
#        print('')
#        print('')
#        print('')
#        print('')
#        print('')
#        print('')
#        print('')
#        source, sink = (pair[0], pair[-1])
#        print('Pair')
#        print(source, sink)
#
#
#
#    sink_pairs
#
#    row = [(source, sink)]
#
#    mc = AbsorbingMC(network, sink, is_spc, is_verbose)
#
#    for source:
#        edt_solver = EDTSolver(mc, onode, network)
#        if not edt_solver.nontriv_dvtrs:
#            print('')
#            continue
#
#        exit()
#
#        adv = PGDAdversary(edt_solver, rho=eps,
#                           steps=100,
#                           step_size=0.02,
#                           random_start=True,
#                           stop_loss=100,
#                           verbose=2,
#                           norm="l_inf",
#                           n_repeat=10,
#                           repeat_mode="any",
#                           dtype=torch.float64)
#
#        best_embedding = adv.perturb(embedding_packer.initial_vector(), get_gradient)
#        k = [0.95, 0.99, 1.01. 1.05]
#        _, t_top, aux_info = get_gradient(best_embedding)
#
#        for t_b in ks:
#            t_b = t_top * k
#            t = time.process_time()
#            #print('COST')
#            #print(const_bound)
#            result = verifier.verify_delivery_cost_bound(
#                    source, sink,
#                    ma,
#                    eps, const_bound)
#            #print(result)
#            elapsed_time = time.process_time() - t
#            #print('ELAPSED')
#            #print(elapsed_time)
#
##                $o_1 \to i_2$ & 83.84 & 55.71 & +   & 1.60 \\              |~
##                              &       & 50.41 &  -  & 0.13 ⏎ 
#
#        counter = counter + 1
#
#    print(counter)
#
#
#def get_pairs(graph):
#        # Найти все упорядоченные пары вершин
#        # Порядок важен, так как он влияет на входные данные
#        nodes = list(graph.nodes)
#        #pairs = [(a, b) for a in nodes for b in nodes if a != b]
#
#        # Оставляем старые стоки
#        pairs = [(a, b) for a in nodes for b in nodes if a != b]
#
#        # Найти кратчайшие пути между ними
#        # Кратчайший путь наиболее вероятен при маршрутизации
#
#
#        # Должен быть не один путь между парой вершни, чтобы 
#        # Анализатор обрезает все и вероятности не нужны
#        pathes = []
#        for pair in pairs:
#            try:
#                path = nx.dijkstra_path(graph, *pair, weight='_')
#
#                for node in path:
#                    if 'diverter' in node:
#                        pathes.append(path)
#                        break
#            except nx.NetworkXNoPath:
#                pass
#
#        pathes.sort(key=len)
#        # Выбрать самый длинный путь
#
#        # Должен присутствовать хотя бы одни разделитель
#
#        while pathes:
#            longest = pathes.pop()
#
#            yield longest
#
#            # Удалить пары, где конечная вершина совпадает, а начальная лежит на пути
#            # Так как кратчайший путь наиболее вероятен при маршрутизации, мы считаем,
#            # что нейронки будут получать эмбеддинги, для которых устойчивость уже
#            # доказана
#
#            cond = lambda e: e[-1] != longest[-1] or e[0] not in longest
#
#            # Для нас доставка на разделителях, которые ведут в другой сток
#            # всегда правильная
#            for path in pathes:
#                if not cond(path):
#                    print('AAAAAAAAAAAAAAAAAAAAAAAAA')
#                    print(path[0], path[-1])
#
#            pathes = list(filter(cond, pathes))
