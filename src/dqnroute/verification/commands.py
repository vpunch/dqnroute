import networkx as nx

from .markov_chain import absorbing_mc
from .network import Network

from .edt_solver import EDTSolver
from .adversarial import PGDAdversary

import torch

from typing import *

def full_emb_advers_verif(verifier,
                          graph,
                          is_spc,
                          is_verbose,
                          is_l2,
                          t_top,
                          softmax_tre,
                          prob_smooth):

    def get_pair(graph):
        # Найти все упорядоченные пары вершин
        # Порядок важен, так как он влияет на входные данные
        nodes = list(graph.nodes)
        #pairs = [(a, b) for a in nodes for b in nodes if a != b]

        # Оставляем старые стоки
        pairs = [(a, b) for a in nodes for b in nodes if a != b]

        # Найти кратчайшие пути между ними
        # Кратчайший путь наиболее вероятен при маршрутизации


        # Должен быть не один путь между парой вершни, чтобы 
        # Анализатор обрезает все и вероятности не нужны
        pathes = []
        for pair in pairs:
            try:
                path = nx.dijkstra_path(graph, *pair, weight='_')

                for node in path:
                    if 'diverter' in node:
                        pathes.append(path)
                        break
            except nx.NetworkXNoPath:
                pass

        pathes.sort(key=len)
        # Выбрать самый длинный путь

        # Должен присутствовать хотя бы одни разделитель

        while pathes:
            longest = pathes.pop()

            yield longest

            # Удалить пары, где конечная вершина совпадает, а начальная лежит на пути
            # Так как кратчайший путь наиболее вероятен при маршрутизации, мы считаем,
            # что нейронки будут получать эмбеддинги, для которых устойчивость уже
            # доказана

            cond = lambda e: e[-1] != longest[-1] or e[0] not in longest

            # Для нас доставка на разделителях, которые ведут в другой сток
            # всегда правильная
            for path in pathes:
                if not cond(path):
                    print('AAAAAAAAAAAAAAAAAAAAAAAAA')
                    print(path[0], path[-1])

            pathes = list(filter(cond, pathes))


    import time
    grr = Network(gr)

    log = []

    counter = 0
    for pair in get_pair(grr.graph):

        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        source, sink = (pair[0], pair[-1])
        print('Pair')
        print(source, sink)



    sink_pairs

    row = [(source, sink)]

    mc = AbsorbingMC(network, sink, is_spc, is_verbose)

    for source:
        edt_solver = EDTSolver(mc, onode, network)
        if not edt_solver.nontriv_dvtrs:
            print('')
            continue

        exit()

        adv = PGDAdversary(edt_solver, rho=eps,
                           steps=100,
                           step_size=0.02,
                           random_start=True,
                           stop_loss=100,
                           verbose=2,
                           norm="l_inf",
                           n_repeat=10,
                           repeat_mode="any",
                           dtype=torch.float64)

        best_embedding = adv.perturb(embedding_packer.initial_vector(), get_gradient)
        k = [0.95, 0.99, 1.01. 1.05]
        _, t_top, aux_info = get_gradient(best_embedding)

        for t_b in ks:
            t_b = t_top * k
            t = time.process_time()
            #print('COST')
            #print(const_bound)
            result = verifier.verify_delivery_cost_bound(
                    source, sink,
                    ma,
                    eps, const_bound)
            #print(result)
            elapsed_time = time.process_time() - t
            #print('ELAPSED')
            #print(elapsed_time)

#                $o_1 \to i_2$ & 83.84 & 55.71 & +   & 1.60 \\              |~
#                              &       & 50.41 &  -  & 0.13 ⏎ 

        counter = counter + 1

    print(counter)

