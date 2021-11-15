import networkx as nx
from .markov_analyzer import MarkovAnalyzer
from .router_graph import RouterGraph

from dqnroute.verification.embedding_packer import EmbeddingPacker
from dqnroute.verification.adversarial import PGDAdversary
import torch

from typing import *
from dqnroute.verification.ml_util import Util


def get_pair(graph):
    # Найти все упорядоченные пары вершин
    # Порядок важен, так как он влияет на входные данные
    nodes = list(graph.nodes)
    #pairs = [(a, b) for a in nodes for b in nodes if a != b]

    # Оставляем старые стоки
    pairs = [(a, b) for a in nodes for b in nodes if a != b 
                                                  #and b[0] == 'sink'
                                                  #and a[0] == 'source'
    ]

    # Найти кратчайшие пути между ними
    # Кратчайший путь наиболее вероятен при маршрутизации


    # Должен быть не один путь между парой вершни, чтобы 
    # Анализатор обрезает все и вероятности не нужны
    pathes = []
    for pair in pairs:
        try:
            path = nx.dijkstra_path(graph, *pair, weight='_')

            #if len(path) > 2:
            #    pathes.append(path)

            ok = False
            for node in path:
                if 'diverter' in node[0]:
                    ok = True

            if ok:
                pathes.append(path)

            #print(path)
        except nx.NetworkXNoPath:
            pass

    pathes.sort(key=len)
    # Выбрать самый длинный путь

    # Должен присутствовать хотя бы одни разделитель

    while len(pathes) != 0:
        #print(pathes)
        longest = pathes.pop()
        #print(pathes)

        #print(longest)
        yield longest

        # Удалить пары, где конечная вершина совпадает, а начальная лежит на пути
        # Так как кратчайший путь наиболее вероятен при маршрутизации, мы считаем,
        # что нейронки будут получать эмбеддинги, для которых устойчивость уже
        # доказана

        cond = lambda e: e[-1] != longest[-1] or e[0] not in longest
        pathes = list(filter(cond, pathes))


def full_embedding_adversarial_verification(verifier, gr, spc, ma_ver, eps,
        const_bound, soft, smooth):
    grr = RouterGraph(gr)

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
        print(source, sink)

        ma = MarkovAnalyzer(grr.graph, sink, spc, ma_ver)
        if len(ma.params) == 0:
            print('NOT PARAMS')
            continue

    # Нужно как-то соблюсти условие поглощения

        adv = PGDAdversary(rho=eps,
                           steps=100,
                           step_size=0.02,
                           random_start=True,
                           stop_loss=100,
                           verbose=2,
                           norm="l_inf",
                           n_repeat=10,
                           repeat_mode="any",
                           dtype=torch.float64)

        sink_embedding = grr.node_to_embeddings(sink, sink)[0]
        print(sink_embedding)

        embedding_packer = EmbeddingPacker(grr, sink, sink_embedding,
                list(ma.chain))
        _, lambdified_objective = ma.get_edt_sol(source)
        print(_)
        print(lambdified_objective)


        def get_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, float, str]:
            x = Util.optimizable_clone(x.flatten())
            objective_value, objective_inputs = \
                    embedding_packer.compute_objective(
                embedding_packer.unpack(x),
                ma.nontrivial_diverters,
                lambdified_objective,
                soft, smooth)
            print(objective_value)
            objective_value.backward()
            aux_info = ", ".join([f"{param}={value.detach().cpu().item():.4f}"
                                  for param, value in zip(ma.params, objective_inputs)])
            return x.grad, objective_value.item(), f"[{aux_info}]"


        best_embedding = adv.perturb(embedding_packer.initial_vector(), get_gradient)
        _, objective, aux_info = get_gradient(best_embedding)
        print('COST')
        print(objective)
        const_bound = objective * 1.05
        print(const_bound)


        result = verifier.verify_delivery_cost_bound(
                *pair,
                ma,
                eps, const_bound)
        counter = counter + 1

    print(counter)

