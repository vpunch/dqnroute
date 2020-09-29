import os
import argparse
import yaml

from pathlib import Path
from tqdm import tqdm
from typing import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sympy

from dqnroute import *
from dqnroute.networks import *
from dqnroute.verification.router_graph import RouterGraph
from dqnroute.verification.adversarial import PGDAdversary
from dqnroute.verification.ml_util import Util
from dqnroute.verification.markov_analyzer import MarkovAnalyzer
from dqnroute.verification.symbolic_analyzer import SymbolicAnalyzer
from dqnroute.verification.nnet_verifier import NNetVerifier
from dqnroute.verification.embedding_packer import EmbeddingPacker

NETWORK_FILENAME = "../network.nnet"
PROPERTY_FILENAME = "../property.txt"

parser = argparse.ArgumentParser(description="Verifier of baggage routing neural networks.")
parser.add_argument("--command", type=str, required=True,
                    help="one of deterministic_test, embedding_adversarial_search, embedding_adversarial_verification, q_adversarial, q_adversarial_lipschitz, compare")
parser.add_argument("--config_file", type=str, required=True,
                    help="YAML config file with the topology graph and other configuration info")
parser.add_argument("--probability_smoothing", type=float, default=0.01,
                    help="smoothing (0..1) of probabilities during learning and verification (defaut: 0.01)")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training (default: 42)")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")
parser.add_argument("--simple_path_cost", action="store_true",
                    help="use the number of transitions instead of the total conveyor length as path cost")
parser.add_argument("--skip_graphviz", action="store_true",
                    help="do not visualize graphs")
parser.add_argument("--softmax_temperature", type=float, default=1.5,
                    help="custom softmax temperature (higher temperature means larger entropy in routing decisions; default: 1.5)")
parser.add_argument("--cost_bound", type=float, default=100.0,
                    help="upper bound on delivery cost to verify (default: 100)")
parser.add_argument("--verification_lr", type=float, default=0.001,
                    help="learning rate in learning step verification (default: 0.001)")
parser.add_argument("--input_max_delta_q", type=float, default=10.0,
                    help="maximum ΔQ in learning step verification (default: 10.0)")

# commands for verification with Marabou
parser.add_argument("--marabou_path", type=str, default=None,
                    help="path to the Marabou executable (for command embedding_adversarial_verification)")
parser.add_argument("--input_eps_l_inf", type=float, default=0.1,
                    help="maximum L-infty not discrepancy of input embeddings in adversarial robustness verification (default: 0.1)")
parser.add_argument("--output_max_delta_q", type=float, default=10.0,
                    help="maximum ΔQ in adversarial robustness verification (default: 10.0)")
parser.add_argument("--output_max_delta_p", type=float, default=0.1,
                    help="maximum Δp in adversarial robustness verification (default: 0.1)")

parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="pretrain_num_episodes (default: 10000)")

args = parser.parse_args()

for dirname in ["../logs", "../img"]:
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

os.environ["IGOR_OVERRDIDDED_SOFTMAX_TEMPERATURE"] = str(args.softmax_temperature)

# 1. load scenario
scenario = args.config_file
print(f"Scenario: {scenario}")

with open(scenario) as file:
    scenario_loaded = yaml.load(file, Loader=yaml.FullLoader)

emb_dim = scenario_loaded["settings"]["router"]["dqn_emb"]["embedding"]["dim"]

# graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
lengths = [len(scenario_loaded["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
    + [len([c for c in scenario_loaded["configuration"]["conveyors"].values()
           if c["upstream"]["type"] == "conveyor"])]
#print(lengths)
graph_size = sum(lengths)
print(f"Embedding dimension: {emb_dim}, graph size: {graph_size}")


# 2. pretrain

def pretrain(args, dir_with_models: str, pretrain_filename: str):
    """ ALMOST COPIED FROM THE PRETRAINING NOTEBOOK """
    
    def gen_episodes_progress(num_episodes, **kwargs):
        with tqdm(total=num_episodes) as bar:
            return gen_episodes(bar=bar, num_episodes=num_episodes, **kwargs)
    
    class CachedEmbedding(Embedding):
        def __init__(self, InnerEmbedding, dim, **kwargs):
            self.dim = dim
            self.InnerEmbedding = InnerEmbedding
            self.inner_kwargs = kwargs
            self.fit_embeddings = {}

        def fit(self, graph, **kwargs):
            h = hash_graph(graph)
            if h not in self.fit_embeddings:
                embed = self.InnerEmbedding(dim=self.dim, **self.inner_kwargs)
                embed.fit(graph, **kwargs)
                self.fit_embeddings[h] = embed

        def transform(self, graph, idx):
            h = hash_graph(graph)
            return self.fit_embeddings[h].transform(idx)
    
    def shuffle(df):
        return df.reindex(np.random.permutation(df.index))
    
    def hash_graph(graph):
        if type(graph) != np.ndarray:
            graph = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes))
        m = hashlib.sha256()
        m.update(graph.tobytes())
        return base64.b64encode(m.digest()).decode('utf-8')
    
    def add_inp_cols(tag, dim):
        return mk_num_list(tag + '_', dim) if dim > 1 else tag

    def qnetwork_batches(net, data, batch_size=64, embedding=None):
        n = net.graph_size
        data_cols = []
        amatrix_cols = get_amatrix_cols(n)
        for tag, dim in net.add_inputs:
            data_cols.append(amatrix_cols if tag == 'amatrix' else add_inp_cols(tag, dim))
        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch['addr'].values
            dst = batch['dst'].values
            nbr = batch['neighbour'].values
            if embedding is not None:
                amatrices = batch[amatrix_cols].values
                new_btch = []
                for addr_, dst_, nbr_, A in zip(addr, dst, nbr, amatrices):
                    A = A.reshape(n, n)
                    embedding.fit(A)
                    new_addr = embedding.transform(A, int(addr_))
                    new_dst = embedding.transform(A, int(dst_))
                    new_nbr = embedding.transform(A, int(nbr_))
                    new_btch.append((new_addr, new_dst, new_nbr))
                [addr, dst, nbr] = stack_batch(new_btch)
            addr_inp = torch.tensor(addr, dtype=torch.float)
            dst_inp = torch.tensor(dst, dtype=torch.float)
            nbr_inp = torch.tensor(nbr, dtype=torch.float)
            inputs = tuple(torch.tensor(batch[cols].values, dtype=torch.float) for cols in data_cols)
            output = torch.tensor(batch['predict'].values, dtype=torch.float)
            yield (addr_inp, dst_inp, nbr_inp) + inputs, output

    def qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
        loss_func = nn.MSELoss()
        for batch, target in qnetwork_batches(net, data, **kwargs):
            optimizer.zero_grad()
            output = net(*batch)
            loss = loss_func(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            yield float(loss)

    def qnetwork_pretrain(net, data, optimizer='rmsprop', epochs=1, save_net=True, **kwargs):
        optimizer = get_optimizer(optimizer)(net.parameters())
        epochs_losses = []
        for i in tqdm(range(epochs)):
            sum_loss = 0
            loss_cnt = 0
            for loss in tqdm(qnetwork_pretrain_epoch(net, optimizer, data, **kwargs), desc=f'epoch {i}'):
                sum_loss += loss
                loss_cnt += 1
            epochs_losses.append(sum_loss / loss_cnt)
        if save_net:
            # label changed by Igor:
            net._label = pretrain_filename
            net.save()
        return epochs_losses
    
    data_conv = gen_episodes_progress(ignore_saved=True, context='conveyors',
                                      num_episodes=args.pretrain_num_episodes,
                                      random_seed=args.random_seed, run_params=scenario,
                                      save_path='../logs/data_conveyor1_oneinp_new.csv')
    data_conv.loc[:, 'working'] = 1.0
    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)
    args = {'scope': dir_with_models, 'activation': 'relu', 'layers': [64, 64], 'embedding_dim': conv_emb.dim}
    conveyor_network_ng_emb = QNetwork(graph_size, **args)
    conveyor_network_ng_emb_ws = QNetwork(graph_size, additional_inputs=[{'tag': 'working', 'dim': 1}], **args)
    conveyor_network_ng_emb_losses = qnetwork_pretrain(conveyor_network_ng_emb, shuffle(data_conv), epochs=10,
                                                       embedding=conv_emb, save_net=True)
    #conveyor_network_ng_emb_ws_losses = qnetwork_pretrain(conveyor_network_ng_emb_ws, shuffle(data_conv), epochs=20,
    #                                                      embedding=conv_emb, save_net=False)

dir_with_models = "conveyor_test_ng"
filename_suffix = f"_{emb_dim}_{graph_size}_{os.path.split(scenario)[1]}.bin"
pretrain_filename = f"igor_pretrained{filename_suffix}"
pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / pretrain_filename
if args.force_pretrain or not pretrain_path.exists():
    print(f"Pretraining {pretrain_path}...")
    pretrain(args, dir_with_models, pretrain_filename)
else:
    print(f"Using the already pretrained model {pretrain_path}...")


# 3. train

# FIXME?? random seed does not work in tranining!

def run_single(file: str, router_type: str, random_seed: int, **kwargs):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=file, router_type=router_type, random_seed=random_seed,
                                 progress_queue=queue, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner

def train(args, dir_with_models: str, pretrain_filename: str, train_filename: str,
          router_type: str, retrain: bool, work_with_files: bool):
    # Igor: I did not see an easy way to change the code in a clean way
    os.environ["IGOR_OVERRIDDEN_DQN_LOAD_FILENAME"] = pretrain_filename
    os.environ["IGOR_TRAIN_PROBABILITY_SMOOTHING"] = str(args.probability_smoothing)
    
    if retrain:
        if "IGOR_OMIT_TRAINING" in os.environ:
            del os.environ["IGOR_OMIT_TRAINING"]
    else:
        os.environ["IGOR_OMIT_TRAINING"] = "True"
    
    event_series, runner = run_single(file=scenario, router_type=router_type, progress_step=500,
                                      ignore_saved=[True], random_seed=args.random_seed)
    
    if router_type == "dqn_emb":
        world = runner.world
        net = next(iter(next(iter(world.handlers.values())).routers.values())).brain
        net._label = train_filename    
        # save or load the trained network
        if work_with_files:
            if retrain:
                net.save()
            else:
                net.restore()
    else:
        world = None
    
    return event_series, world
    

train_filename = f"igor_trained{filename_suffix}"
train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename
retrain = args.force_train or not train_path.exists() or args.command == "compare"
if retrain:
    print(f"Training {train_path}...")
else:
    print(f"Using the already trained model {train_path}...")
    
dqn_log, world = train(args, dir_with_models, pretrain_filename, train_filename, "dqn_emb", retrain, True)


# 4. load the router graph
g = RouterGraph(world)
print("Reachability matrix:")
g.print_reachability_matrix()

def visualize(g: RouterGraph):
    gv_graph = g.to_graphviz()
    prefix = f"../img/topology_graph{filename_suffix}."
    gv_graph.write(prefix + "gv")
    for prog in ["dot", "circo", "twopi"]:
        prog_prefix = f"{prefix}{prog}."
        for fmt in ["pdf", "png"]:
            path = f"{prog_prefix}{fmt}"
            print(f"Drawing {path} ...")
            gv_graph.draw(path, prog=prog, args="-Gdpi=300 -Gmargin=0 -Grankdir=LR")

if not args.skip_graphviz:
    visualize(g)

def transform_embeddings(sink_embedding    : torch.Tensor,
                         current_embedding : torch.Tensor,
                         neighbor_embedding: torch.Tensor) -> torch.Tensor:
    return torch.cat((sink_embedding     - current_embedding,
                      neighbor_embedding - current_embedding), dim=1)

def get_symbolic_analyzer() -> SymbolicAnalyzer:
    return SymbolicAnalyzer(g, args.softmax_temperature, args.probability_smoothing,
                            args.verification_lr, delta_q_max=args.input_max_delta_q)

def get_nnet_verifier() -> NNetVerifier:
    assert args.marabou_path is not None, (
        "It is mandatory to specify --verification_marabou_path for command "
        "embedding_adversarial_verification.")
    return NNetVerifier(g, args.marabou_path, NETWORK_FILENAME, PROPERTY_FILENAME,
                        args.probability_smoothing, args.softmax_temperature, emb_dim)


print(f"Running command {args.command}...")
if args.command == "deterministic_test":
    for source in g.sources:
        for sink in g.sinks:
            print(f"Testing delivery from {source} to {sink}...")
            current_node = source
            visited_nodes = set()
            sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
            while True:
                if current_node in visited_nodes:
                    print("    FAIL due to cycle")
                    break
                visited_nodes.add(current_node)
                print("    in:", current_node)
                if current_node[0] == "sink":
                    print("    " + ("OK" if current_node == sink else "FAIL due to wrong destination"))
                    break
                elif current_node[0] in ["source", "junction"]:
                    out_nodes = g.get_out_nodes(current_node)
                    assert len(out_nodes) == 1
                    current_node = out_nodes[0]
                elif current_node[0] == "diverter":
                    current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(current_node,
                                                                                             sink)
                    q_values = []
                    for neighbor, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                        with torch.no_grad():
                            q = g.q_forward(current_embedding, sink_embedding, neighbor_embedding).item()
                        print(f"        Q({current_node} -> {neighbor} | sink = {sink}) = {q:.4f}")
                        q_values += [q]
                    best_neighbor_index = np.argmax(np.array(q_values))
                    current_node = neighbors[best_neighbor_index]
                else:
                    raise AssertionError()
elif args.command == "embedding_adversarial_search":
    adv = PGDAdversary(rho=1.5, steps=100, step_size=0.02, random_start=True, stop_loss=1e5, verbose=2,
                       norm="scaled_l_2", n_repeat=2, repeat_mode="min", dtype=torch.float64)
    for sink in g.sinks:
        print(f"Measuring adversarial robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        
        # gather all embeddings that we need to compute the objective
        embedding_packer = EmbeddingPacker(g, sink, sink_embedding, ma.reachable_nodes)
        initial_vector = embedding_packer.initial_vector()

        for source in ma.reachable_sources:
            print(f"  Measuring adversarial robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)

            def get_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, float, str]:
                """
                :param x: parameter vector (the one expected to converge to an adversarial example)
                :return: a tuple (gradient pointing to the direction of the adversarial attack,
                                  the corresponding loss function value,
                                  auxiliary information for printing during optimization).
                """
                x = Util.optimizable_clone(x.flatten())
                embedding_dict = embedding_packer.unpack(x)
                objective_inputs = []
                perturbed_sink_embeddings = embedding_dict[sink].repeat(2, 1)
                for diverter in ma.nontrivial_diverters:
                    perturbed_diverter_embeddings = embedding_dict[diverter].repeat(2, 1)
                    _, current_neighbors, _ = g.node_to_embeddings(diverter, sink)
                    perturbed_neighbor_embeddings = torch.cat([embedding_dict[current_neighbor]
                                                               for current_neighbor in current_neighbors])
                    q_values = g.q_forward(perturbed_diverter_embeddings, perturbed_sink_embeddings,
                                           perturbed_neighbor_embeddings).flatten()
                    objective_inputs += [Util.q_values_to_first_probability(q_values,
                                                                            args.softmax_temperature,
                                                                            args.probability_smoothing)]
                objective_value = lambdified_objective(*objective_inputs)
                #print(Util.to_numpy(objective_value))
                objective_value.backward()
                aux_info = ", ".join([f"{param}={value.detach().cpu().item():.4f}"
                                      for param, value in zip(ma.params, objective_inputs)])
                return x.grad, objective_value.item(), f"[{aux_info}]"
            adv.perturb(initial_vector, get_gradient)
elif args.command == "embedding_adversarial_full_verification":
    nv = get_nnet_verifier()
    for sink in g.sinks:
        print(f"Verifying adversarial robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        for source in ma.reachable_sources:
            print(f"  Verifying adversarial robustness of delivery from {source} to {sink}...")
            result = nv.verify_cost_delivery_bound(sink, source, ma, args.input_eps_l_inf, args.cost_bound)
            print(f"    {result}")
elif args.command == "embedding_adversarial_verification":
    nv = get_nnet_verifier()
    for sink in g.sinks:
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost, verbose=False)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        
        # for each node from which the sink is reachable, verify q value stability
        for node in ma.reachable_nodes:
            if node[0] == "sink":
                # sinks do not have any neighbors
                continue
            print(f"Verifying Q value stability for node={node} and sink={sink}...")
            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node, sink)
            # for each neighbor
            for neighbor, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                emb_center = transform_embeddings(sink_embedding, current_embedding, neighbor_embedding)
                with torch.no_grad():
                    actual_output = nv.net(emb_center).item()
                ROUND_DIGITS = 3
                print(f"  Q on real embedding: NN({Util.list_round(emb_center.flatten(), ROUND_DIGITS)})"
                      f" = {round(actual_output, ROUND_DIGITS)}")
    
                # two verification queries: 
                # check whether the output can be less than the bound,
                # then check whether it can be greater
                result = nv.verify_adv_robustness(
                    nv.net, [nv.A, nv.B, nv.C], [nv.a, nv.b, nv.c],
                    emb_center.flatten(), args.input_eps_l_inf,
                    [f"y0 <= {actual_output - args.output_max_delta_q}",
                     f"y0 >= {actual_output + args.output_max_delta_q}"],
                    check_or=True
                )
                print(f"    Verification result: {result}")
            
        # for each non-trivial diverter, verify the stability of routing probability
        for diverter in ma.nontrivial_diverters:
            print(f"Verifying the stability of routing probability for node={diverter} and sink={sink}...")
            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(diverter, sink)
            # the first halves of these vectors are equal:
            embs_center = [transform_embeddings(sink_embedding, current_embedding, neighbor_embedding)
                           for neighbor_embedding in neighbor_embeddings]
            #print(embs_center[1] - embs_center[0])
            emb_center = torch.cat((embs_center[0], embs_center[1][:, emb_dim:]), dim=1)
            
            # get current probability
            q_values = g.q_forward(current_embedding, sink_embedding.repeat(2, 1),
                                   torch.cat(neighbor_embeddings)).flatten()
            p = Util.q_values_to_first_probability(q_values, args.softmax_temperature,
                                                   args.probability_smoothing).item()
            q_diff_min = nv.probability_to_q_diff(p - args.output_max_delta_p)
            q_diff_max = nv.probability_to_q_diff(p + args.output_max_delta_p)
            
            print(f"  Q values: {Util.to_numpy(q_values)}")
            print(f"  p on real embedding: {p}")
            print(f"  Checking whether p is ins [{p - args.output_max_delta_p}, {p + args.output_max_delta_p}].")
            print(f"  Checking whether the difference of Qs of two neighbors is in"
                  f" [{q_diff_min}, {q_diff_max}].")
            cases_to_check = ([f"+y0 -y1 <= {q_diff_min}"] if q_diff_min != -np.infty else []) \
                           + ([f"+y0 -y1 >= {q_diff_max}"] if q_diff_max !=  np.infty else [])
            print(f"  cases to check: {cases_to_check}")
            
            result = nv.verify_adv_robustness(
                nv.net_new, [nv.A_new, nv.B_new, nv.C_new], [nv.a_new, nv.b_new, nv.c_new],
                emb_center.flatten(), args.input_eps_l_inf, cases_to_check, check_or=True
            )
            print(f"  Verification result: {result}")
elif args.command == "q_adversarial":
    sa = get_symbolic_analyzer()
    plot_index = 0
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        sink_embeddings = sink_embedding.repeat(2, 1)

        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)

            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    # compute
                    # we assume a linear change of parameters
                    reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                       neighbor_embedding).flatten().item()
                    NO_POINTS = 351
                    actual_qs = reference_q + np.linspace(-sa.delta_q_max, sa.delta_q_max, NO_POINTS)
                    kappa = sa.get_transformed_cost(ma, objective, args.cost_bound)
                    lambdified_kappa = sympy.lambdify(ma.params, kappa)
                    objective_values = torch.empty(len(actual_qs))
                    kappa_values = torch.empty(len(actual_qs))
                    for i, actual_q in enumerate(actual_qs):
                        ps = sa.compute_ps(ma, diverter, sink, sink_embeddings, reference_q, actual_q)
                        objective_values[i] = lambdified_objective(*ps)
                        kappa_values[i] = lambdified_kappa(*ps)

                    # plot
                    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
                    plt.subplots_adjust(hspace=0.3)
                    caption_starts = ("Delivery cost (τ)", "Transformed delivery cost (κ)")
                    axes[0].set_yscale("log")
                    for ax, caption_start, values in zip(axes, caption_starts,
                                                         (objective_values, kappa_values)):
                        label = (f"{caption_start} from {source} to {sink} when making optimization"
                                 f" step with current={node_key}, neighbor={neighbor_key}")
                        print(f"{label}...")
                        ax.set_title(label)
                        ax.plot(actual_qs, values)
                        gap = values.max() - values.min()
                        y_delta = 0 if gap > 0 else 5
                        # show the zero step value:
                        ax.vlines(reference_q, min(values) - y_delta, max(values) + y_delta)
                        ax.hlines(values[len(values) // 2], min(actual_qs), max(actual_qs))
                    # show the verification bound:
                    axes[0].hlines(args.cost_bound, min(actual_qs), max(actual_qs))
                    axes[1].hlines(0, min(actual_qs), max(actual_qs))
                    plt.savefig(f"../img/{filename_suffix}_{plot_index}.pdf")
                    plt.close()
                    print(f"Empirically found maximum of τ: {objective_values.max():.6f}")
                    print(f"Empirically found maximum of κ: {kappa_values.max():.6f}")
                    plot_index += 1
elif args.command == "q_adversarial_lipschitz":
    sa = get_symbolic_analyzer()
    print(sa.net)
    sa.load_matrices()
    
    for sink in g.sinks:
        print(f"Measuring robustness of delivery to {sink}...")
        ma = MarkovAnalyzer(g, sink, args.simple_path_cost)
        sink_embedding, _, _ = g.node_to_embeddings(sink, sink)
        sink_embeddings = sink_embedding.repeat(2, 1)
        
        ps_function_names = [f"p{i}" for i in range(len(ma.params))]
        function_ps = [sympy.Function(name) for name in ps_function_names]
        evaluated_function_ps = [f(sa.beta) for f in function_ps]
        
        # cached values
        computed_logits_and_derivatives: Dict[AgentId, Tuple[sympy.Expr, sympy.Expr]] = {}
        
        def compute_logit_and_derivative(sa: SymbolicAnalyzer, diverter_key: AgentId) -> Tuple[sympy.Expr, sympy.Expr]:
            if diverter_key not in computed_logits_and_derivatives:
                diverter_embedding, _, neighbor_embeddings = g.node_to_embeddings(diverter_key, sink)
                delta_e = [sa.tensor_to_sympy(transform_embeddings(sink_embedding,
                                                                   diverter_embedding,
                                                                   neighbor_embeddings[i]).T)
                           for i in range(2)]
                logit = sa.to_scalar(sa.sympy_q(delta_e[0]) -
                                     sa.sympy_q(delta_e[1])) / args.softmax_temperature
                dlogit_dbeta = logit.diff(sa.beta)
                computed_logits_and_derivatives[diverter_key] = logit, dlogit_dbeta
            else:
                print("      (using cached value)")
            return computed_logits_and_derivatives[diverter_key]
        
        for source in ma.reachable_sources:
            print(f"  Measuring robustness of delivery from {source} to {sink}...")
            objective, lambdified_objective = ma.get_objective(source)
        
            for node_key in g.node_keys:
                current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)

                for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
                    print(f"    Considering learning step {node_key} -> {neighbor_key}...")
                    reference_q = sa.compute_gradients(current_embedding, sink_embedding,
                                                       neighbor_embedding).flatten().item()
                    print(f"      Reference Q value = {reference_q:.4f}")
                    sa.load_grad_matrices()
                    
                    MOCK = False
                    if MOCK:
                        dim = 7
                        #print(A.shape, b.shape, C.shape, d.shape, E.shape, f.shape)
                        sa.A = sa.A[:dim, :];    sa.A_hat = sa.A_hat[:dim, :]
                        sa.b = sa.b[:dim, :];    sa.b_hat = sa.b_hat[:dim, :]
                        sa.C = sa.C[:dim, :dim]; sa.C_hat = sa.C_hat[:dim, :dim]
                        sa.d = sa.b[:dim, :];    sa.d_hat = sa.d_hat[:dim, :]
                        sa.E = sa.E[:,    :dim]; sa.E_hat = sa.E_hat[:,    :dim]

                    print(f"      τ(p) = {objective}, τ(p) < {args.cost_bound}?")
                    kappa_of_p = sa.get_transformed_cost(ma, objective, args.cost_bound)
                    lambdified_kappa = sympy.lambdify(ma.params, kappa_of_p)
                    print(f"      κ(p) = {kappa_of_p}, κ(p) < 0?")
                    kappa_of_beta = kappa_of_p.subs(list(zip(ma.params, evaluated_function_ps)))
                    print(f"      κ(β) = {kappa_of_beta}, κ(β) < 0?")
                    dkappa_dbeta = kappa_of_beta.diff(sa.beta)
                    print(f"      dκ(β)/dβ = {dkappa_dbeta}")
                    
                    #  compute a pool of bounds
                    derivative_bounds = {}
                    for param, diverter_key in zip(ma.params, ma.nontrivial_diverters):
                        _, current_neighbors, _ = g.node_to_embeddings(diverter_key, sink)
                        print(f"      Computing the logit and its derivative for {param} ="
                              f" P({diverter_key} -> {current_neighbors[0]} | sink = {sink})....")
                        logit, dlogit_dbeta = compute_logit_and_derivative(sa, diverter_key)
                        
                        # surprisingly, the strings are very slow to obtain
                        if False:
                            print(f"      logit = {sa.expr_to_string(logit)[:500]} ...")
                            print(f"      dlogit/dβ = {sa.expr_to_string(dlogit_dbeta)[:500]} ...")
                            
                        print(f"      Computing logit bounds...")
                        derivative_bounds[param.name] = sa.estimate_upper_bound(dlogit_dbeta)

                    print(f"      Computing the final upper bound on dκ(β)/dβ...")
                    top_level_bound = sa.estimate_top_level_upper_bound(dkappa_dbeta, ps_function_names,
                                                                        derivative_bounds)
                    print(f"      Final upper bound on the Lipschitz constant of κ(β): {top_level_bound}")
                    
                    def q_to_kappa(actual_q: float) -> float:
                        ps = sa.compute_ps(ma, diverter, sink, sink_embeddings, reference_q, actual_q)
                        return lambdified_kappa(*ps)
                    
                    def q_to_beta(actual_q: float) -> float:
                        return (actual_q - reference_q) * sa.lr
                    
                    empirical_bound = -np.infty
                    max_depth = 0
                    no_evaluations = 2
                    checked_q_measure = 0.0
                    
                    def prove_bound(left_q: float, right_q: float,
                                    left_kappa: float, right_kappa: float, depth: int) -> bool:
                        global empirical_bound, no_evaluations, max_depth, checked_q_measure
                        mid_q = (left_q + right_q) / 2
                        mid_kappa = q_to_kappa(mid_q)
                        actual_qs = np.array([left_q, mid_q, right_q])
                        kappa_values = np.array([left_kappa, mid_kappa, right_kappa])
                        worst_index = kappa_values.argmax()
                        max_kappa = kappa_values[worst_index]
                        # 1. try to find counterexample
                        if max_kappa > 0:
                            worst_q = actual_qs[worst_index]
                            worst_dq = worst_q - reference_q
                            print(f"        Counterexample found: q = {worst_q:.6f}, Δq = {worst_dq:.6f},"
                                  f" β = {q_to_beta(worst_q):.6f}, κ = {max_kappa:.6f}")
                            return False
                            
                        # 2. try to find proof on [left, right]
                        kappa_upper_bound = -np.infty
                        max_on_interval = np.empty(2)
                        for i, (q_interval, kappa_interval) in enumerate(zip(sa.to_intervals(actual_qs),
                                                                             sa.to_intervals(kappa_values))):
                            left_beta, right_beta = q_to_beta(q_interval[0]), q_to_beta(q_interval[1])
                            max_on_interval[i] = (top_level_bound * (right_beta - left_beta) +
                                                  sum(kappa_interval)) / 2
                        if max_on_interval.max() < 0:
                            checked_q_measure += right_q - left_q
                            return True
                        
                        # logging
                        no_evaluations += 1
                        max_depth = max(max_depth, depth)
                        empirical_bound = max(empirical_bound, max_kappa)
                        if no_evaluations % 100 == 0:
                            percentage = checked_q_measure / sa.delta_q_max / 2 * 100
                            print(f"      Status: {no_evaluations} evaluations, empirical bound is"
                                  f" {empirical_bound:.6f}, maximum depth is {max_depth}, checked Δq"
                                  f" percentage: {percentage:.2f}")
                        
                        # 3. otherwise, try recursively
                        calls = [(lambda: prove_bound(left_q, mid_q,  left_kappa, mid_kappa,   depth + 1)),
                                 (lambda: prove_bound(mid_q,  right_q, mid_kappa, right_kappa, depth + 1))]
                        # to produce counetrexamples faster,
                        # start from the most empirically dangerous subinterval
                        if max_on_interval.argmax() == 1:
                            calls = calls[::-1]
                        return calls[0]() and calls[1]()
                    
                    left_q = -sa.delta_q_max + reference_q
                    right_q = sa.delta_q_max + reference_q
                    if prove_bound(left_q, right_q, q_to_kappa(left_q), q_to_kappa(right_q), 0):
                        print("      Proof found!")
elif args.command == "compare":
    _legend_txt_replace = {
        'networks': {
            'link_state': 'Shortest paths', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'Centralized control'
        }, 'conveyors': {
            'link_state': 'Vyatkin-Black', 'simple_q': 'Q-routing', 'pred_q': 'PQ-routing',
            'glob_dyn': 'Global-dynamic', 'dqn': 'DQN', 'dqn_oneout': 'DQN (1-out)',
            'dqn_emb': 'DQN-LE', 'centralized_simple': 'BSR'
        }
    }
    _targets = {'time': 'avg', 'energy': 'sum', 'collisions': 'sum'}
    _ylabels = {
        'time': 'Mean delivery time', 'energy': 'Total energy consumption', 'collisions': 'Cargo collisions'
    }
    
    router_types = ["dqn_emb", "link_state", "simple_q"]
    # reuse the log for dqn_emb:
    series = [dqn_log.getSeries(add_avg=True)]
    for router_type in router_types[1:]:
        s, _ = train(args, dir_with_models, pretrain_filename, train_filename, router_type, True, False)
        series += [s.getSeries(add_avg=True)]
    
    dfs = []
    for router_type, s in zip(router_types, series):
        df = s.copy()
        add_cols(df, router_type=router_type, seed=args.random_seed)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    
    def print_sums(df):
        for tp in router_types:
            x = df.loc[df['router_type'] == tp, 'count'].sum()
            txt = _legend_txt_replace.get(tp, tp)
            print(f'  {txt}: {x}')
    
    def plot_data(data, meaning='time', figsize=(15,5), xlim=None, ylim=None,
              xlabel='Simulation time', ylabel=None, font_size=14, title=None, save_path=None,
              draw_collisions=False, context='networks', **kwargs):
        if 'time' not in data.columns:
            datas = split_dataframe(data, preserved_cols=['router_type', 'seed'])
            for tag, df in datas:
                if tag == 'collisions' and not draw_collisions:
                    print('Number of collisions:')
                    print_sums(df)
                    continue

                xlim = kwargs.get(tag+'_xlim', xlim)
                ylim = kwargs.get(tag+'_ylim', ylim)
                save_path = kwargs.get(tag+'_save_path', save_path)
                plot_data(df, meaning=tag, figsize=figsize, xlim=xlim, ylim=ylim,
                          xlabel=xlabel, ylabel=ylabel, font_size=font_size,
                          title=title, save_path=save_path, context='conveyors')
            return

        target = _targets[meaning]
        if ylabel is None:
            ylabel = _ylabels[meaning]

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x='time', y=target, hue='router_type', data=data, err_kws={'alpha': 0.1})

        handles, labels = ax.get_legend_handles_labels()
        new_labels = list(map(lambda l: _legend_txt_replace[context].get(l, l), labels[1:]))
        ax.legend(handles=handles[1:], labels=new_labels, fontsize=font_size)

        ax.tick_params(axis='both', which='both', labelsize=int(font_size*0.75))

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if save_path is not None:
            fig.savefig('../img/' + save_path, bbox_inches='tight')
    
    plot_data(dfs, figsize=(10, 8), font_size=22, energy_ylim=(7e6, 2.3e7),
              time_save_path='conveyors-break-1-time.pdf', energy_save_path='conveyors-break-1-energy.pdf')

else:
    raise RuntimeError(f"Unknown command {args.command}.")
