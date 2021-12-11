import os
import argparse
import yaml
import re

import hashlib
import base64

from pathlib import Path
from tqdm import tqdm
from typing import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import networkx as nx

from dqnroute.constants import TORCH_MODELS_DIR
from dqnroute.event_series import split_dataframe
from dqnroute.generator import gen_episodes
from dqnroute.networks.common import get_optimizer
from dqnroute.networks.embeddings import Embedding, LaplacianEigenmap
from dqnroute.networks.q_network import QNetwork
from dqnroute.networks.actor_critic_networks import PPOActor, PPOCritic
from dqnroute.simulation.common import mk_job_id, add_cols, DummyProgressbarQueue
from dqnroute.simulation.conveyors import ConveyorsRunner
from dqnroute.utils import AgentId, get_amatrix_cols, make_batches, stack_batch, mk_num_list

import dqnroute.verification.commands as verif_cmds
#from dqnroute.verification.symbolic_analyzer import SymbolicAnalyzer, LipschitzBoundComputer

parser = argparse.ArgumentParser(
    description="Script to train, simulate and verify deep neural networks for baggage routing.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# general parameters
parser.add_argument("config_files", type=str, nargs="+",
                    help="YAML config file(s) with the conveyor topology graph, input scenario and settings "
                         "of routing algorithms (all files will be concatenated into one)")
parser.add_argument("--routing_algorithms", type=str, default="dqn_emb,centralized_simple,link_state,simple_q",
                    help="comma-separated list of routing algorithms to run "
                         "(possible entries: dqn_emb, centralized_simple, link_state, simple_q, ppo_emb, random)")
parser.add_argument("--command", type=str, default="run",
                    help="possible options: run, compute_expected_cost, embedding_adversarial_search, "
                         "embedding_adversarial_verification, q_adversarial_search, q_adversarial_verification")
parser.add_argument("--random_seed", type=int, default=42,
                    help="random seed for pretraining and training")
parser.add_argument("--pretrain_num_episodes", type=int, default=10000,
                    help="number of episodes for supervised pretraining")
parser.add_argument("--pretrain_num_epochs", type=int, default=32,
                    help="number of episodes for supervised pretraining")
parser.add_argument("--force_pretrain", action="store_true",
                    help="whether not to load previously saved pretrained models and force recomputation")
parser.add_argument("--train_num_episodes", type=int, default=10000,
                    help="number of episodes for supervised pretraining")
parser.add_argument("--force_train", action="store_true",
                    help="whether not to load previously saved trained models and force recomputation")

verif_cmds.add_arguments(parser)

parser.add_argument("--learning_step_indices", type=str, default=None,
                    help="in learning step verification, consider only learning steps with these indices "
                         "comma-separated list without spaces; all steps will be considered if not specified)")

# parameters specific to learning step verification
# (q_adversarial_search, q_adversarial_verification)
parser.add_argument("--verification_lr", type=float, default=0.001,
                    help="learning rate in learning step verification")
parser.add_argument("--input_max_delta_q", type=float, default=10.0,
                    help="maximum ΔQ in learning step verification")
parser.add_argument("--q_adversarial_no_points", type=int, default=351,
                    help="number of points used to create plots in command q_adversarial")
parser.add_argument("--q_adversarial_verification_no_points", type=int, default=351,
                    help="number of points to search for counterexamples before estimating the Lipschitz "
                         "constant in command q_adversarial_lipschitz (setting to less than 2 disables "
                         "this search)")

args = parser.parse_args()

# dqn_emb = DQNroute-LE, centralized_simple = BSR
router_types_supported = 'dqn_emb ppo_emb centralized_simple link_state simple_q reinforce_emb'.split(' ')
router_types = args.routing_algorithms
assert len(router_types) > 0, '--routing_algorithms cannot be empty'
router_types = re.split(', *', args.routing_algorithms)
assert len(set(router_types) - set(router_types_supported)) == 0, \
    f'unsupported algorithm in --routing_algorithms was found; supported ones: {router_types_supported}'

dqn_emb_exists = 'dqn_emb' in router_types
ppo_emb_exists = 'ppo_emb' in router_types
reinforce_emb_exists = 'reinforce_emb' in router_types
nn_loading_needed = "dqn_emb" in router_types or args.command != "run"

random_seed = args.random_seed

# Create directories for logs and results
for dirname in ['../logs', '../img']:
    os.makedirs(dirname, exist_ok=True)

# 1. load scenario from one or more config files
string_scenario, filename_suffix = [], []
for config_filename in args.config_files:
    filename_suffix += [os.path.split(config_filename)[1].replace(".yaml", "")]
    with open(config_filename, "r") as f:
        string_scenario += f.readlines()
string_scenario = "".join(string_scenario)
scenario = yaml.safe_load(string_scenario)
print(f"Configuration files: {args.config_files}")

router_settings = scenario["settings"]["router"]
emb_dim = router_settings["embedding"]["dim"]
softmax_temperature = router_settings["dqn"]["softmax_temperature"]
probability_smoothing = router_settings["dqn"]["probability_smoothing"]

# graphs size = #sources + #diverters + #sinks + #(conveyors leading to other conveyors)
lengths = [len(scenario["configuration"][x]) for x in ["sources", "diverters", "sinks"]] \
          + [len([c for c in scenario["configuration"]["conveyors"].values()
                  if c["upstream"]["type"] == "conveyor"])]
graph_size = sum(lengths)
filename_suffix = "__".join(filename_suffix)
filename_suffix = f"_{emb_dim}_{graph_size}_{filename_suffix}.bin"
print(f"Embedding dimension: {emb_dim}, graph size: {graph_size}")

# pretrain common params and function
pretrain_data_size = args.pretrain_num_episodes
pretrain_epochs_num = args.pretrain_num_epochs
force_pretrain = args.force_pretrain


def gen_episodes_progress(router_type, num_episodes, **kwargs):
    with tqdm(total=num_episodes) as bar:
        return gen_episodes(router_type, bar=bar, num_episodes=num_episodes, **kwargs)


class CachedEmbedding(Embedding):
    def __init__(self, InnerEmbedding, dim, **kwargs):
        super().__init__(dim, **kwargs)

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


def hash_graph(graph):
    if type(graph) != np.ndarray:
        graph = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes))
    m = hashlib.sha256()
    m.update(graph.tobytes())
    return base64.b64encode(m.digest()).decode("utf-8")


def add_inp_cols(tag, dim):
    return mk_num_list(tag + "_", dim) if dim > 1 else tag


# train common params and function
train_data_size = args.train_num_episodes
force_train = args.force_train


# TODO check whether setting a random seed makes training deterministic
def run_single(
        run_params: dict,
        router_type: str,
        random_seed: int,
        **kwargs
):
    job_id = mk_job_id(router_type, random_seed)
    with tqdm(desc=job_id) as bar:
        queue = DummyProgressbarQueue(bar)
        runner = ConveyorsRunner(run_params=run_params, router_type=router_type, random_seed=random_seed,
                                 progress_queue=queue, omit_training=False, **kwargs)
        event_series = runner.run(**kwargs)
    return event_series, runner


# DQN part (pre-train + train)
def pretrain_dqn(
        generated_data_size: int,
        num_epochs: int,
        dir_with_models: str,
        pretrain_filename: str = None,
        pretrain_dataset_filename: str = None,
        use_full_topology: bool = True,
):
    def qnetwork_batches(net, data, batch_size=64, embedding=None):
        n = graph_size
        data_cols = []
        amatrix_cols = get_amatrix_cols(n)
        for tag, dim in net.add_inputs:
            data_cols.append(amatrix_cols if tag == "amatrix" else add_inp_cols(tag, dim))
        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch["addr"].values
            dst = batch["dst"].values
            nbr = batch["neighbour"].values
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
            addr_inp = torch.FloatTensor(addr)
            dst_inp = torch.FloatTensor(dst)
            nbr_inp = torch.FloatTensor(nbr)
            inputs = tuple(torch.FloatTensor(batch[cols].values) for cols in data_cols)
            output = torch.FloatTensor(batch["predict"].values)
            yield (addr_inp, dst_inp, nbr_inp) + inputs, output

    def qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
        loss_func = torch.nn.MSELoss()
        for batch, target in qnetwork_batches(net, data, **kwargs):
            optimizer.zero_grad()
            output = net(*batch)
            loss = loss_func(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            yield float(loss)

    def qnetwork_pretrain(net, data, optimizer="rmsprop", **kwargs):
        optimizer = get_optimizer(optimizer)(net.parameters())
        epochs_losses = []
        for _ in tqdm(range(num_epochs), desc='DQN Pretraining...'):
            sum_loss = 0
            loss_cnt = 0
            for loss in qnetwork_pretrain_epoch(net, optimizer, data, **kwargs):
                sum_loss += loss
                loss_cnt += 1
            epochs_losses.append(sum_loss / loss_cnt)
        if pretrain_filename is not None:
            # label changed by Igor:
            net.change_label(pretrain_filename)
            # net._label = pretrain_filename
            net.save()
        return epochs_losses

    data_conv = gen_episodes_progress(
        'dqn_oneout',  # TODO fix it
        generated_data_size,
        ignore_saved=True,
        context="conveyors",
        random_seed=random_seed,
        run_params=scenario,
        save_path=pretrain_dataset_filename,
        use_full_topology=use_full_topology
    )
    data_conv.loc[:, "working"] = 1.0
    shuffled_data = data_conv.sample(frac=1)

    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)

    network_args = {
        'scope': dir_with_models,
        'activation': router_settings['dqn']['activation'],
        'layers': router_settings['dqn']['layers'],
        'embedding_dim': emb_dim,
    }
    conveyor_network_ng_emb = QNetwork(graph_size, **network_args)

    conveyor_network_ng_emb_losses = qnetwork_pretrain(
        conveyor_network_ng_emb,
        shuffled_data,
        embedding=conv_emb
    )

    return conveyor_network_ng_emb_losses


def train_dqn(
        progress_step: int,
        router_type: str,
        dir_with_models: str,
        pretrain_filename: str,
        train_filename: str,
        random_seed: int,
        work_with_files: bool,
        retrain: bool,
        use_reinforce: bool = True,
        use_combined_model: bool = False
):
    scenario["settings"]["router"][router_type]["use_reinforce"] = use_reinforce
    scenario["settings"]["router"][router_type]["use_combined_model"] = use_combined_model
    scenario["settings"]["router"][router_type]["scope"] = dir_with_models
    scenario["settings"]["router"][router_type]["load_filename"] = pretrain_filename

    if retrain:
        # TODO get rid of this environmental variable
        if "OMIT_TRAINING" in os.environ:
            del os.environ["OMIT_TRAINING"]
    else:
        os.environ["OMIT_TRAINING"] = "True"

    event_series, runner = run_single(
        run_params=scenario,
        router_type=router_type,
        progress_step=progress_step,
        ignore_saved=[True],
        random_seed=random_seed
    )

    world = runner.world

    some_router = next(iter(next(iter(world.handlers.values())).routers.values()))

    net = some_router.brain
    net.change_label(train_filename)

    # save or load the trained network
    if work_with_files:
        if retrain:
            if some_router.use_single_neural_network:
                net.save()
            else:
                print(
                    "Warning: saving/loading models trained in simulation is only implemented "
                    "when use_single_neural_network = True. The models were not saved to disk."
                )
        else:
            net.restore()

    return event_series, world


def dqn_experiments(
        n: int,
        use_combined_model: bool = True,
        use_full_topology: bool = True,
        use_reinforce: bool = True,
        process_pretrain: bool = True,
        process_train: bool = True
):
    dqn_logs = []
    dqn_worlds = []

    for _ in range(n):
        if process_pretrain:
            print('Pretraining DQN Models...')
            dqn_losses = pretrain_dqn(
                pretrain_data_size,
                pretrain_epochs_num,
                dir_with_models,
                pretrain_filename,
                data_path,
                use_full_topology=use_full_topology,
            )
        else:
            print(f'Using the already pretrained model...')

        if process_train:
            print('Training DQN Model...')
            dqn_log, dqn_world = train_dqn(
                train_data_size,
                'dqn_emb',
                dir_with_models,
                pretrain_filename,
                train_filename,
                random_seed,
                True,
                True,
                use_reinforce=use_reinforce,
                use_combined_model=use_combined_model
            )
        else:
            print('Skip training process...')

        dqn_logs.append(dqn_log.getSeries(add_avg=True))
        dqn_worlds.append(dqn_world)

    return dqn_logs, dqn_worlds


# whole pipeline
if dqn_emb_exists:
    dqn_serieses = []

    dqn_emp_config = scenario['settings']['router']['dqn_emb']

    dir_with_models = 'conveyor_models_dqn'

    pretrain_filename = f'pretrained{filename_suffix}'
    pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / pretrain_filename

    data_filename = f'pretrain_data_ppo{filename_suffix}'
    data_path = f'../logs/{data_filename}'

    train_filename = f'trained{filename_suffix}'
    train_path = Path(TORCH_MODELS_DIR) / dir_with_models / train_filename

    do_pretrain = force_pretrain or not pretrain_path.exists() or True
    do_train = force_train or not train_path.exists() or args.command == 'run' or True

    print(f'Model: {pretrain_path}')

    # dqn_combined_model_results, _ = dqn_experiments(1, True, True, True, False, True)
    dqn_single_model_results, dqn_single_model_worlds = dqn_experiments(1, False, True, True, False, True)
    #dqn_single_model_results, dqn_single_model_worlds = dqn_experiments(1, False, True, True, False, False)


# PPO part (pre-train + train)
def pretrain_ppo(
        generated_data_size: int,
        num_epochs: int,
        actor_config: dict,
        critic_config: dict,
        dir_with_models: str,
        actor_pretrain_filename: str = None,
        critic_pretrain_filename: str = None,
        pretrain_dataset_filename: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    def ppo_batches(data, batch_size=64, embedding=None):
        n = graph_size
        amatrix_cols = get_amatrix_cols(n)

        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch["addr"].values
            dst = batch["dst"].values
            new_addr = batch['next_addr'].values
            v_func = batch['addr_v_func'].values
            allowed_neighbours = []

            if embedding is not None:
                amatrices = batch[amatrix_cols].values

                nets_inputs = []
                actor_outputs = []

                for addr_, dst_, new_addr_, A in zip(addr, dst, new_addr, amatrices):
                    A = A.reshape(n, n)

                    embedding.fit(A)

                    current_neighbours = []
                    for idx, distance in enumerate(A[int(addr_)]):
                        if distance != 0:
                            current_neighbours.append(
                                embedding.transform(A, idx)
                            )

                    allowed_neighbours.append(current_neighbours)

                    addr_emb = embedding.transform(A, int(addr_))
                    dst_emb = embedding.transform(A, int(dst_))
                    new_addr_emb = embedding.transform(A, int(new_addr_))

                    nets_inputs.append([addr_emb, dst_emb])
                    actor_outputs.append(new_addr_emb)

                [addr, dst] = stack_batch(nets_inputs)
                new_addr = np.array(actor_outputs)

            net_input = (torch.FloatTensor(addr), torch.FloatTensor(dst))

            actor_output = torch.FloatTensor(new_addr)
            critic_output = torch.FloatTensor(v_func)

            yield net_input, actor_output, critic_output, allowed_neighbours

    def critic_pretrain_epoch(net, data, **kwargs):
        loss_func = torch.nn.MSELoss()
        for critic_input, actor_target, critic_target, allowed_neighbours in ppo_batches(data, **kwargs):
            net.optimizer.zero_grad()
            output = net(*critic_input)
            loss = loss_func(output, critic_target.unsqueeze(1))
            loss.backward()
            net.optimizer.step()
            yield float(loss)

    def actor_pretrain_epoch(net, data, **kwargs):
        loss_func = torch.nn.MSELoss()
        for actor_input, actor_target, critic_target, allowed_neighbours in ppo_batches(data, **kwargs):
            net.optimizer.zero_grad()
            output = net(*actor_input)
            loss = loss_func(output, actor_target)
            loss.backward()
            net.optimizer.step()
            yield float(loss)

    def critic_pretrain(net, data, **kwargs) -> np.ndarray:
        critic_losses = []
        for _ in tqdm(range(num_epochs), desc='Critic pretrain'):
            sum_loss = 0
            loss_cnt = 0
            for loss in critic_pretrain_epoch(net, data, **kwargs):
                sum_loss += loss
                loss_cnt += 1
            critic_losses.append(sum_loss / loss_cnt)
        if critic_pretrain_filename is not None:
            net.change_label(pretrain_filename)
            # net._label = critic_pretrain_filename
            net.save()
        return np.array(critic_losses, dtype=np.float32)

    def actor_pretrain(net, data, **kwargs) -> np.ndarray:
        actor_losses = []
        for _ in tqdm(range(num_epochs), desc='Actor pretrain'):
            sum_loss = 0
            loss_cnt = 0
            for loss in actor_pretrain_epoch(net, data, **kwargs):
                sum_loss += loss
                loss_cnt += 1
            actor_losses.append(sum_loss / loss_cnt)
        if actor_pretrain_filename is not None:
            net.change_label(pretrain_filename)
            # net._label = actor_pretrain_filename
            net.save()
        return np.array(actor_losses, dtype=np.float32)

    def networks_pretrain(
            data: pd.DataFrame,
            actor_model: torch.nn.Module,
            critic_model: torch.nn.Module,
            conv_emb=None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        actor_losses = actor_pretrain(
            actor_model, data, embedding=conv_emb
        )

        critic_losses = critic_pretrain(
            critic_model, data, embedding=conv_emb
        )

        return actor_losses, critic_losses

    data = gen_episodes_progress(
        'ppo_emb',  # TODO fix it
        generated_data_size,
        ignore_saved=True,
        context="conveyors",
        random_seed=random_seed,
        run_params=scenario,
        save_path=pretrain_dataset_filename
    )
    shuffled_data = data.sample(frac=1)

    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)

    actor_args = {
        'scope': dir_with_models,
        'embedding_dim': emb_dim
    }
    actor_args = dict(**actor_config, **actor_args)
    actor_model = PPOActor(**actor_args)

    critic_args = {
        'scope': dir_with_models,
        'embedding_dim': emb_dim
    }
    critic_args = dict(**critic_config, **critic_args)
    critic_model = PPOCritic(**critic_args)

    actor_losses, critic_losses = networks_pretrain(shuffled_data, actor_model, critic_model, conv_emb=conv_emb)

    return actor_losses, critic_losses


def train_ppo(
        progress_step: int,
        router_type: str,
        dir_with_models: str,
        actor_pretrain_filename: str,
        critic_pretrain_filename: str,
        actor_train_filename: str,
        critic_train_filename: str,
        random_seed: int,
        work_with_files: bool,
        retrain: bool
):
    scenario["settings"]["router"][router_type]["dir_with_models"] = dir_with_models
    scenario["settings"]["router"][router_type]["actor_load_filename"] = actor_pretrain_filename
    scenario["settings"]["router"][router_type]["critic_load_filename"] = critic_pretrain_filename

    event_series, runner = run_single(
        run_params=scenario,
        router_type=router_type,
        progress_step=progress_step,
        ignore_saved=[True],
        random_seed=random_seed
    )

    world = runner.world
    some_router = next(iter(next(iter(world.handlers.values())).routers.values()))

    actor_model = some_router.actor
    actor_model.change_label(actor_train_filename)

    critic_model = some_router.critic
    critic_model.change_label(critic_train_filename)

    if work_with_files:
        if retrain:
            if False:  # some_router.use_single_neural_network: TODO implement
                actor_model.save()
                critic_model.save()
            else:
                print("Warning: saving/loaded models trained in simulation is only implemented "
                      "when use_single_neural_network = True. The models were not saved to disk.")
        else:
            actor_model.restore()
            critic_model.restore()

    return event_series, world


if ppo_emb_exists:
    ppo_emb_config = scenario['settings']['router']['ppo_emb']
    actor_config = ppo_emb_config['actor']
    critic_config = ppo_emb_config['critic']

    dir_with_models = 'conveyor_models_ppo'

    actor_pretrain_filename = f'actor_pretrained{filename_suffix}'
    actor_pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / actor_pretrain_filename

    critic_pretrain_filename = f'critic_pretrained{filename_suffix}'
    critic_pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / critic_pretrain_filename

    actor_trained_filename = f'actor_trained{filename_suffix}'
    actor_trained_path = Path(TORCH_MODELS_DIR) / dir_with_models / actor_trained_filename

    critic_trained_filename = f'critic_trained{filename_suffix}'
    critic_trained_path = Path(TORCH_MODELS_DIR) / dir_with_models / critic_trained_filename

    do_pretrain = force_pretrain or not actor_pretrain_path.exists() or not critic_pretrain_path.exists()
    do_train = force_train or not actor_trained_path.exists() or not critic_trained_path.exists()

    print(f'Actor: {actor_pretrain_path}')
    print(f'Critic: {critic_pretrain_path}')

    if do_pretrain:
        print('Pretraining PPO Models...')
        actor_losses, critic_losses = pretrain_ppo(
            pretrain_data_size,
            pretrain_epochs_num,
            actor_config,
            critic_config,
            dir_with_models,
            actor_pretrain_filename,
            critic_pretrain_filename,
            '../logs/data_conveyor_ppo.csv'
        )
        print(f'Actor loss: {actor_losses.tolist()}')
        print(f'Critic loss: {critic_losses.tolist()}')
    else:
        print('Using already pretrained models')

    if do_train:
        print('Training PPO Model...')
        ppo_log, ppo_world = train_ppo(
            train_data_size,
            'ppo_emb',
            dir_with_models,
            actor_pretrain_filename,
            critic_pretrain_filename,
            actor_trained_filename,
            critic_trained_filename,
            random_seed,
            True,
            True
        )
    else:
        print('Skip training process...')


# REINFORCE part (pre-train + train)
def pretrain_reinforce(
        generated_data_size: int,
        num_epochs: int,
        actor_config: dict,
        dir_with_models: str,
        actor_pretrain_filename: str = None,
        pretrain_dataset_filename: str = None
) -> np.ndarray:
    def reinforce_batches(data, batch_size=64, embedding=None):
        n = graph_size
        amatrix_cols = get_amatrix_cols(n)

        for a, b in make_batches(data.shape[0], batch_size):
            batch = data[a:b]
            addr = batch["addr"].values
            dst = batch["dst"].values
            new_addr = batch['next_addr'].values
            allowed_neighbours = []

            if embedding is not None:
                amatrices = batch[amatrix_cols].values

                nets_inputs = []
                actor_outputs = []

                for addr_, dst_, new_addr_, A in zip(addr, dst, new_addr, amatrices):
                    A = A.reshape(n, n)

                    embedding.fit(A)

                    current_neighbours = []
                    for idx, distance in enumerate(A[int(addr_)]):
                        if distance != 0:
                            current_neighbours.append(
                                embedding.transform(A, idx)
                            )

                    allowed_neighbours.append(current_neighbours)

                    addr_emb = embedding.transform(A, int(addr_))
                    dst_emb = embedding.transform(A, int(dst_))
                    new_addr_emb = embedding.transform(A, int(new_addr_))

                    nets_inputs.append([addr_emb, dst_emb])
                    actor_outputs.append(new_addr_emb)

                [addr, dst] = stack_batch(nets_inputs)
                new_addr = np.array(actor_outputs)

            net_input = (torch.FloatTensor(addr), torch.FloatTensor(dst))

            actor_output = torch.FloatTensor(new_addr)

            yield net_input, actor_output, allowed_neighbours

    def actor_pretrain_epoch(net, data, **kwargs):
        loss_func = torch.nn.MSELoss()
        for actor_input, actor_target, allowed_neighbours in reinforce_batches(data, **kwargs):
            net.optimizer.zero_grad()
            output = net(*actor_input)
            loss = loss_func(output, actor_target)
            loss.backward()
            net.optimizer.step()
            yield float(loss)

    def actor_pretrain(net, data, **kwargs) -> np.ndarray:
        actor_losses = []
        for _ in tqdm(range(num_epochs), desc='Actor pretrain'):
            sum_loss = 0
            loss_cnt = 0
            for loss in actor_pretrain_epoch(net, data, **kwargs):
                sum_loss += loss
                loss_cnt += 1
            actor_losses.append(sum_loss / loss_cnt)
        if actor_pretrain_filename is not None:
            net.change_label(actor_pretrain_filename)
            # net._label = actor_pretrain_filename
            net.save()
        return np.array(actor_losses, dtype=np.float32)

    data = gen_episodes_progress(
        'ppo_emb',  # TODO fix it
        generated_data_size,
        ignore_saved=True,
        context="conveyors",
        random_seed=random_seed,
        run_params=scenario,
        save_path=pretrain_dataset_filename
    )
    shuffled_data = data.sample(frac=1)

    conv_emb = CachedEmbedding(LaplacianEigenmap, dim=emb_dim)

    actor_args = {
        'scope': dir_with_models,
        'embedding_dim': emb_dim
    }
    actor_args = dict(**actor_config, **actor_args)
    actor_model = PPOActor(**actor_args)

    actor_losses = actor_pretrain(
        actor_model, shuffled_data, embedding=conv_emb
    )

    return actor_losses


def train_reinforce(
        progress_step: int,
        router_type: str,
        dir_with_models: str,
        pretrain_filename: str,
        train_filename: str,
        random_seed: int,
        work_with_files: bool,
        retrain: bool
):
    scenario["settings"]["router"][router_type]["dir_with_models"] = dir_with_models
    scenario["settings"]["router"][router_type]["load_filename"] = pretrain_filename

    event_series, runner = run_single(
        run_params=scenario,
        router_type=router_type,
        progress_step=progress_step,
        ignore_saved=[True],
        random_seed=random_seed
    )

    world = runner.world
    some_router = next(iter(next(iter(world.handlers.values())).routers.values()))

    actor_model = some_router.actor
    actor_model.change_label(train_filename)

    if work_with_files:
        if retrain:
            # print(dir(some_router))
            if some_router.use_single_network:
                actor_model.save()
            else:
                print("Warning: saving/loaded models trained in simulation is only implemented "
                      "when use_single_neural_network = True. The models were not saved to disk.")
        else:
            actor_model.restore()

    return event_series, world


# pretrain
if reinforce_emb_exists:
    reinforce_serieses = []

    from dqnroute.agents.routers.reinforce import PackageHistory
    from collections import defaultdict

    reinforce_emb_config = scenario['settings']['router']['reinforce_emb']
    reinforce_config = reinforce_emb_config['actor']

    dir_with_models = 'conveyor_models_reinforce'

    reinforce_pretrain_filename = f'pretrained{filename_suffix}'
    reinforce_pretrain_path = Path(TORCH_MODELS_DIR) / dir_with_models / reinforce_pretrain_filename

    trained_filename = f'actor_trained{filename_suffix}'
    trained_path = Path(TORCH_MODELS_DIR) / dir_with_models / trained_filename

    do_pretrain = force_pretrain or not reinforce_pretrain_path.exists() or True
    do_train = force_train or not trained_path.exists() or True

    print(f'Reinforce model: {reinforce_pretrain_path}')

    for _ in range(30):
        PackageHistory.routers = defaultdict(dict)
        PackageHistory.rewards = defaultdict(list)
        PackageHistory.log_probs = defaultdict(list)
        PackageHistory.finished_packages = set()
        PackageHistory.started_packages = set()

        if do_pretrain:
            print('Pretraining REINFORCE Models...')
            reinforce_losses = pretrain_reinforce(
                pretrain_data_size,
                pretrain_epochs_num,
                reinforce_config,
                dir_with_models,
                reinforce_pretrain_filename,
                '../logs/data_conveyor_reinforce.csv'
            )
            print(f'Actor loss: {reinforce_losses.tolist()}')
        else:
            print('Using already pretrained models')

        if do_train:
            print('Training REINFORCE Model...')
            reinforce_log, reinforce_world = train_reinforce(
                train_data_size,
                'reinforce_emb',
                dir_with_models,
                reinforce_pretrain_filename,
                trained_filename,
                random_seed,
                True,
                True
            )
        else:
            print('Skip training process...')

        reinforce_serieses.append(reinforce_log.getSeries(add_avg=True))


def train(
        progress_step: int,
        router_type: str,
        random_seed: int,
):
    event_series, runner = run_single(
        run_params=scenario,
        router_type=router_type,
        progress_step=progress_step,
        ignore_saved=[True],
        random_seed=random_seed
    )

    world = None

    return event_series, world


def get_learning_step_indices() -> Optional[Set[int]]:
    if args.learning_step_indices is None:
        return None
    return set([int(s) for s in args.learning_step_indices.split(",")])


#def get_symbolic_analyzer() -> SymbolicAnalyzer:
#    return SymbolicAnalyzer(g, softmax_temperature, probability_smoothing,
#                            args.verification_lr, delta_q_max=args.input_max_delta_q)


print(f"Running command {args.command}...")

# Simulate and make plots
if args.command == "run":
    _legend_txt_replace = {
        "networks": {
            "link_state": "Shortest paths", "simple_q": "Q-routing", "pred_q": "PQ-routing",
            "glob_dyn": "Global-dynamic", "dqn": "DQN", "dqn_oneout": "DQN (1-out)",
            "dqn_emb": "DQN-LE", "centralized_simple": "Centralized control", "ppo_emb": "PPO",
            'reinforce_emb': 'REINFORCE'
        }, "conveyors": {
            "link_state": "Vyatkin-Black", "simple_q": "Q-routing", "pred_q": "PQ-routing",
            "glob_dyn": "Global-dynamic", "dqn": "DQN", "dqn_oneout": "DQN (1-out)",
            "dqn_emb": "DQN-LE", "centralized_simple": "BSR", "ppo_emb": "PPO",
            'reinforce_emb': 'REINFORCE'
        }
    }

    _targets = {"time": "avg", "energy": "sum", "collisions": "sum"}

    _ylabels = {
        "time": "Mean delivery time",
        "energy": "Total energy consumption",
        "collisions": "Cargo collisions"
    }

    series = []
    series_types = []

    def get_results(results, name):
        df = pd.concat(results, ignore_index=True)
        print(type(df))
        df.to_csv(f'{name}_all_results.csv')

        global series
        global series_types

        basic_series = None

        for s in results:
            if basic_series is None:
                basic_series = s
            else:
                basic_series += s
        basic_series /= len(results)

        series += [basic_series]
        series_types += [name]

        print(f'{name} mean delivery time: {np.mean(basic_series["time_avg"])}')
        print(f'{name} mean energy consumption: {np.mean(basic_series["energy_avg"])}')
        print(f'{name} sum collision number: {np.sum(basic_series["collisions_sum"])}')

        return basic_series

    if dqn_emb_exists:
        single_series = get_results(dqn_single_model_results, 'DQN-LE')
        # combined_series = get_results(dqn_combined_model_results, 'DQN-LE-COMBINED')

    if ppo_emb_exists:
        series += [ppo_log.getSeries(add_avg=True)]
        print(np.mean(series[-1]['time_avg']))
        series_types += ['ppo_emb']

    if reinforce_emb_exists:
        reinforce_series = get_results(reinforce_serieses, 'REINFORCE')
        print(type(reinforce_series))
        # reinforce_basic_series = None
        # for s in reinforce_serieses:
        #     if reinforce_basic_series is None:
        #         reinforce_basic_series = s
        #     else:
        #         reinforce_basic_series += s
        # reinforce_basic_series /= len(reinforce_serieses)

    # perform training/simulation with other approaches
    for router_type in router_types:
        if router_type != "dqn_emb" and router_type != 'ppo_emb' and router_type != 'reinforce_emb':
            s, _ = train(train_data_size, router_type, random_seed)
            series += [s.getSeries(add_avg=True)]
            series_types += [router_type]

    dfs = []
    for router_type, s in zip(series_types, series):
        df = s.copy()
        add_cols(df, router_type=router_type, seed=random_seed)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)

    def print_sums(df):
        for tp in router_types:
            x = df.loc[df["router_type"] == tp, "count"].sum()
            txt = _legend_txt_replace.get(tp, tp)
            print(f"  {txt}: {x}")

    def plot_data(data, meaning="time", figsize=(15, 5), xlim=None, ylim=None,
                  xlabel="Simulation time", ylabel=None, font_size=14, title=None, save_path=None,
                  draw_collisions=False, context="networks", **kwargs):
        if "time" not in data.columns:
            datas = split_dataframe(data, preserved_cols=["router_type", "seed"])
            for tag, df in datas:
                if tag == "collisions" and not draw_collisions:
                    print("Number of collisions:")
                    print_sums(df)
                    continue
                xlim = kwargs.get(tag + "_xlim", xlim)
                ylim = kwargs.get(tag + "_ylim", ylim)
                save_path = kwargs.get(tag + "_save_path", save_path)
                plot_data(df, meaning=tag, figsize=figsize, xlim=xlim, ylim=ylim,
                          xlabel=xlabel, ylabel=ylabel, font_size=font_size,
                          title=title, save_path=save_path, context="conveyors")
            return

        target = _targets[meaning]
        if ylabel is None:
            ylabel = _ylabels[meaning]

        fig = plt.figure(figsize=figsize)
        ax = sns.lineplot(x="time", y=target, hue="router_type", data=data, err_kws={"alpha": 0.1}, )
        handles, labels = ax.get_legend_handles_labels()
        new_labels = list(map(lambda l: _legend_txt_replace[context].get(l, l), labels[:]))
        ax.legend(handles=handles[:], labels=new_labels, fontsize=font_size)
        ax.tick_params(axis="both", which="both", labelsize=int(font_size * 0.75))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)

        if save_path is not None:
            fig.savefig(f"../img/{save_path}", bbox_inches="tight")

    plot_data(dfs, figsize=(14, 8), font_size=22,
              time_save_path="time-plot.pdf",
              energy_save_path="energy-plot.pdf",
              collisions_save_path='collisions-plot.pdf'
              )
elif args.command == 'compute_edt':
    edt = getattr(verif_cmds, args.command)(
            dqn_single_model_worlds[0],
            args.sink,
            args.source,
            softmax_temperature,
            probability_smoothing,
            args.simple_path_cost,
            args.verbose)

    print(f'EDT: {edt}')
elif args.command == 'find_advers_emb':
    emb, edt  = getattr(verif_cmds, args.command)(
        dqn_single_model_worlds[0],
        args.sink,
        args.source,
        softmax_temperature,
        probability_smoothing,
        args.max_perturbation_norm,
        args.simple_path_cost,
        args.euclid_norm,
        args.verbose)

    print(emb)
    print(f'EDT: {edt}')
elif args.command == 'verif_edt_bound_wrt_embs':
    # dynamic graph?
    func = getattr(verif_cmds, args.command)(
        dqn_single_model_worlds[0],
        args.sink,
        args.source,
        softmax_temperature,
        probability_smoothing,
        args.max_perturbation_norm,
        args.marabou_path,
        args.edt_bound,
        args.simple_path_cost,
        args.verify_dynamic_graph,
        args.marabou_memory_limit,
        args.verbose
    )
    print(func)
#elif args.command == 'verif_edt_robustness_wrt_emb':
#    func = getattr(verif.commands, args.command)(
#        wordl
#        use_l2
#        norm_bound
#        const_bound
#        verbose
#    )
# Evaluate the expected delivery cost assuming a change in NN parameters and make plots            
#elif args.command == "q_adversarial_search":
#    sa = get_symbolic_analyzer()
#    learning_step_index = -1
#    requested_indices = get_learning_step_indices()
#    for source, sink, sink_embedding, ma in get_source_sink_pairs("Measuring robustness of delivery"):
#        sink_embeddings = sink_embedding.repeat(2, 1)
#        objective, lambdified_objective = ma.get_edt_sol(source)
#        for node_key in g.node_keys:
#            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)
#            for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
#                learning_step_index += 1
#                if requested_indices is not None and learning_step_index not in requested_indices:
#                    continue
#                print(f"  Considering learning step {node_key} → {neighbor_key}...")
#                # compute
#                # we assume a linear change of parameters
#                reference_q = sa.compute_gradients(current_embedding, sink_embedding,
#                                                   neighbor_embedding).flatten().item()
#                actual_qs = np.linspace(-sa.delta_q_max, sa.delta_q_max,
#                                        args.q_adversarial_no_points) + reference_q
#                kappa, lambdified_kappa = sa.get_transformed_cost(ma, objective, args.cost_bound)
#                objective_values, kappa_values = [torch.empty(len(actual_qs)) for _ in range(2)]
#                for i, actual_q in enumerate(actual_qs):
#                    ps = sa.compute_ps(ma, sink, sink_embeddings, reference_q, actual_q)
#                    objective_values[i] = lambdified_objective(*ps)
#                    kappa_values[i] = lambdified_kappa(*ps)
#                # print(((objective_values > args.cost_bound) != (kappa_values > 0)).sum())
#                fig, axes = plt.subplots(3, 1, figsize=(10, 10))
#                plt.subplots_adjust(hspace=0.4)
#                caption_starts = *(["Delivery cost (τ)"] * 2), "Transformed delivery cost (κ)"
#                values = *([objective_values] * 2), kappa_values
#                axes[0].set_yscale("log")
#                for ax, caption_start, values in zip(axes, caption_starts, values):
#                    label = (f"{caption_start} from {source} to {sink} when making optimization"
#                             f" step with current={node_key}, neighbor={neighbor_key}")
#                    print(f"    Plotting: {caption_start}...")
#                    ax.set_title(label)
#                    ax.plot(actual_qs, values)
#                    y_delta = 0 if np.ptp(values) > 0 else 5
#                    # show the zero step value:
#                    ax.vlines(reference_q, min(values) - y_delta, max(values) + y_delta)
#                    ax.hlines(values[len(values) // 2], min(actual_qs), max(actual_qs))
#                # show the verification bound:
#                for i in range(2):
#                    axes[i].hlines(args.cost_bound, min(actual_qs), max(actual_qs))
#                axes[2].hlines(0, min(actual_qs), max(actual_qs))
#                plt.savefig(f"../img/{filename_suffix}_{learning_step_index}.pdf", bbox_inches="tight")
#                plt.close()
#                print(f"    Empirically found maximum of τ: {objective_values.max():.6f}")
#                print(f"    Empirically found maximum of κ: {kappa_values.max():.6f}")
## Formally verify the bound on the expected delivery cost w.r.t. learning step magnitude  
#elif args.command == "q_adversarial_verification":
#    sa = get_symbolic_analyzer()
#    sa.load_matrices()
#    learning_step_index = -1
#    requested_indices = get_learning_step_indices()
#    for source, sink, sink_embedding, ma in get_source_sink_pairs("Verifying robustness of delivery"):
#        objective, _ = ma.get_edt_sol(source)
#        for node_key in g.node_keys:
#            current_embedding, neighbors, neighbor_embeddings = g.node_to_embeddings(node_key, sink)
#            for neighbor_key, neighbor_embedding in zip(neighbors, neighbor_embeddings):
#                learning_step_index += 1
#                if requested_indices is not None and learning_step_index not in requested_indices:
#                    continue
#                print(f"  Considering learning step {node_key} → {neighbor_key}...")
#                lbc = LipschitzBoundComputer(sa, ma, objective, sink, current_embedding,
#                                             sink_embedding, neighbor_embedding, args.cost_bound)
#                if lbc.prove_bound(args.q_adversarial_verification_no_points):
#                    print("    Proof found!")
#                print(f"    Number of evaluations of κ: {lbc.no_evaluations}")
#                print(f"    Maximum depth reached: {lbc.max_depth}")
#
#else:
#    raise RuntimeError(f"Unknown command {args.command}.")
