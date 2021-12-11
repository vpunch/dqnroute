## Preparation

### Project build

```
git clone <link to this repository>
cd dqnroute
python -m venv pythonenv
source pythonenv/bin/activate
pip install -r requirements.txt
```

### Marabou build

```
git clone https://github.com/NeuralNetworkVerification/Marabou
cd Marabou
mkdir build
cd build
cmake ..
```

During the build, the following error may occur: placement new constructing an
object of type... . This is probably due to the wrong version of the compiler.
To fix the error, you can use the command:

```
sed -i '1i #pragma GCC diagnostic ignored "-Wplacement-new="' src/nlr/*.cc ../src/nlr/*.cpp
```

To build the executable file, run:

```
cmake --build . -j 8
```

Now the file `Marabou` must be present in the current directory.

## Run

See `dqnroute/src/VerificationExperiments.py` to run verification experiments.

```
python ./Run.py
../launches/conveyor_topology_mukhutdinov/original_example_graph.yaml
../launches/conveyor_topology_mukhutdinov/original_example_settings_break_test.yaml
--routing_algorithms=dqn_emb
--command compute_edt
--sink=1
--source=1
--verbose
```

```
python ./Run.py
../launches/conveyor_topology_mukhutdinov/original_example_graph.yaml
../launches/conveyor_topology_mukhutdinov/original_example_settings_break_test.yaml
--command find_advers_emb --routing_algorithms=dqn_emb --sink=3 --source=1
--max_perturbation_norm=0.4 --verbose
```

The example of running the experiment on the command line:

```
python ./Run.py \
../launches/conveyor_topology_mukhutdinov/original_example_graph.yaml \
../launches/conveyor_topology_mukhutdinov/original_example_settings_energy_test.yaml \
--command embedding_adversarial_verification \
--routing_algorithms=dqn_emb \
--input_eps_l_inf=0.01 \
--skip_graphviz \
--marabou_path=/home/vanya/build/Marabou/build/Marabou
```

See `dqnroute/README.md` for list of available commands and their description.
