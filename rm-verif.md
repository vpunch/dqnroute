## Подготовка к запуску

```
python -m venv dqnroute-env
cd dqnroute
source env/bin/activate
pip install -r requirements.txt
```

### Сборка Marabou:

```
git clone https://github.com/NeuralNetworkVerification/Marabou
cd Marabou
mkdir build
cd build
cmake ..
```

Далее во время сборки у меня возникала следующая ошибка: placement new
constructing an object of type... . Возможно, это произошло из-за слишком новой
версии компилятора. Чтобы ее исправить, мне пришлось внести изменения в
исходники этой командой:

```
sed -i '1i #pragma GCC diagnostic ignored "-Wplacement-new="' src/nlr/*.cc ../src/nlr/*.cpp
```

Сборка исполняемого файла:

```
cmake --build . -j 8
```

В текущей директории должен появиться файл `Marabou`.

## Запуск

Общая команда:

```
python ./Run.py \
../launches/conveyor_topology_mukhutdinov/original_example_graph.yaml \
../launches/conveyor_topology_mukhutdinov/original_example_settings_break_test.yaml \
--command <command> \
--marabou_path=<marabou_path> \
--routing_algorithms=dqn_emb \
--skip_graphviz
```

Заменить в ней `<marabou_path>` на путь к исполняемому файлу Marabou.

Выполнять в `dqnroute/src`.

### Расчет ожидаемой стоимости

Заменить `<command>` на `compute_expected_cost`.

### Поиск состязательных примеров в первом алгоритме

Заменить `<command>` на `embedding_adversarial_search`.

### Первый алгоритм

Заменить `<command>` на `embedding_adversarial_verification`

### Поиск состязательных примеров во втором алгоритме

Заменить `<command>` на `q_adversarial_search`.

### Второй алгоритм

Заменить `<command>` на `q_adversarial_verification`.
