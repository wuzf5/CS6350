# ID3-categorical



## Usage

For a regular run:

```bash
cd ./DecisionTree/categorical
python main.py --purity_measure entropy --max_depth 6
```

To run a full sweep on all possible values of `purity_measure` and `max_depth`, run:

```bash
cd ./DecisionTree/categorical
chmod +x ./run.sh
./run.sh
```

