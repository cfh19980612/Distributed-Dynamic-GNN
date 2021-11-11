# Distributed-Dynamic-GNN
Distributed dynamic GNN system, which uses the EvolveGCN (git@github.com:IBM/EvolveGCN.git)
## Run Distributed.py
```
python Distributed.py --config_file ./experiments/parameters_example.yaml
```

## Some Issues
### 1. yaml issue
```
TypeError: load() missing 1 required positional argument: 'Loader'
```
### Solution
```
!pip install pyyaml==5.4.1
```
