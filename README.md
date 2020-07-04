# Tie-decay temporal networks
Code accompanying "Tie-decay temporal networks in continuous time and eigenvector-based centralities" by Walid Ahmad, Mason Porter, and Mariano Beguerisse-Díaz.
<sup>1</sup>

This repository contains utilities for loading and computing tie-decay centrality scores for temporal networks.

<img src="https://user-images.githubusercontent.com/10912088/86521655-93bc2280-be21-11ea-8d2a-662a5a951e74.gif" width="75%">

### Setup
```
conda env create -f conda_environment.yml
python setup.py develop
```

### Load Dataset
```python
from tiedecay.dataset import Dataset

raw_data = [(1, 5, "2020-01-01-00:01:23"), (3, 2, "2019-08-12-11:01:34"), ...]
user_mapping = {1: "henry ford",  2: "nikola tesla", ...}

dataset = Dataset(raw_data, user_mapping)
```

### Compute centrality scores
```python
from tiedecay.construct import TieDecayNetwork

# half-life of one day
alpha = np.log(2)/24/3600
tdn = TieDecayNetwork(dataset, alpha=alpha)
```

### References
[1] arXiv preprint, 2018 [arXiv:1805.00193v2](https://arxiv.org/abs/1805.00193v2)
