# Mamba4Rec

> **Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models**\
> Chengkai Liu, Jianghao Lin, Jianling Wang, Hanzhou Liu, James Caverlee\
> Paper: https://arxiv.org/abs/2403.03900

## Usage

### Requirements

* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+
* Install RecBole:
  * `pip install recbole`
* Install causal Conv1d and the core Mamba package:
  * `pip install causal-conv1d>=1.2.0`
  * `pip install mamba-ssm`

### Run

`python run.py`


Specifying the dataset in `config.yaml` will trigger an automatic download. Please set an appropriate maximum sequence length for each dataset before training.


## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba), [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d), and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.