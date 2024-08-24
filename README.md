# Mamba4Rec

> **Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models (RelKD@KDD 2024 Best Paper Award Award)**\
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

You can also refer to the required environment specifications in `environment.yaml`.

### Run

```python run.py```


Specifying the dataset in `config.yaml` will trigger an automatic download. Please set an appropriate maximum sequence length in `config.yaml` for each dataset before training.


## Citation
```bibtex
@article{liu2024mamba4rec,
      title={Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models}, 
      author={Chengkai Liu and Jianghao Lin and Jianling Wang and Hanzhou Liu and James Caverlee},
      journal={arXiv preprint arXiv:2403.03900},
      year={2024}
}
```


## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba), [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d), and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.
