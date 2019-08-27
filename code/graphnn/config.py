
from collections import OrderedDict
import random


def random_hparams():
    hparams = OrderedDict()
    hparams['mlp_layers'] = random.randint(0, 6)  # 5
    hparams['mlp_base_dim'] = random.randint(25, 151)  # [25,150]
    hparams['mlp_dim_ratio'] = random.uniform(0.5, 1.50)
    hparams['mlp_dropout'] = random.uniform(0.0, 0.5)
    hparams['mlp_act'] = random.choice(['relu', 'selu', 'celu', 'softplus'])
    hparams['mlp_batchnorm'] = random.choice([True, False])
    hparams['emb_steps'] = random.randint(2, 7)
    hparams['atom_dim'] = random.randint(32, 251)
    hparams['residual'] = random.choice([True, False])
    hparams['conv_act'] = random.choice(['relu', 'selu', 'celu', 'softplus'])
    hparams['conv_dim'] = random.randint(25, 251)
    hparams['conv_aggr'] = 'add'
    hparams['conv_n'] = random.randint(3, 9)
    return hparams
