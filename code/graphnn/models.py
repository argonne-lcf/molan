import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


def str2act(act):
    activations = {'relu': nn.ReLU(), 'selu': nn.SELU(), 'celu': nn.CELU(
    ), 'softplus': nn.Softplus(), 'softmax': nn.Softmax(), 'sigmoid': nn.Sigmoid()}
    return activations[act]


def str2funct_act(act):
    activations = {'relu': F.relu, 'selu': F.selu, 'celu': F.celu,
                   'softplus': F.softplus, 'softmax': F.softmax, 'sigmoid': F.sigmoid}
    return activations[act]


def net_pattern(n_layers, base_size, ratio, maxv=1024):
    return [int(min(max(math.ceil(base_size * (ratio**i)), 0), maxv)) for i in range(0, n_layers)]


def make_mlp(start_dim, n_layers, ratio, act, batchnorm, dropout):
    layer_sizes = net_pattern(n_layers + 1, start_dim, ratio)
    layers = []
    for index in range(n_layers):
        layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))
        layers.append(str2act(act))
        if batchnorm:
            layers.append(nn.BatchNorm1d(layer_sizes[index + 1]))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers), layer_sizes[-1]


class MPNN(torch.nn.Module):
    def __init__(self, hparams, node_dim=None, edge_dim=None):
        super(MPNN, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hparams = hparams
        self.output_dim = 1

        # Linear atom embedding
        atom_dim = hparams['atom_dim']
        self.linatoms = torch.nn.Linear(self.node_dim, atom_dim)

        # MPNN part
        conv_dim = atom_dim * 2
        nnet = nn.Sequential(*[
            nn.Linear(self.edge_dim, conv_dim),
            str2act(hparams['conv_act']),
            nn.Linear(conv_dim, atom_dim * atom_dim)])
        self.conv = gnn.NNConv(atom_dim, atom_dim, nnet,
                               aggr=hparams['conv_aggr'], root_weight=False)
        self.gru = nn.GRU(atom_dim, atom_dim)

        # Graph embedding
        self.set2set = gnn.Set2Set(
            atom_dim, processing_steps=hparams['emb_steps'])

        # Build mlp
        self.using_mlp = hparams['mlp_layers'] > 0
        if self.using_mlp:
            self.mlp, last_dim = make_mlp(atom_dim * 2,
                                          hparams['mlp_layers'],
                                          hparams['mlp_dim_ratio'],
                                          hparams['mlp_act'],
                                          hparams['mlp_batchnorm'],
                                          hparams['mlp_dropout'])
        else:
            last_dim = atom_dim * 2

        # Prediction
        self.pred = nn.Linear(last_dim, self.output_dim)

    def forward_gnn(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = F.relu(self.linatoms(x))

        h = out.unsqueeze(0)
        embs = []
        for i in range(self.hparams['conv_n']):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            embs.append(self.set2set(out, data.batch))

        # Graph embedding
        if self.hparams['residual'] == 'residual':
            out = torch.sum(embs, axis=-1)
        else:
            out = embs[-1]

        # NNet part
        if self.using_mlp:
            out = self.mlp(out)
        return out

    def forward(self, data):
        out = self.forward_gnn(data)
        # Prediction
        out = self.pred(out)
        return out.view(-1)


class GCN(torch.nn.Module):
    def __init__(self, hparams, node_dim, edge_dim):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hparams = hparams
        self.output_dim = 1

        # Linear atom embedding
        self.linatoms = torch.nn.Linear(
            self.node_dim, hparams['conv_base_size'])

        # Graph Convolution
        emb_dim = hparams['emb_dim']
        conv_dims = net_pattern(hparams['conv_n_layers'],
                                hparams['conv_base_size'],
                                hparams['conv_ratio']) + [emb_dim]
        conv_layers = []
        for index in range(hparams['conv_n_layers']):
            conv_layers.append(gnn.GCNConv(
                conv_dims[index], conv_dims[index + 1], cached=False))

        self.graph_conv = nn.ModuleList(conv_layers)
        if self.hparams['conv_batchnorm']:
            self.bn = nn.ModuleList([nn.BatchNorm1d(dim)
                                     for dim in conv_dims[1:]])
        # Graph embedding
        if hparams['emb_set2set']:
            self.graph_emb = gnn.Set2Set(emb_dim, processing_steps=3)
            emb_dim = emb_dim * 2
        else:
            self.graph_emb = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                           str2act(hparams['emb_act']))

        # Build mlp
        self.using_mlp = hparams['mlp_layers'] > 0
        if self.using_mlp:
            self.mlp, last_dim = make_mlp(emb_dim,
                                          hparams['mlp_layers'],
                                          hparams['mlp_dim_ratio'],
                                          hparams['mlp_act'],
                                          hparams['mlp_batchnorm'],
                                          hparams['mlp_dropout'])
        else:
            last_dim = emb_dim

        # Prediction
        self.pred = nn.Linear(last_dim, self.output_dim)

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, data):
        x, edge_index = data.x, data.edge_index
        # Linear atom embedding
        x = self.linatoms(x)
        # GCN part
        for index in range(self.hparams['conv_n_layers']):
            x = self.graph_conv[index](x, edge_index)
            x = str2funct_act(self.hparams['conv_act'])(x)
        return x

    def forward_gnn(self, data, gradcam=False):
        x, edge_index = data.x, data.edge_index
        # Linear atom embedding
        x = self.linatoms(x)
        # GCN part
        for index in range(self.hparams['conv_n_layers']):
            x = self.graph_conv[index](x, edge_index)
            x = str2funct_act(self.hparams['conv_act'])(x)
            if gradcam and index == self.hparams['conv_n_layers'] - 1:
                # register the hook
                x.register_hook(self.activations_hook)

            if self.hparams['conv_batchnorm']:
                x = self.bn[index](x)

        # Graph embedding
        if self.hparams['emb_set2set']:
            x = self.graph_emb(x, data.batch)
        else:
            x = self.graph_emb(x)
            x = gnn.global_add_pool(x, data.batch)
        # NNet
        if self.using_mlp:
            x = self.mlp(x)
        return x

    def forward(self, data, gradcam=False):
        x = self.forward_gnn(data, gradcam)
        # Prediction
        x = self.pred(x)
        return x.view(-1)
