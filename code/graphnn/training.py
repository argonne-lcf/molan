from collections import OrderedDict
import sklearn
import sklearn.metrics
import numpy as np
import torch
import torch.nn.functional as F
from . mol2graph import mol2torchdata
from torch_geometric.data import DataLoader


def clear_model(model):
    del model
    torch.cuda.empty_cache()


def get_dataloader(df, index, target, mol_column, batch_size, y_scaler):
    y_values = df.loc[index, target].values.reshape(-1, 1)
    y = y_scaler.transform(y_values).ravel().astype(np.float32)
    x = df.loc[index, mol_column].progress_apply(mol2torchdata).tolist()
    for data, y_i in zip(x, y):
        data.y = torch.tensor([y_i], dtype=torch.float)
    data_loader = DataLoader(x, batch_size=batch_size,
                             shuffle=True, drop_last=True)
    return data_loader


def train_step(model, data_loader, optimizer, scheduler, device):
    model.train()
    loss_sum = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_sum += loss.item() * data.num_graphs
        optimizer.step()

    n = float(sum([data.num_graphs for data in data_loader]))
    stats = {'train_loss': loss_sum / n}
    if scheduler:
        scheduler.step(loss_sum)
    return stats


def reg_stats(y_true, y_pred):
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return r2, mae


def eval_step(model, data_loader, y_scaler, device, cv_result,
              best_value):
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        y_pred = []
        y_true = []
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
            loss = F.mse_loss(output, data.y)
            loss_sum += loss.item() * data.num_graphs

        y_pred = y_scaler.inverse_transform(
            np.array(y_pred).reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(
            np.array(y_true).reshape(-1, 1)).ravel()

        n = float(sum([data.num_graphs for data in data_loader]))
        stats = OrderedDict({'test_loss': loss_sum / n})
        stats['test_r2'], stats['test_mae'] = reg_stats(y_true, y_pred)
        if stats['test_r2'] >= best_value:
            best_value = stats['test_r2']
            cv_result['target'] = y_true
            cv_result['pred'] = y_pred

        return stats


def get_embeddings(model, data_loader, y_scaler, device):
    with torch.no_grad():
        model.eval()
        z = []
        y = []
        for data in data_loader:
            data = data.to(device)
            z_data = model.forward_gnn(data)
            y_data = model.pred(z_data)
            y.append(y_data.cpu().numpy())
            z.append(z_data.cpu().numpy())

        y = y_scaler.inverse_transform(np.vstack(y).reshape(-1, 1)).ravel()
        z = np.vstack(z)
    return z, y
