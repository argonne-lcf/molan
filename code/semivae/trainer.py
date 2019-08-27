import torch
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.nn.utils import clip_grad_norm_
from .utils import Logger, CircularBuffer, valid_smiles
from .annealers import KLCylicAnnealer, WAnnealer
import numpy as np
import sklearn.metrics
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from .utils import set_seed


def r2_score(y_true, y_pred):
    try:
        return sklearn.metrics.r2_score(y_true, y_pred)
    except:
        pass
    return np.nan


class MosesTrainer(ABC):

    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def get_dataloader(self, model, data, collate_fn=None, shuffle=True):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        return DataLoader(data, batch_size=self.config.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_seed
                          if self.n_workers > 0 else None)

    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def get_vocabulary(self, data):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass


class VAETrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
        self.at_results_dir = lambda x: os.path.join(
            self.config.results_dir, x)
        self.at_snapshot_dir = lambda x: os.path.join(self.config.results_dir,
                                                      self.config.snapshot_dir, x)

    def is_valid(self, model, x):
        valid_data = [valid_smiles(model.tensor2string(i_x)) for i_x in x]
        return sum(valid_data) / float(len(valid_data)) * 100.0

    def get_vocabulary(self, data):
        raise ValueError('Not implemented!')

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=lambda x: len(x[0]), reverse=True)

            def ids2tensor(x): return torch.tensor(
                x, dtype=torch.long, device=device)
            x_tensors = [ids2tensor(t[0]) for t in data]

            y_tensors = np.array([tup[1] for tup in data],
                                 dtype=np.float32).reshape(-1, 1)
            y_tensors = torch.from_numpy(y_tensors).to(device)
            mask_tensors = np.array(
                [tup[2] for tup in data], dtype=np.float32).reshape(-1, 1)
            mask_tensors = torch.from_numpy(mask_tensors).to(device)
            return x_tensors, y_tensors, mask_tensors

        return collate

    def train(self, model, train_loader, val_loader=None, logger=None):

        device = model.device
        n_epochs = self.config.n_epochs
        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr)
        kl_annealer = KLCylicAnnealer.from_config(self.config)
        y_annealer = WAnnealer.from_config(self.config)

        model.zero_grad()
        pbar = tqdm(range(n_epochs), desc='Epochs')
        best_loss = np.inf
        for epoch in pbar:

            kl_weight = kl_annealer(int(epoch))
            y_weight = y_annealer(epoch)

            postfix = self.train_epoch(model, epoch,
                                       train_loader, kl_weight, y_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.at_results_dir(self.config.log_file))

            if val_loader is not None:
                postfix = self.train_epoch(
                    model, epoch, val_loader, kl_weight, y_weight)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            best_model = postfix[self.config.save_monitor] < best_loss
            if best_model:
                best_loss = postfix[self.config.save_monitor]
            if (self.config.model_save is not None) and best_model:
                model = model.to('cpu')
                filename = '{}_best{}.pt'.format(self.config.model_save, epoch)
                torch.save(model.state_dict(), self.at_snapshot_dir(filename))
                model = model.to(device)

            pbar.set_postfix(postfix)

    def train_epoch(self, model, epoch, data_loader, kl_weight, y_weight, optimizer=None):

        is_training = optimizer is not None
        mode = 'Train' if is_training else 'Eval'
        if is_training:
            model.train()

        else:
            model.eval()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        y_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        valid_values = CircularBuffer(self.config.n_last)

        preds = []
        targets = []

        pbar = tqdm(data_loader, desc='{} {}'.format(mode, epoch), leave=False)
        for i, input_batch in enumerate(pbar):
            x = tuple(data.to(model.device) for data in input_batch[0])
            y = input_batch[1]
            mask = input_batch[2]
            kl_loss, recon_loss, y_loss, x_hat, y_hat = model(x, y, mask)
            loss = recon_loss + kl_weight * kl_loss + y_weight * y_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            y_loss_values.add(y_loss.item())
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr'] if optimizer is not None
                  else np.nan)
            if not is_training:
                cpu_mask = mask.detach().cpu().numpy().astype(bool)
                valid_values.add(self.is_valid(
                    model, x_hat.detach().cpu().numpy()))
                preds.extend(y_hat.detach().cpu().numpy()[cpu_mask])
                targets.extend(y.detach().cpu().numpy()[cpu_mask])

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            y_loss_value = y_loss_values.mean()
            postfix = [f'loss={loss_value:.2e}',
                       f'(kl={kl_loss_value:.2e}',
                       f'recon={recon_loss_value:.2e}',
                       f'y={y_loss_value:.2e}',
                       f') klw={kl_weight:.3f}',
                       f' yw={y_weight:.3f}',
                       f'lr={lr:.2e}']
            pbar.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'y_weight': y_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'y_loss': y_loss_value,
            'loss': loss_value,
            'r2': r2_score(targets, preds) if not is_training else np.nan,
            'valid': valid_values.mean() if not is_training else np.nan,
            'mode': mode}

        return postfix

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )

        self.train(model, train_loader, val_loader, logger)
        return model
