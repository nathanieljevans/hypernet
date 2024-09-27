
import torch
from GaNN.models.GaNN import GaNN
from GaNN.models.MCDO import MCDO
from GaNN.models.MCBN import MCBN
import numpy as np
import torch.nn as nn

# TODO: implement a wassertein loss, KL loss, KDE-jensenshannon, etc; some kind of non-parametric density estimation for 03....ipynb

class EnergyDistanceLoss(nn.Module):
    def __init__(self):
        super(EnergyDistanceLoss, self).__init__()

    def forward(self, p_samples, q_sample):
        # p_samples: (n_samples, n_batch, n_outputs)
        # q_sample: (n_batch, n_outputs)

        # Define a function to compute energy distance for one batch element
        def energy_distance_single_batch(p, q):
            # p: (n_samples, n_outputs)
            # q: (1, n_outputs)
            dist_x = torch.cdist(p, p, p=2)  # Pairwise distance between predicted samples
            dist_y = torch.cdist(q.unsqueeze(0), q.unsqueeze(0), p=2)  # Pairwise distance for target (trivial)
            dist_xy = torch.cdist(p, q.unsqueeze(0), p=2)  # Pairwise distance between predicted and target

            # Energy distance computation
            energy_dist = 2 * dist_xy.mean() - dist_x.mean() - dist_y.mean()
            return energy_dist

        # Apply vmap to compute the energy distance across all batches
        energy_distance_batched = torch.vmap(energy_distance_single_batch)

        # Compute energy distance across all batches
        energy_distances = energy_distance_batched(p_samples.transpose(0, 1), q_sample)

        # Return the average energy distance across batches
        return energy_distances.mean()

def train_ien(x,y, model_kwargs, loss_fn='mse', lr=1e-4, batch_size=124, num_epochs=1000, nsamples=1000, compile=False, use_cuda=True): 

    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    x = x.to(device)
    y = y.to(device)

    if loss_fn == 'ce':
        out_channels = y.unique().view(-1).size(0) 
    else: 
        out_channels = y.size(1)

    model = GaNN(in_channels=x.size(1), 
                out_channels=out_channels, 
                **model_kwargs).to(device)
    
    if compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn == 'mse': 
        crit = torch.nn.MSELoss()
    elif loss_fn == 'l1': 
        crit = torch.nn.SmoothL1Loss()
    elif loss_fn == 'nll': 
        crit = torch.nn.GaussianNLLLoss()
    elif loss_fn in ['edl']: 
        crit = EnergyDistanceLoss()
    elif loss_fn == 'ce': 
        crit = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('unrecognized loss string, options: [mse, l1, nll, was]')

    losses = []
    for i in range(num_epochs): 

        batch_loss = []
        for idx in torch.split(torch.randperm(x.size(0)), batch_size): 
            optim.zero_grad()
            yhat = model(x[idx], samples=nsamples)

            if loss_fn in ['mse', 'l1']: 
                loss = crit(yhat, y[idx].unsqueeze(0).expand(nsamples,-1,-1))
            elif loss_fn == 'ce':
                loss = crit(yhat.contiguous().view(-1, out_channels), y[idx].unsqueeze(0).expand(nsamples,-1,-1).contiguous().view(-1))
            elif loss_fn in ['edl']: 
                loss = crit(yhat, y[idx])
            elif loss_fn == 'nll': 
                loss = crit(yhat.mean(0), y[idx], yhat.var(0) )
            else: 
                raise Exception('no loss objective defined')
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())
        losses.append(np.mean(batch_loss))

        print(f'progress: {i}/{num_epochs} --> loss: {losses[-1]:.3f}', end='\r')

    return model.cpu(), losses



def train_mcdo(x,y, model_kwargs, loss_fn='mse', lr=1e-4, batch_size=124, num_epochs=1000, compile=False, use_cuda=True): 

    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    x = x.to(device)
    y = y.to(device)

    if loss_fn == 'ce':
        out_channels = y.unique().view(-1).size(0) 
    else: 
        out_channels = y.size(1)

    model = MCDO(in_channels=x.size(1), 
                out_channels=out_channels, 
                **model_kwargs).to(device)
    
    if compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn == 'mse': 
        crit = torch.nn.MSELoss()
    elif loss_fn == 'l1': 
        crit = torch.nn.SmoothL1Loss()
    elif loss_fn == 'ce': 
        crit = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('unrecognized loss string, options: [mse, l1]')

    losses = []
    for i in range(num_epochs): 

        batch_loss = []
        for idx in torch.split(torch.randperm(x.size(0)), batch_size): 
            optim.zero_grad()
            yhat = model(x[idx])
            if loss_fn in ['mse', 'l1']: 
                loss = crit(yhat, y[idx])
            elif loss_fn == 'ce':
                loss = crit(yhat, y[idx].view(-1))
            else: 
                raise Exception('no loss objective defined')
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())
        losses.append(np.mean(batch_loss))

        print(f'progress: {i}/{num_epochs} --> loss: {losses[-1]:.3f}', end='\r')

    return model.cpu(), losses



def train_mcbn(x,y, model_kwargs, loss_fn='mse', lr=1e-4, batch_size=124, num_epochs=1000, compile=False, use_cuda=True): 

    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    x = x.to(device)
    y = y.to(device)

    if loss_fn == 'ce':
        out_channels = y.unique().view(-1).size(0) 
    else: 
        out_channels = y.size(1)

    model = MCBN(in_channels=x.size(1), 
                out_channels=out_channels, 
                **model_kwargs).to(device)
    
    if compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn == 'mse': 
        crit = torch.nn.MSELoss()
    elif loss_fn == 'l1': 
        crit = torch.nn.SmoothL1Loss()
    elif loss_fn == 'ce': 
        crit = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('unrecognized loss string, options: [mse, l1]')

    losses = []
    for i in range(num_epochs): 

        batch_loss = []
        for idx in torch.split(torch.randperm(x.size(0)), batch_size): 
            if (len(idx)) < 2: continue
            optim.zero_grad()
            yhat = model(x[idx])
            if loss_fn in ['mse', 'l1']: 
                loss = crit(yhat, y[idx])
            elif loss_fn == 'ce':
                loss = crit(yhat, y[idx].view(-1))
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())
        losses.append(np.mean(batch_loss))

        print(f'progress: {i}/{num_epochs} --> loss: {losses[-1]:.3f}', end='\r')

    return model.cpu(), losses