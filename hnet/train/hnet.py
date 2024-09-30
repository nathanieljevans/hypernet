
import torch
from hnet.models.MLP import MLP 
from hnet.models.HyperNet import HyperNet
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


def init_hnet(model, init_dict, samples=100, iters=100, lr=1e-3, verbose=True): 
    '''
    pretrain the HyperNet weight initializations 
    
    Args: 
        model       HyperNet 
        init_dict   dict            {param_name -> (mean, var)}  ; assumes normal distribution
    
    Returns
        HyperNet; pretrained model such that theta is in in the right initialization ranges. 
    '''

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ii in range(iters): 

        optim.zero_grad()

        theta_dict = {k:[] for k in init_dict}
        for jj in range(samples): 
            state_dict = model.sample() 
            for k in theta_dict.keys(): theta_dict[k].append(state_dict[k])
        theta_dict = {k:torch.stack(v, dim=0).cpu() for k,v in theta_dict.items()}
        mean_dict = {k:v.mean(0) for k,v in theta_dict.items()}
        var_dict = {k:v.var(0) for k,v in theta_dict.items()}

        loss = torch.zeros((1,), dtype=torch.float32)
        for k, (mu, var) in init_dict.items(): 
            muhat = mean_dict[k]
            varhat = var_dict[k]

            kl_div = torch.log(var / varhat) + (varhat + (muhat - mu)**2) / (2 * var) - 0.5
            loss += kl_div.sum()

        loss.backward() 
        optim.step() 

        if verbose: print(f'iter: {ii} --> kl loss: {loss.item():.4f}', end='\r')

    return model
            

def train_hnet(x,y, mlp_kwargs, hnet_kwargs, loss_fn='mse', lr=1e-4, batch_size=124, num_epochs=1000, nsamples=1000, compile=False, use_cuda=True, pretrain_init=True): 

    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    x = x.to(device)
    y = y.to(device)

    if loss_fn == 'ce':
        out_channels = y.unique().view(-1).size(0) 
    else: 
        out_channels = y.size(1)

    mlp = MLP(in_channels=x.size(1), out_channels=out_channels, **mlp_kwargs)
    model = HyperNet(mlp, **hnet_kwargs).to(device)

    if pretrain_init: 
        print('initializing hypernet...')
        init_dict = mlp.get_init_dict()
        model = init_hnet(model, init_dict, samples=100, iters=100, lr=1e-3, verbose=True)
        print()
    
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
