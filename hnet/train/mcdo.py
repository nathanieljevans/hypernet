
import torch
from hnet.models.MCDO import MCDO
import numpy as np

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
