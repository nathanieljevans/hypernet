
import torch
from GaNN.deprecated.GaNN import GaNN
from GaNN.models.MCDO import MCDO
from GaNN.train.mcbn import train_mcbn
from GaNN.train.mcdo import train_mcdo
from GaNN.train.hnet import train_hnet
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_diabetes, fetch_california_housing
from torchvision import datasets, transforms
from GaNN.train.utils import expected_calibration_error

def run_sklearn_dataset(dataset_name, model_name, mlp_kwargs, train_kwargs, hnet_kwargs=None, seed=42):
    
    if dataset_name in ['diabetes', 'california']: 
        if dataset_name == "diabetes":
            data = load_diabetes(scaled=False)
        elif dataset_name == "california":
            data = fetch_california_housing()
        
        X, y = data.data.astype(np.float32), data.target.astype(np.float32)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        x_mu = X_train.mean(0)
        x_std = X_train.std(0)
        X_train = (X_train - x_mu)/(x_std + 1e-8)
        X_test = (X_test - x_mu)/(x_std + 1e-8)

        y_mu = y_train.mean(0)
        y_std = y_train.std(0)
        y_train = (y_train - y_mu)/(y_std + 1e-8)
        y_test = (y_test - y_mu)/(y_std + 1e-8)

        task = 'regression'

    elif dataset_name == 'mnist':  

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        X_train = train_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32)
        y_train = train_dataset.targets.numpy().astype(int)
        X_test = test_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32)
        y_test = test_dataset.targets.numpy().astype(int)

        task = 'classification'

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)  # Convert to column vector
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    if model_name == 'hnet': 
        model, losses = train_hnet(X_train_tensor, y_train_tensor, mlp_kwargs=mlp_kwargs, hnet_kwargs=hnet_kwargs, **train_kwargs)
        
        with torch.no_grad():
            model.eval()
            y_pred = []
            for idx in torch.split(torch.arange(X_test_tensor.size(0)), train_kwargs['batch_size']): 
                y_pred.append( model(X_test_tensor[idx], 1000) ) 
            y_pred = torch.cat(y_pred, dim=1)
            y_pred_mu = y_pred.mean(0)
            y_pred_var = y_pred.var(0)
            y_true = y_test_tensor

    elif model_name == 'mcdo': 
        model, losses = train_mcdo(X_train_tensor, y_train_tensor, model_kwargs=mlp_kwargs, **train_kwargs)
    
        with torch.no_grad():
            model.train() # make sure do is active
            y_pred = []
            for idx in torch.split(torch.arange(X_test_tensor.size(0)), train_kwargs['batch_size']): 
                y_pred.append( torch.stack([model(X_test_tensor[idx]) for _ in range(1000)], dim=0) ) 
            y_pred = torch.cat(y_pred, dim=1)
            y_pred_mu = y_pred.mean(0)
            y_pred_var = y_pred.var(0)
            y_true = y_test_tensor
        
    elif model_name == 'mcbn': 
        model, losses = train_mcbn(X_train_tensor, y_train_tensor, model_kwargs=mlp_kwargs, **train_kwargs)
        
        with torch.no_grad():
            model.train() # make sure do is active
            y_pred = []
            for idx in torch.split(torch.arange(X_test_tensor.size(0)), train_kwargs['batch_size']): 
                y_pred.append( model.predict(X_test_tensor[idx], nsamples=1000, batch_size=train_kwargs['batch_size']) ) 
            y_pred = torch.cat(y_pred, dim=1)
            y_pred_mu = y_pred.mean(0)
            y_pred_var = y_pred.var(0)
            y_true = y_test_tensor

    else:
        raise NotImplementedError('unrecognized model name')
    
    if task == 'regression': 
        nll = torch.nn.functional.gaussian_nll_loss(y_pred_mu, y_true, y_pred_var).item()
        y_pred = y_pred_mu.detach().cpu().numpy().ravel()
        y_true = y_true.detach().cpu().numpy().ravel()
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {'mse':mse, 'r2':r2, 'nll':nll}
    elif task == 'classification':
        ce = torch.nn.functional.cross_entropy(y_pred_mu, y_true.view(-1)).item()
        acc = (y_true.view(-1) == y_pred_mu.argmax(dim=-1).view(-1)).float().mean().item()
        ece = expected_calibration_error(y_pred_mu, y_true, n_bins=10)
        return {'ce':ce, 'acc':acc, 'ece':ece}
