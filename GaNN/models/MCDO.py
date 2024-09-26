import torch
import torch.nn.functional

class MCDO(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels, layers, dropout=0.25, nonlin='relu', norm='none'): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        self.layers = layers

        if nonlin == 'relu':
            nonlin = torch.nn.ReLU
        elif nonlin == 'elu': 
            nonlin = torch.nn.ELU
        elif nonlin == 'gelu': 
            nonlin = torch.nn.GELU
        elif nonlin == 'mish': 
            nonlin = torch.nn.Mish
        elif nonlin == 'tanh': 
            nonlin = torch.nn.Tanh
        else: 
            raise NotImplementedError('unrecognized nonlin string')
        
        if norm == 'none':
            norm_ = torch.nn.Identity
        elif norm == 'batch': 
            norm_ = lambda: torch.nn.BatchNorm1d(hidden_channels, affine=True)
        else: 
            raise NotImplementedError('unrecognized norm string')
        
        seq = [torch.nn.Linear(in_channels, hidden_channels), norm_(), nonlin(), torch.nn.Dropout(dropout)]
        for i in range(layers): seq += [torch.nn.Linear(hidden_channels, hidden_channels), norm_(), nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        self.f = torch.nn.Sequential(*seq)

    def forward(self, x): 
        
        x = self.f(x)
        return x

        

