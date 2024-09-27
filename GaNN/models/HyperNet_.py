'''
This is a looped version of HyperNet and it is significantly slower than it's vectorized counterpart (HyperNet), however, vmap does not handle 
BatchNorm appropriately. We are keeping this version as a backup for cases when we need to use HyperNets with modules that utilize BatchNorm.
'''

import torch

class HyperNet_(torch.nn.Module): 
    def __init__(self, model, stochastic_channels=8, width=10, nonlin='relu'): 
        super().__init__()
        print("warning: `HyperNet_` is a looped version appropriate for modules that utilize `BatchNorm`, however, the vectorized version (`HyperNet`) is significantly faster.")

        self.model = model 
        nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])

        state_idx_dict = {}
        state_size_dict = {}
        offset = 0 
        for name, value in model.state_dict().items(): 
            if value.requires_grad == False: continue
            n = value.numel()
            state_idx_dict[name] = torch.arange(offset, offset+n)
            state_size_dict[name] = value.size()
            offset += n 
        self.state_idx_dict =state_idx_dict
        self.state_size_dict = state_size_dict
        
        nonlin_map = {
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU,
            'gelu': torch.nn.GELU,
            'mish': torch.nn.Mish,
            'tanh': torch.nn.Tanh
        }
        nonlin = nonlin_map[nonlin]

        self.f_phi = torch.nn.Sequential(torch.nn.Linear(stochastic_channels, width, bias=False), 
                                         nonlin(), 
                                         torch.nn.Linear(width, nparams, bias=False) )
        
        self.register_buffer('mu', torch.zeros((stochastic_channels), requires_grad=False))
        self.register_buffer('std', torch.ones((stochastic_channels), requires_grad=False))

    def forward(self, x, samples=10):
        
        out = [] 
        for _ in range(samples): 
            m = torch.distributions.Normal(self.mu, self.std)
            z = m.sample()
            theta = self.f_phi(z)
            state_dict = {n:theta[idx].view(self.state_size_dict[n]) for n,idx in self.state_idx_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            out.append( self.model(x))
        return torch.stack(out, dim=0)
    