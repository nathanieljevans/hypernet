import torch

class MLP(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels, layers, nonlin='relu', norm='none'): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        self.layers = layers

        nonlin_map = {
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU,
            'gelu': torch.nn.GELU,
            'mish': torch.nn.Mish,
            'tanh': torch.nn.Tanh
        }
        nonlin = nonlin_map[nonlin]

        norm_map = {
            'none': lambda: torch.nn.Identity(),
            'batch':lambda: torch.nn.BatchNorm1d(hidden_channels),
            'layer':lambda: torch.nn.LayerNorm(hidden_channels), 
            'group':lambda: torch.nn.GroupNorm(num_groups=hidden_channels//5, num_channels=hidden_channels), 
        }
        norm = norm_map[norm]
        
        seq = [torch.nn.Linear(in_channels, hidden_channels), norm(), nonlin()]
        for i in range(layers): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels), norm(), nonlin()]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        
        self.f = torch.nn.Sequential(*seq)

    def forward(self, x): 
        return self.f(x)