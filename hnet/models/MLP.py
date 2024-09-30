import torch

class MLP(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels, layers, nonlin='relu', dropout='none', norm='none'): 
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

        if dropout == 'none': 
            do = torch.nn.Identity()
        else: 
            do = torch.nn.Dropout(dropout)
        
        seq = [torch.nn.Linear(in_channels, hidden_channels), norm(), do, nonlin()]
        for i in range(layers): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels), norm(), do, nonlin()]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        
        self.f = torch.nn.Sequential(*seq)

    def get_init_dict(self, init='infer'): 
        '''
        create a dictionary (param_name -> init_params (mu, var))
        '''
        init_dict = {}
        
        for name, param in self.named_parameters():
            if init == 'xavier': 
                # Only include weights and biases from Linear layers
                if 'weight' in name:
                    fan_in, fan_out = param.size(1), param.size(0)
                    # Xavier initialization variance
                    var = 2.0 / (fan_in + fan_out)
                    init_dict[name] = (0.0, var)
                elif 'bias' in name:
                    init_dict[name] = (0.0, 0.01)

            elif init == 'infer':
                if param.numel() > 2: 
                    init_dict[name] = (param.mean().detach().cpu(), torch.clamp(param.var().detach().cpu(), 1e-2, 10))


        return init_dict


    def forward(self, x): 
        return self.f(x)