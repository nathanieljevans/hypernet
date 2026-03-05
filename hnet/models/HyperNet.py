import torch
from hnet.models.RealNVP import RealNVP

class HyperNet(torch.nn.Module): 
    def __init__(self, model, 
                       stochastic_channels=8, 
                       width=10, 
                       cond_dim=0,
                       nonlin='elu', 
                       dropout=0, 
                       norm='none',
                       bias=False,
                       affine=False,
                       init_dict=None, 
                       pz='normal',
                       learn_pz=False, 
                       nvp_kwargs={'hidden_dim': 64, 
                                   'num_layers': 8, 
                                   'nonlin': 'elu', 
                                   'mask_type': 'alternating'}): 
        super().__init__()

        self.model = model 
        nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])

        state_idx_dict = {}
        state_size_dict = {}
        offset = 0 
        #for name, value in model.state_dict().items(): 
        for name, value in model.named_parameters(): 
            #if not value.requires_grad: continue
            n = value.numel()
            state_idx_dict[name] = torch.arange(offset, offset+n)
            state_size_dict[name] = value.size()
            offset += n 
        self.state_idx_dict = state_idx_dict
        self.state_size_dict = state_size_dict

        nonlin_map = {
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU,
            'gelu': torch.nn.GELU,
            'mish': torch.nn.Mish,
            'tanh': torch.nn.Tanh
        }
        nonlin = nonlin_map[nonlin]

        norm_map = {
            'none': torch.nn.Identity,
            'layer': lambda: torch.nn.LayerNorm(width),
        }

        in_channels = stochastic_channels + cond_dim

        self.f_phi = torch.nn.Sequential(torch.nn.Linear(in_channels, width, bias=bias), 
                                         norm_map[norm](),
                                         nonlin(), 
                                         torch.nn.Dropout(dropout), 
                                         torch.nn.Linear(width, width, bias=bias), 
                                         norm_map[norm](),
                                         nonlin(), 
                                         torch.nn.Dropout(dropout),
                                         torch.nn.Linear(width, nparams, bias=bias) )

        
        self.register_buffer('mu', torch.zeros((stochastic_channels), requires_grad=False))
        self.register_buffer('std', torch.ones((stochastic_channels), requires_grad=False))

        if learn_pz: 
            self.normalizing_flow = RealNVP(input_dim=stochastic_channels, 
                                            hidden_dim=nvp_kwargs['hidden_dim'], 
                                            num_layers=nvp_kwargs['num_layers'], 
                                            nonlin=nvp_kwargs['nonlin'], 
                                            mask_type=nvp_kwargs['mask_type'])
        else: 
            self.normalizing_flow = None 

        self.affine = affine
        if affine: self.scale = torch.nn.Parameter(torch.ones((nparams)), requires_grad=True)

        self.init_dict = init_dict
        self.pz = pz 

    def sample_theta_(self, ret_z=False, C=None): 

        if self.pz == 'normal': 
            m = torch.distributions.Normal(self.mu, self.std)
            z = m.sample()
        elif self.pz == 'uniform': 
            m = torch.distributions.Uniform(self.mu, self.std)
            z = m.sample()
        elif self.pz == 'bernoulli': 
            m = torch.distributions.Bernoulli(probs=0.5*torch.ones_like(self.mu))
            z = m.sample()
        elif self.pz == 'categorical': 
            m = torch.distributions.Categorical(probs=0.5*torch.ones_like(self.mu))
            z = m.sample()
        else: 
            raise ValueError(f'Invalid pz: {self.pz}')
        
        if self.normalizing_flow is not None: 
            z = self.normalizing_flow(z) 

        if C is not None:
            z = torch.cat([z, C], dim=-1)

        theta = self.f_phi(z) 

        if ret_z: 
            return theta,z 
        else: 
            return theta 

    def sample(self, C=None): 
        '''
        init_dict, {param_name->(mean,var)}
        '''
        theta = self.sample_theta_(C=C) 

        if self.affine: 
            theta = theta * self.scale

        state_dict = {n:theta[idx].view(self.state_size_dict[n]) for n,idx in self.state_idx_dict.items()}

        # DEV
        if self.init_dict is not None: 
            for name, (mu, var) in self.init_dict.items(): 
                if name in state_dict: 
                    state_dict[name] = state_dict[name] * torch.sqrt(var) + mu

        return state_dict

    def diversity_loss(self, p=2): 

        theta1, z1 = self.sample_theta_(ret_z=True)
        theta2, z2 = self.sample_theta_(ret_z=True)

        w = torch.cdist(z1.unsqueeze(0), z2.unsqueeze(0), p=p)

        loss = torch.nn.functional.cosine_similarity(theta1.view(1,-1), theta2.view(1,-1)) 

        loss = torch.abs(loss)

        return w*loss
        



        
    def forward(self, x, samples=10, C=None):

        def sample_(x):
            state_dict = self.sample(C=C)
            return torch.func.functional_call(self.model, state_dict, x)

        return torch.func.vmap(sample_, 
                               in_dims=0, 
                               randomness='different')(x.unsqueeze(0).expand(samples, -1, -1))

 