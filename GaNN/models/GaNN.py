import torch


def init_(in_channels, out_channels, hidden_channels, layers):

        theta_size = (in_channels*hidden_channels + hidden_channels) + \
                     (hidden_channels*hidden_channels + hidden_channels)*layers + \
                     (hidden_channels*out_channels + out_channels)
        

        W_indices = [torch.arange(in_channels*hidden_channels)]
        offset = in_channels*hidden_channels
        for l in range(layers): 
            W_indices.append( torch.arange(offset, offset + hidden_channels*hidden_channels) )
            offset += hidden_channels*hidden_channels
        W_indices.append(torch.arange(offset, offset + hidden_channels*out_channels))
        offset += hidden_channels*out_channels 


        bias_indices = [] 
        for i in range(layers+1): 
            bias_indices.append( torch.arange(offset, offset + hidden_channels))
            offset += hidden_channels 
        bias_indices += torch.arange(offset, offset + out_channels)
        offset += out_channels

        assert theta_size == offset, f'expected theta_size ({theta_size}) to be the same as final offset ({offset})'

        return theta_size, W_indices, bias_indices

        
class GaNN(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels, layers, gaussian_channels, width, nonlin='relu'): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.gaussian_channels =gaussian_channels
        self.width = width

        if nonlin == 'relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlin == 'elu': 
            self.nonlin = torch.nn.ELU()
        elif nonlin == 'gelu': 
            self.nonlin = torch.nn.GELU()
        elif nonlin == 'mish': 
            self.nonlin = torch.nn.Mish()
        elif nonlin == 'tanh': 
            self.nonlin = torch.nn.Tanh()
        else: 
            raise NotImplementedError('unrecognized nonlin string')

        theta_size, W_indices, bias_indices = init_(in_channels, out_channels, hidden_channels, layers)

        self.W_sizes = [(in_channels, hidden_channels)] + [(hidden_channels, hidden_channels) for _ in range(layers)] + [(hidden_channels, out_channels)]
        
        print(bias_indices)
        for i,Widx in enumerate(W_indices): self.register_buffer(f'W{i}', Widx)
        for i,Bidx in enumerate(bias_indices): self.register_buffer(f'B{i}', Bidx)
        
        self.f_phi = torch.nn.Sequential(torch.nn.Linear(gaussian_channels, width, bias=False), 
                                         torch.nn.ReLU(), 
                                         torch.nn.Linear(width, theta_size, bias=False))

        self.register_buffer('mu', torch.zeros((gaussian_channels), requires_grad=False))
        self.register_buffer('std', torch.ones((gaussian_channels), requires_grad=False))

    def forward(self, x, samples=1): 
        x = x.T.unsqueeze(0).expand(samples, -1, -1) # x size: (batch, in_channels)

        ## thought: we could use the reparameterization trick to learn mu, std... not all functions may be useful? 
        ## thought: could also use bernoulli dist instead of normal...
        m = torch.distributions.Normal(self.mu, self.std)
        z = m.sample((samples,))
        theta = self.f_phi(z)

        for i in range(self.layers + 1): 
            W = theta[:, getattr(self, f'W{i}')].view(-1, *self.W_sizes[i]) # size (samples, channels_in, channels_out)
            B = theta[:, getattr(self, f'B{i}')].unsqueeze(2)               # size (samples, 1, channels_out)
            x = torch.matmul(W.permute(0,2,1), x) + B                                    # size (samples, )
            x = self.nonlin(x)

        print(self.layers)
        print(dir(self))
        
        print(theta[:, getattr(self, f'B{self.layers+1}')].size())
        W = theta[:, getattr(self, f'W{self.layers+1}')].view(-1, *self.W_sizes[self.layers+1])
        B = theta[:, getattr(self, f'B{self.layers+1}')].view(-1, self.out_channels,1)
        print(W.size())
        print(x.size())
        print(B.size())
        x = torch.matmul(W.permute(0,2,1), x) + B
        x = x.permute(0,2,1)
        return x
