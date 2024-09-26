import torch
import torch.nn.functional as F

class MCBN(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels, layers, nonlin='relu', mc_dropout=True): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.mc_dropout = mc_dropout

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
        
        # Sequential layers with BatchNorm and Dropout
        seq = [torch.nn.Linear(in_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), nonlin()]
        for i in range(layers): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.BatchNorm1d(hidden_channels), nonlin()]
        seq += [torch.nn.Linear(hidden_channels, out_channels)]
        
        self.f = torch.nn.Sequential(*seq)

    def forward(self, x, mc_bn=True): 
        if mc_bn:
            # Keep BatchNorm layers in train mode during inference
            self.train()
        else:
            self.eval()
        x = self.f(x)
        return x

    def predict(self, x, nsamples=10, batch_size=125):

        out = torch.zeros((nsamples, x.size(0), self.out_channels))
        for i in range(nsamples):
                
            for idx in torch.split(torch.randperm(x.size(0)), batch_size): 
                if (len(idx)) < 2: continue

                xx = x[idx]

                with torch.no_grad():
                    out[i, idx, :] = self.forward(xx, mc_bn=True)

        return out