
import torch 


def simple1d(N=1000, scale=3): 

    def f(x, noise=False):
        if noise: 
            return x*torch.sin(x) + scale*torch.randn_like(x)*torch.cos(x/2), x*torch.sin(x) - scale*1.96*torch.cos(x/2), x*torch.sin(x) + scale*1.96*torch.cos(x/2)
        else: 
            return x*torch.sin(x)

    x = torch.linspace(-5,10,N)
    y_true = f(x)
    y, true_lcb, true_ucb = f(x, noise=True)

    return x,y,y_true,true_lcb,true_ucb


def simple1d_addition(N=1000, scale=0.1):

    def f1(x, noise=False, amp=0.1, phase=0, bias=0):
        if noise:
            return amp*torch.sin(x + phase) + bias + scale*torch.randn_like(x), amp*torch.sin(x + phase) + bias - scale*torch.ones_like(x), amp*torch.sin(x + phase) + bias + scale*torch.ones_like(x)
        else:
            return amp*torch.sin(x + phase) + bias 

    x = torch.linspace(-6, 6, N//2)
    
    y_true_1 = f1(x)
    y_true_2 = f1(x, phase=3.14, bias=1)
    
    y1, true_lcb1, true_ucb1 = f1(x, noise=True)
    y2, true_lcb2, true_ucb2 = f1(x, noise=True, phase=3.14, bias=1)

    Y = torch.cat((y1,y2), dim=0)
    X = torch.cat((x,x), dim=0)
    
    return X, Y, x, y_true_1, y_true_2, true_lcb1, true_ucb1, true_lcb2, true_ucb2