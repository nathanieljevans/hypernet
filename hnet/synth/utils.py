import torch 
import numpy as np
from matplotlib import pyplot as plt 

def plot1d(x, y, y_true, true_lcb, true_ucb, x2=None, yhat=None, title='', plot_ci=True, ylim=None): 

    x = x.squeeze().detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy() 
    y_true = y_true.squeeze().detach().cpu().numpy()
    true_lcb = true_lcb.squeeze().detach().cpu().numpy()
    true_ucb = true_ucb.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(8, 5))

    if plot_ci: 
        plt.fill_between(x,
                        true_lcb,
                        true_ucb,
                        color='r', alpha=0.1, label='True CI')
    
    if (x2 is not None) and (yhat is not None): 
        x2 = x2.squeeze().detach().cpu().numpy()

        if plot_ci:
            pred_lcb = yhat.quantile(0.025, dim=0).squeeze().detach().cpu().numpy()
            pred_ucb = yhat.quantile(0.975, dim=0).squeeze().detach().cpu().numpy()
            yhat_mean = yhat.mean(dim=0).squeeze().detach().cpu().numpy() 
            plt.plot(x2, yhat_mean, 'b-', label='pred (mean)')
            plt.fill_between(x2,
                            pred_lcb,
                            pred_ucb,
                            color='blue', alpha=0.25, label='True CI')
        else: 
            alpha_ = np.clip(1/(yhat.size(0)/50), 0.001,1) 
            for yyhat in yhat.detach().cpu().numpy(): 

                plt.plot(x2, yyhat, 'b-', alpha=alpha_)
        
    plt.plot(x, y_true, 'r-', label='true')
    plt.plot(x, y, 'k.', label='data', alpha=0.25)

    if ylim is not None: 
        plt.ylim(ylim)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.show()


def plot1d_addition(x, y, x_part, y_true_1, y_true_2, true_lcb_1, true_ucb_1, true_lcb_2, true_ucb_2, 
                    title='', plot_ci=True, ylim=None): 
    
    # Convert tensors to numpy for plotting
    x = x.squeeze().detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()

    x_part = x_part.squeeze().detach().cpu().numpy()  # half of x for each function
    y_true_1 = y_true_1.squeeze().detach().cpu().numpy()  # Only plot the first output (no noise)
    true_lcb_1 = true_lcb_1.squeeze().detach().cpu().numpy()
    true_ucb_1 = true_ucb_1.squeeze().detach().cpu().numpy()

    y_true_2 = y_true_2.squeeze().detach().cpu().numpy()
    true_lcb_2 = true_lcb_2.squeeze().detach().cpu().numpy()
    true_ucb_2 = true_ucb_2.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(8, 5))

    if plot_ci:
        # Plot confidence interval for the first function
        plt.fill_between(x_part, true_lcb_1, true_ucb_1, color='r', alpha=0.1, label='True CI (f1)')
        # Plot confidence interval for the second function
        plt.fill_between(x_part, true_lcb_2, true_ucb_2, color='g', alpha=0.1, label='True CI (f2)')
    
    # Plot the true functions
    plt.plot(x_part, y_true_1, 'r-', label='true (f1)')
    plt.plot(x_part, y_true_2, 'g-', label='true (f2)')

    # Plot the noisy data points (y)
    plt.plot(x, y, 'k.', label='data', alpha=0.25)

    # Set y-axis limits if provided
    if ylim is not None: 
        plt.ylim(ylim)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.show()
