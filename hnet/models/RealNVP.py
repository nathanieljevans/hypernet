import torch
import torch.nn as nn
import numpy as np


class CouplingLayer(nn.Module):
    """
    Simplified coupling layer for transforming standard normal to complex distribution.
    Applies z = x * exp(s(x_masked)) + t(x_masked) where x~N(0,1).
    """
    def __init__(self, input_dim, hidden_dim, mask, nonlin='relu'):
        super().__init__()
        self.register_buffer('mask', mask)
        
        nonlin_map = {
            'relu': nn.ReLU,
            'elu': nn.ELU, 
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU
        }
        activation = nonlin_map[nonlin]()
        
        # Scale network - outputs log-scale for numerical stability
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        # Translation network
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim), 
            activation,
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize to near-identity transform
        self._init_identity()
    
    def _init_identity(self):
        """Initialize networks to output near zero (identity transform)."""
        # Scale network: initialize to output zeros (exp(0) = 1)
        nn.init.zeros_(self.scale_net[-2].weight)
        nn.init.zeros_(self.scale_net[-2].bias)
        
        # Translation network: initialize to output zeros
        nn.init.zeros_(self.translate_net[-1].weight)
        nn.init.zeros_(self.translate_net[-1].bias)
    
    def forward(self, x):
        """
        Transform x through coupling layer: z = x * exp(s) + t
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            z: Transformed tensor (batch_size, input_dim)
        """
        # Mask input for conditioning
        x_masked = x * self.mask
        
        # Compute scale and translation
        s = self.scale_net(x_masked)  # log-scale
        t = self.translate_net(x_masked)
        
        # Apply mask to transformations (only transform unmasked dimensions)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        # Apply affine transformation: z = x * exp(s) + t
        z = x * torch.exp(s) + t
        
        return z


class RealNVP(nn.Module):
    """
    Simplified Real-valued Non-Volume Preserving flow for generating
    complex distributions from standard normal noise.
    
    Usage: z = RealNVP(x) where x ~ N(0,1) and z ~ p(z)
    """
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_layers=4,
                 nonlin='relu',
                 mask_type='alternating'):
        """
        Args:
            input_dim: Dimension of input/output
            hidden_dim: Hidden dimension for coupling networks
            num_layers: Number of coupling layers
            nonlin: Nonlinearity ('relu', 'elu', 'gelu', 'tanh', 'leaky_relu')
            mask_type: Masking pattern ('alternating', 'random')
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Create masks
        masks = self._create_masks(input_dim, num_layers, mask_type)
        
        # Create coupling layers
        self.layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim, mask, nonlin)
            for mask in masks
        ])
    
    def _create_masks(self, input_dim, num_layers, mask_type):
        """Create binary masks for coupling layers."""
        masks = []
        
        if mask_type == 'alternating':
            # Simple alternating pattern
            for i in range(num_layers):
                mask = torch.zeros(input_dim)
                mask[i % 2::2] = 1  # Every other dimension starting from i%2
                masks.append(mask)
                
        elif mask_type == 'random':
            # Fixed random masks for reproducibility
            torch.manual_seed(42)
            for i in range(num_layers):
                mask = torch.randint(0, 2, (input_dim,)).float()
                # Ensure both parts of mask are non-empty
                if mask.sum() == 0 or mask.sum() == input_dim:
                    mask = torch.zeros(input_dim)
                    mask[:input_dim//2] = 1
                masks.append(mask)
        
        return masks
    
    def forward(self, x):
        """
        Transform standard normal input to complex distribution.
        
        Args:
            x: Standard normal input (batch_size, input_dim) or (input_dim,)
        
        Returns:
            z: Transformed output with complex distribution
        """
        # Handle both batched and single sample inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        z = x
        for layer in self.layers:
            z = layer(z)
        
        if squeeze_output:
            z = z.squeeze(0)
        
        return z
    
    def sample(self, num_samples=1):
        """
        Generate samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated samples (num_samples, input_dim)
        """
        device = next(self.parameters()).device
        
        # Sample from standard normal
        x = torch.randn(num_samples, self.input_dim, device=device)
        
        # Transform through flow
        z = self.forward(x)
        
        return z if num_samples > 1 else z.squeeze(0)


class SimpleRealNVP(nn.Module):
    """
    Ultra-simplified RealNVP for quick experimentation.
    Just 2-4 coupling layers with minimal configuration.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        
        # Create simple alternating masks
        masks = []
        for i in range(num_layers):
            mask = torch.zeros(input_dim)
            mask[i % 2::2] = 1
            masks.append(mask)
        
        # Simple coupling layers
        self.layers = nn.ModuleList([
            self._make_coupling_layer(input_dim, hidden_dim, mask)
            for mask in masks
        ])
    
    def _make_coupling_layer(self, input_dim, hidden_dim, mask):
        """Create a single coupling layer."""
        
        class SimpleCoupling(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('mask', mask)
                
                # Simple networks
                self.scale_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Tanh()
                )
                
                self.shift_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, input_dim)
                )
                
                # Initialize to identity
                nn.init.zeros_(self.scale_net[-2].weight)
                nn.init.zeros_(self.scale_net[-2].bias)
                nn.init.zeros_(self.shift_net[-1].weight)
                nn.init.zeros_(self.shift_net[-1].bias)
            
            def forward(self, x):
                x_masked = x * self.mask
                s = self.scale_net(x_masked) * (1 - self.mask)
                t = self.shift_net(x_masked) * (1 - self.mask)
                return x * torch.exp(s) + t
        
        return SimpleCoupling()
    
    def forward(self, x):
        """Transform x ~ N(0,1) to z ~ p(z)."""
        # Handle single samples
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        z = x
        for layer in self.layers:
            z = layer(z)
        
        return z.squeeze(0) if squeeze_output else z
    
    def sample(self, num_samples=1):
        """Generate samples."""
        device = next(self.parameters()).device
        x = torch.randn(num_samples, self.input_dim, device=device)
        z = self.forward(x)
        return z if num_samples > 1 else z.squeeze(0)