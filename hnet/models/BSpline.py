
import torch 
import torch.nn as nn


class BSpline(torch.nn.Module): 
    """
    B-Spline neural network layer that learns control points and applies
    B-spline basis functions as activation functions.
    """

    def __init__(self, in_dim, out_dim, n_knots, degree): 
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_knots = n_knots
        self.degree = degree
        
        # Number of control points/basis functions
        self.n_control_points = n_knots - degree - 1
        
        # Learnable knot vector - we'll keep it fixed for now but could make it learnable
        # Standard approach: uniform knot vector with repeated end knots
        knots = torch.zeros(n_knots)
        for i in range(n_knots):
            if i <= degree:
                knots[i] = 0.0
            elif i >= n_knots - degree - 1:
                knots[i] = 1.0
            else:
                knots[i] = (i - degree) / (n_knots - 2 * degree - 1)
        
        # Register as buffer so it moves with the model but isn't trained
        self.register_buffer('knots', knots)
        
        # Learnable control points/coefficients for each input dimension and output dimension
        # Shape: (in_dim, n_control_points, out_dim)
        self.control_points = nn.Parameter(torch.randn(in_dim, self.n_control_points, out_dim))
        
        # Initialize control points with small random values
        #nn.init.xavier_uniform_(self.control_points, gain=0.1)

    def _compute_all_basis_functions(self, x, knots, degree):
        """
        Compute ALL B-spline basis functions at once using fully vectorized approach.
        This is much more efficient than computing each basis function individually.
        
        Args:
            x: input values (batch_size,)
            knots: knot vector
            degree: spline degree
            
        Returns:
            All basis function values (n_control_points, batch_size)
        """
        batch_size = x.shape[0]
        n_knots = len(knots)
        n_intervals = n_knots - 1
        
        # Initialize with degree 0 basis functions (indicator functions)
        # Handle boundary conditions properly for x=1
        knots_expanded = knots.unsqueeze(0)  # (1, n_knots)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1)
        
        # Create indicator functions for all intervals at once
        left_bounds = knots_expanded[..., :-1]  # (1, n_intervals)
        right_bounds = knots_expanded[..., 1:]   # (1, n_intervals)
        
        # Special handling for the last interval to include x=1
        in_interval = (x_expanded >= left_bounds) & (x_expanded < right_bounds)  # (batch_size, n_intervals)
        
        # For the last interval, include the right endpoint (x=1)
        last_interval_mask = (x_expanded >= left_bounds[..., -1:]) & (x_expanded <= right_bounds[..., -1:])
        in_interval = torch.cat([in_interval[..., :-1], last_interval_mask], dim=-1)
        
        basis_prev = in_interval.float().T  # (n_intervals, batch_size)
        
        # Iteratively compute higher degree basis functions
        for curr_degree in range(1, degree + 1):
            n_basis = n_intervals - curr_degree
            
            # Collect all terms in lists first (no in-place operations)
            left_terms = []
            right_terms = []
            
            for j in range(n_basis):
                # Left term
                denom_left = knots[j + curr_degree] - knots[j]
                if denom_left.abs() > 1e-8:  # Static check on knot values
                    left_coeff = (x - knots[j]) / denom_left
                    left_term = left_coeff * basis_prev[j]
                else:
                    left_term = torch.zeros_like(x)
                left_terms.append(left_term)
                
                # Right term
                denom_right = knots[j + curr_degree + 1] - knots[j + 1]
                if denom_right.abs() > 1e-8:  # Static check on knot values
                    right_coeff = (knots[j + curr_degree + 1] - x) / denom_right
                    right_term = right_coeff * basis_prev[j + 1]
                else:
                    right_term = torch.zeros_like(x)
                right_terms.append(right_term)
            
            # Stack and combine - all out-of-place operations
            left_stack = torch.stack(left_terms, dim=0)   # (n_basis, batch_size)
            right_stack = torch.stack(right_terms, dim=0) # (n_basis, batch_size)
            basis_prev = left_stack + right_stack          # (n_basis, batch_size)
        
        return basis_prev  # (n_control_points, batch_size)

    def forward(self, x):
        """
        Forward pass through B-spline layer.
        
        Args:
            x: Input tensor (batch_size, in_dim) or (in_dim,) for single sample
            
        Returns:
            Output tensor (batch_size, out_dim) or (out_dim,) for single sample
        """
        # Handle both 1D and 2D inputs
        if x.dim() == 1:
            # Single sample case: (in_dim,) -> (1, in_dim)
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        batch_size = x.shape[0]
        
        # Validate input dimensions
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input dimension {self.in_dim}, got {x.shape[1]}")
        
        # Clamp input to [0, 1] range for B-spline evaluation
        # Note: Removed assert for vmap compatibility - user should ensure inputs are in [0,1]
        x = torch.clamp(x, 0.0, 1.0)
        
        # Initialize output with proper shape for vmap compatibility
        output = torch.zeros(batch_size, self.out_dim, device=x.device, dtype=x.dtype)
        
        # For each input dimension
        for d in range(self.in_dim):
            x_d = x[:, d]  # (batch_size,)
            
            # Compute ALL basis functions at once for this input dimension
            all_basis_vals = self._compute_all_basis_functions(x_d, self.knots, self.degree)
            # all_basis_vals: (n_control_points, batch_size)
            
            # Vectorized computation of contributions
            # self.control_points[d, :, :]: (n_control_points, out_dim)
            # all_basis_vals: (n_control_points, batch_size)
            # We want: (batch_size, out_dim)
            
            # Use einsum for efficient tensor contraction
            contribution = torch.einsum('ij,ik->jk', all_basis_vals, self.control_points[d, :, :])
            # This computes: sum_i(basis_vals[i, j] * control_points[d, i, k]) for all j,k
            
            output = output + contribution  # (batch_size, out_dim)
        
        # Handle single sample case
        if single_sample:
            output = output.squeeze(0)  # (1, out_dim) -> (out_dim,)
            
        return output 
