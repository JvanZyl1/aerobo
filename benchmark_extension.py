import numpy as np
import torch 

class extend_benchmark_with_random_constraints:
    """
    Extend a given benchmark problem with additional input dimensions and random independent constraints
    on the new dimensions while ensuring that the original optimum and feasible region remain unchanged.

    Parameters:
    -----------
    benchmark_problem : dict
        A dictionary representing the original benchmark problem with the following keys:
        - 'f': callable, the original objective function f(y)
        - 'c': callable or list of callables, the original constraints c(y) ≤ 0 (can be vector-valued)
        - 'dim_input': int, the original input dimensionality d
        - 'num_constraints': int, the original number of constraints m
    
    D_extra : int
        The number of extra input dimensions to add (artificial dimensions).
    
    m_extra : int
        The number of new random constraints to add in the extra dimensions.
    
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    extended_problem : dict
        A dictionary with the extended problem that has the same keys as the original benchmark problem but
        with extra input dimensions and constraints.
        - 'f_extended': callable, the extended objective function f(Px)
        - 'c_extended': callable or list of callables, the extended constraints
        - 'dim_input': int, the extended input dimensionality (d + D_extra)
        - 'num_constraints': int, the extended number of constraints (m + m_extra)
    """
    def __init__(
            self,
            M, # original model
            D=None, # target dimensionality 
            G=None, # target number of constraints
            c_latent=1, # number of independent constraints
            nl_subspace = False,
            seed=42
            ):
        
        self.M = M
        self.nl_subspace = nl_subspace
        self.eval_orig_objective = M.eval_objective
        self.eval_orig_constraints = M.eval_constraints
        

        self.g = M.num_constraints
        self.d = M.dim
        self.name = M.name
        self.best_value = M.best_value
        self.G = G

        if D is not None: 
            d_extra = D - self.d
            # Generate random projection matrix for input (to ensure input space extension is irrelevant to original problem)
            self.P = torch.eye(self.d,D,dtype=float)  # Input projection matrix: Identity matrix ensuring no loss in original dimensions
        else: 
            self.P = torch.eye(self.d,self.d,dtype=float)
        
        if G is not None: 
            g_extra = G - self.g
            if D is not None: 
                # Generate new random constraints on the extra dimensions
                self.A_basis = torch.tensor(np.random.randn(c_latent, d_extra))  # Random matrix for linear constraints (shape: m_extra x D_extra)
                self.b_basis = torch.tensor(np.random.rand(c_latent) + np.array(self.M.ub.max()))  # Offset vector for constraints to ensure a feasible region that includes the origin
                self.lin_comb = torch.tensor(np.random.randn(g_extra, c_latent)) # random coeffs for combination
                self.A = self.lin_comb @ self.A_basis
                self.b = self.lin_comb @ self.b_basis
            else:             
                self.A = torch.tensor(np.random.randn(g_extra, self.d))  # Random matrix for linear constraints (shape: m_extra x D_extra)
                self.b = torch.tensor(np.random.rand(g_extra) + np.array(self.M.ub.max()))  # Offset vector for constraints to ensure a feasible region that includes the origin

        self.eval_constraints = self.c_extended
        self.eval_objective = self.f_extended
        self.dim = D
        self.num_constraints = self.g if G is None else G

    # Extended objective function (works in the high-dimensional input space)
    def f_extended(self,x):
        # Project x to the original input space by using the first d dimensions of x
        x_proj = self.P @ x
        return self.eval_orig_objective(x_proj)     # Call the original objective function on the projected input
        # Extended constraints
    
    def c_extended(self,x):
        # Project x to the original input space
        x_proj = self.P @ x
        # Apply the original constraints to the projected input
        original_constraints = self.eval_orig_constraints(x_proj)
        
        if self.G is not None:
            # New random constraints applied to the extra dimensions
            x_extra = x[self.d:]  # Extract the extra dimensions
            if self.nl_subspace: x_extra = self.nonlinear_transform(x_extra)

            new_constraints = self.A @ x_extra - self.b - 1e3   # Linear constraints on the extra dimensions: A * x_extra <= b

            return torch.concatenate([original_constraints, new_constraints])
        else: 
            return original_constraints


        # Combine the original constraints with the new random constraints
    
    def nonlinear_transform(self,x):
        return torch.sin(torch.tensor(x)) + torch.tensor(x)**2
        


