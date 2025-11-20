
import math
import os
import warnings
from dataclasses import dataclass

import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, VariationalStrategy
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.optim import optimize_acqf
from gpytorch.models.exact_gp import ExactGP

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP
from botorch.utils.transforms import standardize, unnormalize, normalize
from botorch.generation.utils import _flip_sub_unique
from botorch.utils.sampling import batched_multinomial
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import Interval
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import VariationalELBO
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
import numpy as np

from sampling import *
from models_aerobo import *
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}
max_cholesky_size = float("inf")  # Always use Cholesky

SMOKE_TEST = os.environ.get("SMOKE_TEST")

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed) #, seed=seed
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def compute_doe(model, n_init):
    # Get initial data 
    # Must get initial values for both objective and constraints 
    train_X = get_initial_points(model.dim, n_init)
    train_Y = torch.tensor([], dtype=dtype, device=device)
    train_C = torch.tensor([], dtype=dtype, device=device)

    for x in train_X:
        y = model.eval_objective(x)
        c = model.eval_constraints(x)
        train_Y = torch.cat((train_Y,y.unsqueeze(-1)),dim=0)
        train_C = torch.cat((train_C,c.unsqueeze(-1)),dim=1) 

    train_Y = train_Y.unsqueeze(-1)
    train_C = train_C.T

    return train_X, train_Y, train_C

@dataclass
class ScboState:
    dim: int
    batch_size: int
    num_constraints: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        #self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / (1*self.batch_size)]))
        self.failure_tolerance = math.ceil(max([20.0 / self.batch_size, float(self.dim) / (1*self.batch_size)]))
        self.best_constraint_values = torch.ones(self.num_constraints,)*torch.inf


class standard_scbo:

    def __init__(
            self,
            model
            ):
        self.model = model

    def update_tr_length(self, state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        if state.length < state.length_min:  # Restart when trust region becomes too small
            state.restart_triggered = True

        return state

    def update_state(self, state, Y_next, C_next):
        """Method used to update the TuRBO state after each step of optimization.

        Success and failure counters are updated according to the objective values
        (Y_next) and constraint values (C_next) of the batch of candidate points
        evaluated on the optimization step.

        As in the original TuRBO paper, a success is counted whenver any one of the
        new candidate points improves upon the incumbent best point. The key difference
        for SCBO is that we only compare points by their objective values when both points
        are valid (meet all constraints). If exactly one of the two points being compared
        violates a constraint, the other valid point is automatically considered to be better.
        If both points violate some constraints, we compare them inated by their constraint values.
        The better point in this case is the one with minimum total constraint violation
        (the minimum sum of constraint values)
        """

        # Pick the best point from the batch
        best_ind = self.get_best_index_for_batch(Y=Y_next, C=C_next)
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            # At least one new candidate is feasible
            improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
            if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1
        else:
            # No new candidate is feasible
            total_violation_next = c_next.clamp(min=0).sum(dim=-1)
            total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
            if total_violation_next < total_violation_center:
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1

        # Update the length of the trust region according to the success and failure counters
        state = self.update_tr_length(state)
        
        return state

    def update_state_unconstrained(self, state, Y_next):
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

    def get_initial_points(self, dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init

    def get_best_index_for_batch(self, Y: Tensor, C: Tensor):
        """Return the index for the best point."""
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():  # Choose best feasible candidate
            score = Y.clone()
            score[~is_feas] = -float("inf")
            return score.argmax()
        return C.clamp(min=0).sum(dim=-1).argmin()
    
    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        C,  # Constraint values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
        sobol: SobolEngine,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        # Create the TR bounds
        best_ind = self.get_best_index_for_batch(Y=Y, C=C)
        x_center = X[best_ind, :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO)
        dim = X.shape[-1]
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
        )
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        return X_next
    
    def generate_batch_unconstrained(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        sobol,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ts",  # "ei" or "ts"
    ):
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next
    
    def compute_doe(self, n_init):
        # Get initial data 
        # Must get initial values for both objective and constraints 
        train_X = self.get_initial_points(self.model.dim, n_init)
        train_Y = torch.tensor([], dtype=dtype, device=device)
        train_C = torch.tensor([], dtype=dtype, device=device)

        for x in train_X:
            y = self.model.eval_objective(x)
            c = self.model.eval_constraints(x)
            train_Y = torch.cat((train_Y,y.unsqueeze(-1)),dim=0)
            train_C = torch.cat((train_C,c.unsqueeze(-1)),dim=1) 

        train_Y = train_Y.unsqueeze(-1)
        train_C = train_C.T

        return train_X, train_Y, train_C
    
    def get_fitted_model(self, X, Y):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=self.model.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)

        test = model(X)
        test = test.variance
        return model

    def gaussian_copula_transform(self,data):
        """
        This procedure magnifies differences between values that are at 
        the end of the observed range, i.e., minima or maxima. It affects 
        the observed values but not their location.
        Eriksson, et al.; Scalable Constrained Bayesian Optimisation
        """
        data = data.squeeze(1)
        # Calculate the ranks of the data
        ranks = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data)
        # Transform ranks to uniform [0, 1] using the inverse CDF of the standard normal distribution
        uniform_values = norm.ppf((ranks + 1) / (data.shape[0] + 1))
        # Create a covariance matrix from the standardized data
        covariance = np.cov(uniform_values, rowvar=False)
        # Transform the uniform values to multivariate normal using the Cholesky decomposition
        transformed_data = uniform_values * covariance
        
        return torch.tensor(transformed_data).unsqueeze(-1)

    def bilog_transformation(self,data): 
        return data.sign()*(data.abs()+1).log()    


class aerobo_scbo:

    def __init__(
            self,
            model
            ):
        self.model = model

    def update_tr_length(self, state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        if state.length < state.length_min:  # Restart when trust region becomes too small
            state.restart_triggered = True

        return state

    def update_state(self, state, Y_next, C_next):
        """Method used to update the TuRBO state after each step of optimization.

        Success and failure counters are updated according to the objective values
        (Y_next) and constraint values (C_next) of the batch of candidate points
        evaluated on the optimization step.

        As in the original TuRBO paper, a success is counted whenver any one of the
        new candidate points improves upon the incumbent best point. The key difference
        for SCBO is that we only compare points by their objective values when both points
        are valid (meet all constraints). If exactly one of the two points being compared
        violates a constraint, the other valid point is automatically considered to be better.
        If both points violate some constraints, we compare them inated by their constraint values.
        The better point in this case is the one with minimum total constraint violation
        (the minimum sum of constraint values)
        """

        # Pick the best point from the batch
        best_ind = self.get_best_index_for_batch(Y=Y_next, C=C_next)
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            # At least one new candidate is feasible
            improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
            if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1
        else:
            # No new candidate is feasible
            total_violation_next = c_next.clamp(min=0).sum(dim=-1)
            total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
            if total_violation_next < total_violation_center:
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1

        # Update the length of the trust region according to the success and failure counters
        state = self.update_tr_length(state)
        
        return state

    def get_best_index_for_batch(self, Y: Tensor, C: Tensor):
        """Return the index for the best point."""
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():  # Choose best feasible candidate
            score = Y.clone()
            score[~is_feas] = -float("inf")
            return score.argmax()
        return C.clamp(min=0).sum(dim=-1).argmin()
    
    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        C,  # Constraint values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        turbo, # is turbo on, trust region is used, otherwise no TR
        acqf, # cts, cei, ...
        sobol: SobolEngine,
    ):

        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        
        # draw random samples in [0,1]^D
        pert        = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        best_ind    = self.get_best_index_for_batch(Y=Y, C=C)  
        dim         = X.shape[-1]
        # Create the TR bounds
        x_center    = X[best_ind, :].clone()

        tr_lb       = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub       = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
        
        # scale the samples into the TR
        if turbo: pert = tr_lb + (tr_ub - tr_lb) * pert
            
        # Create a perturbation mask
        prob_perturb    = min(20.0 / dim, 1.0)
        # get a random matrix with size n_cand x dim and check which value is <= prob_perturb
        mask            = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        # mask.sum counts the number of trues, then check if there is a column where there is 0 trues 
        ind             = torch.where(mask.sum(dim=1) == 0)[0]
        # replace random entry in ind column with one 
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        # X_cand are samples centered at the best candidate and then randomly perturbed via prob_perturb
        X_cand[mask] = pert[mask]

        # map Xcand onto latent space via autoencoder
        if model.input_reduction:
            X_cand_hat = model.ae_input.encode(X_cand)
        else: X_cand_hat = X_cand

        if acqf == 'cts': 
            # constrained Thompson Sampling is chosen
            constrained_acquisition_function = ConstrainedMaxPosteriorSampling(
                model=model.gp_obj, 
                constraint_model=ModelListGP(*model.gp_con), 
                replacement=False
            )
            with torch.no_grad():
                X_next_hat, idcs = constrained_acquisition_function(X_cand_hat, num_samples=batch_size)
                
        elif acqf=='cei':
            # constrained Expected Improvement
            constrained_acquisition_function = ConstrainedExpectedImprovement(
                model   = ModelListGP(*model.gp_con,model.gp_obj),
                best_f  =Y[best_ind],
                constraints={i: (None, None) for i in range(model.ld_out)},
                objective_index=-1
            )
            with torch.no_grad():
                efi = constrained_acquisition_function(X_cand_hat.unsqueeze(1))
                _, idcs = torch.sort(efi)
                idcs = idcs[-batch_size:]
                X_next_hat = X_cand_hat[idcs]

        # map X_next_hat back to the original space
        #X_next = model.ae_input.decoder(X_next_hat)
        # Alternative: Use the alternative from Joco 
        idcs = torch.unique(idcs)
        X_next = X_cand[idcs,:]

        return X_next

    def gaussian_copula_transform(self,data):
        """
        This procedure magnifies differences between values that are at 
        the end of the observed range, i.e., minima or maxima. It affects 
        the observed values but not their location.
        Eriksson, et al.; Scalable Constrained Bayesian Optimisation
        """
        data = data.squeeze(1)
        # Calculate the ranks of the data
        ranks = np.apply_along_axis(lambda x: x.argsort().argsort(), axis=0, arr=data)
        # Transform ranks to uniform [0, 1] using the inverse CDF of the standard normal distribution
        uniform_values = norm.ppf((ranks + 1) / (data.shape[0] + 1))
        # Create a covariance matrix from the standardized data
        covariance = np.cov(uniform_values, rowvar=False)
        # Transform the uniform values to multivariate normal using the Cholesky decomposition
        transformed_data = uniform_values * covariance
        
        return torch.tensor(transformed_data).unsqueeze(-1)

    def bilog_transformation(self,data): 
        return data.sign()*(data.abs()+1).log()    


class autoencoder(nn.Module):

    def __init__(
            self,
            input_dim, 
            latent_dim,
            dtype,
            normalisation='minmax'
            ):
        
        self.dtype = dtype
        
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        ).to(dtype=dtype)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        ).to(dtype=dtype)
        
        self.normalisation = normalisation


    def _normalise(self, x):
        """Computes normalization dynamically and normalizes input."""
        if self.normalisation == "standard":
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-8  # Avoid division by zero
            x_norm = (x - mean) / std
            return x_norm, (mean, std)

        elif self.normalisation == "minmax":
            min_val = x.min(dim=0)[0]
            max_val = x.max(dim=0)[0]
            x_norm = (x - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
            return x_norm, (min_val, max_val)

        else:
            raise ValueError("Unsupported normalization type. Choose 'standard' or 'minmax'.")

    def _denormalise(self, x, norm_params):
        """Reverts normalization after decoding."""
        if self.normalisation == "standard":
            mean, std = norm_params
            return x * std + mean

        elif self.normalisation == "minmax":
            min_val, max_val = norm_params
            return x * (max_val - min_val) + min_val

        else:
            raise ValueError("Unsupported normalization type. Choose 'standard' or 'minmax'.")
    
    def forward(self, X):
        x_norm, norm_params = self._normalise(X)
        z = self.encoder(x_norm)
        x_rec = self.decoder(z)
        return self._denormalise(x_rec, norm_params)
    
    def encode(self, X):
        x_norm, _ = self._normalise(X)
        x = self.encoder(x_norm)
        return x

    def train(
            self,
            train_data,
            batch_size=5, 
            epochs=100, 
            learning_rate=0.01
            ):
        
        # Create DataLoader for training
        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                output = self(batch[0])
                loss = loss_function(output, batch[0])
                loss.backward()
                optimizer.step()
        
        #print('{}/{}, loss:{}'.format(epoch, epochs, loss))  

    def reduce_dimensionality_with_autoencoder(
            self,
            data
            ):
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Convert data to PyTorch tensors
        train_data = torch.Tensor(scaled_data).to(dtype=self.dtype)
        # train the autoencoder
        self.train(train_data)

        encoded_data = self.encode(train_data)

        return torch.tensor(encoded_data, dtype=dtype).T
    
    def map_data_onto_rb(self, data):
        
        # standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Convert data to PyTorch tensors
        data = torch.FloatTensor(scaled_data)
        
        # Convert data to PyTorch tensors
        encoded_data = self.encode(data)
        
        return torch.tensor(encoded_data, dtype=self.dtype).T
    

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, output, target, weights):
        loss = torch.mean((output - target)**2,dim=1)
        return torch.sum(torch.mul(loss,weights))
    
    def get_weights(self,X,Y,C,k=1e-3):

        # TODO: rank based on objective and another one based on 
        # feasibility and combine them, then it would be equally 
        # important to find a feasible point than a good objective value    

        # number of samples being compared 
        n = X.shape[0]
        r = torch.zeros(n)

        Cmax,_ = torch.max(C.abs(),dim=0)
        C = torch.div(C,Cmax)

        # Clamp the points which are feasible
        #C = torch.clamp(C,min=0) # dont know if this is so clever
        # masks to decide feasible/infeasible
        mask_feas   = torch.all(C < 0, axis=1) 
        mask_infeas = ~mask_feas
        
        # total violation based on constraints
        _,idc_tv    = torch.sort(torch.sum(C,dim=1)[mask_infeas], descending=False)
        _,idc_obj   = torch.sort(Y.squeeze(1)[mask_feas], descending=True)
        
        ranks_infeas = torch.zeros(sum(mask_infeas))
        
        if len(idc_tv) > 0:
            ranks_infeas[idc_tv] = torch.range(1,idc_tv.shape[0])
            r[mask_infeas] = ranks_infeas

        if len(idc_obj) > 0:
            ranks_feas = torch.zeros(sum(mask_feas))
            ranks_feas[idc_obj] = torch.range(1,idc_obj.shape[0])
            r += sum(mask_feas)
            r[mask_feas] = ranks_feas
        
        #wi = torch.log(torch.tensor(n)) - torch.log(r)
        wi = 1/(k*n+r) # only proportional to this, then all need to sum up to =1
        wi /= torch.sum(wi) # normalise such that \sum wi = 1

        return wi

from torch.utils.data import TensorDataset, DataLoader

class AEROBO_variationalGP:
    # AEROBO: Autoencoder-Enhanced joint dimensionality Reduction 
    # for Bayesian Optimization
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            ld_in: int, 
            ld_out: int,
            input_reduction: str,
            weighted_training: bool, 
            dtype
    ):
    
        self.input_reduction = input_reduction
        self.dtype = dtype
        self.input_dim = input_dim # X.shape[1]
        self.output_dim = output_dim # C.shape[1]
        self.ld_in = ld_in
        self.ld_out = ld_out
        self.weighted_training = weighted_training

        # initialise models
        if input_reduction:
            self.ae_input = autoencoder(
                input_dim=input_dim,
                latent_dim=ld_in,
                dtype=dtype
                ).to(dtype=dtype)
        
        self.ae_output = autoencoder(
            input_dim=output_dim,
            latent_dim=ld_out,
            dtype=dtype
            ).to(dtype=dtype)
            
        self.initial_training = True
    

    def grads_on(self,model):
        if hasattr(model,'__len__'):
            for mod in model:
                for param in mod.parameters():
                    param.requires_grad = True  
        else:
            for param in model.parameters():
                param.requires_grad = True  
    
    def grads_off(self,model):
        if hasattr(model,'__len__'):
            for mod in model:
                for param in mod.parameters():
                    param.requires_grad = False  
        else:
            for param in model.parameters():
                param.requires_grad = False  

    def joint_learning(
            self, 
            train_X, 
            train_Y,
            train_C,
            num_epochs, 
            lr,
            lookback, 
            ):
        
        # add given parameters to self 
        self.lr, self.num_epochs = lr, num_epochs

        # either map X onto subspace or use original one 
        if self.input_reduction:
            train_X_hat = self.ae_input.encode(X=train_X)
        else: train_X_hat = train_X
        # project outputs into latent space 
        train_C_hat = self.ae_output.encode(X=train_C)

        # VARIATIONAL MULTI-TASK GPs 
        # self.gp_con = VariationalIndependentMultitaskGPModel(
        #     inducing_points=train_X_hat,
        #     num_tasks=self.ld_out
        #     ).to(dtype=self.dtype)
        # initialise objective
            
        # initialize models
        if self.initial_training:
            
            # initialise objecitve
            self.gp_obj = VariationalGPModel(train_X_hat).to(dtype=self.dtype)
            self.mll_obj = VariationalELBO(self.gp_obj.likelihood, self.gp_obj, num_data=train_Y.shape[0])
            self.gp_obj.train()
            self.gp_obj.likelihood.train()
            self.gp_obj_opti     = torch.optim.Adam(self.gp_obj.parameters(), lr=lr)

            # initialise constraints
            self.gp_con, self.mll_con, self.gp_con_opti = [], [], []
            for i in range(self.ld_out):
                model = VariationalGPModel(train_X_hat).to(dtype=self.dtype)
                mll_con_i = VariationalELBO(model.likelihood, model, num_data=train_C.shape[0])
                model.train()
                model.likelihood.train()
                gp_con_i_opti = torch.optim.Adam(model.parameters(), lr=lr)
                
                self.mll_con.append(mll_con_i)
                self.gp_con.append(model)
                self.gp_con_opti.append(gp_con_i_opti)
            
            # set intialisation to False such that next iteration 
            # the models are updated and not reinitialised
            self.initial_training = False
        
        if self.input_reduction:
            ae_input_opti   = torch.optim.Adam(self.ae_input.parameters(), lr=lr)
        ae_output_opti  = torch.optim.Adam(self.ae_output.parameters(), lr=lr)
            
        loss_function = nn.MSELoss()

        if self.weighted_training: weighted_loss = WeightedLoss()


        data = TensorDataset(train_X, train_X_hat, train_Y, train_C, train_C_hat)
        dloader = DataLoader(data,batch_size=64,shuffle=True)

        for _ in range(num_epochs):
            for x, x_hat, y, c, c_hat in dloader:
                
                # forward passes through models
                x_prime = self.ae_input(x) if self.input_reduction else x
                c_prime = self.ae_output(c)
                
                obj_output   = self.gp_obj(x_hat)
                
                # Calculate reconstruction loss

                if self.input_reduction:
                    if self.weighted_training:
                        weights = weighted_loss.get_weights(x,y,c) 
                        loss_ae_in = weighted_loss(x_prime,x,weights)
                    else:
                        loss_ae_in   = loss_function(x_prime, x)
                
                loss_ae_out  = loss_function(c_prime, c)
                loss_obj     = -self.mll_obj(obj_output, y.squeeze(-1))

                if self.input_reduction:
                    joint_loss = loss_ae_in + loss_ae_out + loss_obj
                else:
                    joint_loss = loss_ae_out + loss_obj

                for i in range(self.ld_out):
                    con_output   = self.gp_con[i](x_hat)
                    loss_con     = -self.mll_con[i](con_output, c_hat[:,i])
                    joint_loss  += loss_con

                if self.input_reduction:
                    joint_loss /= 3 + self.ld_out
                else: 
                    joint_loss /= 2 + self.ld_out

                # backprop
                joint_loss.backward(retain_graph=True)

                # backprop, step and reset grads
                if self.input_reduction:
                    ae_input_opti.step()
                    ae_input_opti.zero_grad()
 
                ae_output_opti.step()
                ae_output_opti.zero_grad()
                
                self.gp_obj_opti.step()
                self.gp_obj_opti.zero_grad()
                
                for i in range(self.ld_out):
                    self.gp_con_opti[i].step()
                    self.gp_con_opti[i].zero_grad()
    
    def non_joint_learning(
            self, 
            train_X, 
            train_Y,
            train_C,
            num_epochs, 
            lr,
            lookback,
            ):
        
        # add given parameters to self 
        self.lr, self.num_epochs = lr, num_epochs

        if lookback is not None:
            train_X = train_X[-lookback:,:]
            train_Y = train_Y[-lookback:]
            train_C = train_C[-lookback:,:]
        
        # either map X onto subspace or use original one 
        if self.input_reduction:
            train_X_hat = self.ae_input.encode(X=train_X)
        else: train_X_hat = train_X
        # project outputs into latent space 
        train_C_hat = self.ae_output.encode(X=train_C)
        
        data = TensorDataset(train_X, train_X_hat, train_Y, train_C, train_C_hat)
        dloader = DataLoader(data,batch_size=64,shuffle=True)
        
        loss_function = nn.MSELoss()
        
        if self.weighted_training: weighted_loss = WeightedLoss()
        
        ae_input_opti   = torch.optim.Adam(self.ae_input.parameters(), lr=lr) 
        self.grads_on(self.ae_input) 
        for _ in range(num_epochs):
            for x, x_hat, y, c, c_hat in dloader:
                # forward passes through models
                x_prime    = self.ae_input(x)
                
                if self.input_reduction:
                    if self.weighted_training:
                        weights = weighted_loss.get_weights(x,y,c) 
                        loss_ae_in = weighted_loss(x_prime,x,weights)
                    else:
                        loss_ae_in   = loss_function(x_prime, x)
                
                loss_ae_in.backward()
                # backprop, step and reset grads
                ae_input_opti.step()
                ae_input_opti.zero_grad()
        self.grads_off(self.ae_input) 

        # OUTPUT AE

        ae_output_opti  = torch.optim.Adam(self.ae_output.parameters(), lr=lr)
        self.grads_on(self.ae_output)
        for _ in range(num_epochs):
            for x, x_hat, y, c, c_hat in dloader:
                # forward passes through models
                c_prime    = self.ae_output(c)
                loss_ae_out   = loss_function(c_prime, c)
                loss_ae_out.backward()
                # backprop, step and reset grads
                ae_output_opti.step()
                ae_output_opti.zero_grad()
        self.grads_off(self.ae_output)
        
        # OBJECTIVE GP
        self.gp_obj = VariationalGPModel(train_X_hat).to(dtype=self.dtype)
        self.gp_obj.train()
        self.gp_obj.likelihood.train()
        mll_obj = VariationalELBO(self.gp_obj.likelihood, self.gp_obj, num_data=train_Y.shape[0])
        gp_obj_opti     = torch.optim.Adam(self.gp_obj.parameters(), lr=lr)
        self.grads_on(self.gp_obj)
        for _ in range(num_epochs):
            for x, x_hat, y, c, c_hat in dloader:    
                obj_output   = self.gp_obj(x_hat)
                loss_obj     = -mll_obj(obj_output, y.squeeze(-1))
                loss_obj.backward(retain_graph=True)
                gp_obj_opti.step()
                gp_obj_opti.zero_grad()
        self.grads_off(self.gp_obj)

        # initialize models
        self.gp_con, self.mll_con, self.gp_con_opti = [], [], []
        for i in range(self.ld_out):
            model = VariationalGPModel(train_X_hat).to(dtype=self.dtype)
            model.train()
            model.likelihood.train()
            mll_con_i = VariationalELBO(model.likelihood, model, num_data=train_C.shape[0])
            gp_con_i_opti = torch.optim.Adam(model.parameters(), lr=lr)
            
            self.mll_con.append(mll_con_i)
            self.gp_con.append(model)
            self.gp_con_opti.append(gp_con_i_opti)

        self.grads_on(self.gp_con)
        for _ in range(num_epochs):
            for x, x_hat, y, c, c_hat in dloader:
                for i in range(self.ld_out):
                    con_output   = self.gp_con[i](x_hat)
                    loss_con_i   = -self.mll_con[i](con_output, c_hat[:,i])
                    if i==0: loss_con = loss_con_i
                    else: loss_con += loss_con_i

                loss_con.backward(retain_graph=True)
                for i in range(self.ld_out):
                    self.gp_con_opti[i].step()
                    self.gp_con_opti[i].zero_grad()
        self.grads_off(self.gp_con)