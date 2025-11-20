'''
Adjusted sampling strategies
FROM BOTORCH
'''

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import SamplingStrategy
from botorch.models import MultiTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.model import Model
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model
from botorch.utils.sampling import batched_multinomial
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.utils.transforms import t_batch_mode_transform #, convert_to_target_pre_hook
from botorch.acquisition import ExpectedImprovement, ConstrainedExpectedImprovement

import torch
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi,
    log_prob_normal_in,
    ndtr as Phi,
    phi,
)
from torch import Tensor
from sklearn.model_selection import train_test_split
from scipy.stats import norm

class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs), idcs

class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model

    def _convert_samples_to_scores(self, Y_samples, C_samples) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = -total_violation
        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            # Note: `posterior_transform` is only used for the objective
            posterior_transform=self.posterior_transform,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        c_posterior = self.constraint_model.posterior(
            X=X, observation_noise=observation_noise
        )
        C_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples, C_samples=C_samples
        )

        X_new, idcs = self.maximize_samples(X=X, samples=scores, num_samples=num_samples)

        return X_new, idcs
    
class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Base class for analytic acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        if posterior_transform is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma
    
class ConstrainedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Constrained Expected Improvement (feasibility-weighted).

    Computes the analytic expected improvement for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports non-batch mode (i.e. `q=1`). The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        # example where the 0th output has a non-negativity constraint and
        # 1st output is the objective
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> cEI = ConstrainedExpectedImprovement(model, 0.2, 1, constraints)
        >>> cei = cEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        maximize: bool = True,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        # Use AcquisitionFunction constructor to avoid check for posterior transform.
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.model = model
        self.constraints = constraints
        self.register_buffer("best_f", torch.as_tensor(best_f))
        _preprocess_constraint_bounds(self, constraints=constraints)
        #self.register_forward_pre_hook(convert_to_target_pre_hook)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  
        # objective gp needs to at idx = -1
        mean_obj, sigma_obj = means[..., -1], sigmas[..., -1]
        u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        ei = sigma_obj * _ei_helper(u)
        log_prob_feas = _compute_log_prob_feas(self, means[...,:-1], sigmas[...,:-1])
        return ei.mul(log_prob_feas.exp())


#----------
# Helper Functions
#----------

def _preprocess_constraint_bounds(
    acqf: Union[ConstrainedExpectedImprovement],
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
) -> None:
    r"""Set up constraint bounds.

    Args:
        constraints: A dictionary of the form `{i: [lower, upper]}`, where
            `i` is the output index, and `lower` and `upper` are lower and upper
            bounds on that output (resp. interpreted as -Inf / Inf if None)
    """
    con_lower, con_lower_inds = [], []
    con_upper, con_upper_inds = [], []
    con_both, con_both_inds = [], []
    con_indices = list(constraints.keys())
    if len(con_indices) == 0:
        raise ValueError("There must be at least one constraint.")
    if acqf.objective_index in con_indices:
        raise ValueError(
            "Output corresponding to objective should not be a constraint."
        )
    for k in con_indices:
        if constraints[k][0] is not None and constraints[k][1] is not None:
            if constraints[k][1] <= constraints[k][0]:
                raise ValueError("Upper bound is less than the lower bound.")
            con_both_inds.append(k)
            con_both.append([constraints[k][0], constraints[k][1]])
        elif constraints[k][0] is not None:
            con_lower_inds.append(k)
            con_lower.append(constraints[k][0])
        elif constraints[k][1] is not None:
            con_upper_inds.append(k)
            con_upper.append(constraints[k][1])
    # tensor-based indexing is much faster than list-based advanced indexing
    for name, indices in [
        ("con_lower_inds", con_lower_inds),
        ("con_upper_inds", con_upper_inds),
        ("con_both_inds", con_both_inds),
        ("con_both", con_both),
        ("con_lower", con_lower),
        ("con_upper", con_upper),
    ]:
        acqf.register_buffer(name, tensor=torch.as_tensor(indices))


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)

def _compute_log_prob_feas(
    acqf: Union[ConstrainedExpectedImprovement],
    means: Tensor,
    sigmas: Tensor,
) -> Tensor:
    r"""Compute logarithm of the feasibility probability for each batch of X.

    Args:
        X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
            points each.
        means: A `(b) x m`-dim Tensor of means.
        sigmas: A `(b) x m`-dim Tensor of standard deviations.
    Returns:
        A `b`-dim tensor of log feasibility probabilities

    Note: This function does case-work for upper bound, lower bound, and both-sided
    bounds. Another way to do it would be to use 'inf' and -'inf' for the
    one-sided bounds and use the logic for the both-sided case. But this
    causes an issue with autograd since we get 0 * inf.
    TODO: Investigate further.
    """
    acqf.to(device=means.device)
    log_prob = torch.zeros_like(means[..., 0])
    if len(acqf.con_lower_inds) > 0:
        i = acqf.con_lower_inds
        dist_l = (acqf.con_lower - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_Phi(-dist_l).sum(dim=-1)  # 1 - Phi(x) = Phi(-x)
    if len(acqf.con_upper_inds) > 0:
        i = acqf.con_upper_inds
        dist_u = (acqf.con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_Phi(dist_u).sum(dim=-1)
    if len(acqf.con_both_inds) > 0:
        i = acqf.con_both_inds
        con_lower, con_upper = acqf.con_both[:, 0], acqf.con_both[:, 1]
        # scaled distance to lower and upper constraint boundary:
        dist_l = (con_lower - means[..., i]) / sigmas[..., i]
        dist_u = (con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_prob_normal_in(a=dist_l, b=dist_u).sum(dim=-1)
    return log_prob

