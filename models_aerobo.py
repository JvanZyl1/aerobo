# Adapted from BoTorch (https://github.com/meta-pytorch/botorch), licensed under the MIT License.

from __future__ import annotations

import warnings
from typing import Dict, NoReturn, Optional, Union

import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
    MIN_INFERRED_NOISE_LEVEL,
)
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)

from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, VariationalStrategy, MultitaskVariationalStrategy
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.models.exact_gp import ExactGP
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.models import IndependentModelList
import gpytorch.variational.lmc_variational_strategy as lmc
from gpytorch.module import Module
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood

class BatchedSingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""A single-task exact GP model, supporting both known and inferred noise levels.

    A single-task exact GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the
    same training data. If outputs are independent and outputs have different
    training data, use the MtodelListGP. When modeling correlaions between
    outputs, use the MultiTaskGP.

    An example of a case in which noise levels are known is online
    experimentation, where noise can be measured using the variability of
    different observations from the same arm, or provided by outside software.
    Another use case is simulation optimization, where the evaluation can
    provide variance estimates, perhaps from bootstrapping. In any case, these
    noise levels can be provided to `SingleTaskGP` as `train_Yvar`.

    `SingleTaskGP` can also be used when the observations are known to be
    noise-free. Noise-free observations can be modeled using arbitrarily small
    noise values, such as `train_Yvar=torch.full_like(train_Y, 1e-6)`.

    Example:
        Model with inferred noise levels:

        >>> import torch
        >>> from botorch.models.gp_regression import SingleTaskGP
        >>> from botorch.models.transforms.outcome import Standardize
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> outcome_transform = Standardize(m=1)
        >>> inferred_noise_model = SingleTaskGP(
        ...     train_X, train_Y, outcome_transform=outcome_transform,
        ... )

        Model with a known observation variance of 0.2:

        >>> train_Yvar = torch.full_like(train_Y, 0.2)
        >>> observed_noise_model = SingleTaskGP(
        ...     train_X, train_Y, train_Yvar,
        ...     outcome_transform=outcome_transform,
        ... )

        With noise-free observations:

        >>> train_Yvar = torch.full_like(train_Y, 1e-6)
        >>> noise_free_model = SingleTaskGP(
        ...     train_X, train_Y, train_Yvar,
        ...     outcome_transform=outcome_transform,
        ... )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred noise level if `train_Yvar`
                is None, and a `FixedNoiseGaussianLikelihood` with the given
                noise observations if `train_Yvar` is not None.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            ignore_X_dims=ignore_X_dims,
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        if likelihood is None:
            if train_Yvar is None:
                likelihood = get_gaussian_likelihood_with_gamma_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=train_Yvar, batch_shape=self._aug_batch_shape
                )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=transformed_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            )
            self._subset_batch_dict = {
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
            if train_Yvar is None:
                self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.covar_module: Module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    @classmethod
    def construct_inputs(
        cls, training_data: SupervisedDataset, *, task_feature: Optional[int] = None
    ) -> Dict[str, Union[BotorchContainer, Tensor]]:
        r"""Construct `SingleTaskGP` keyword arguments from a `SupervisedDataset`.

        Args:
            training_data: A `SupervisedDataset`, with attributes `train_X`,
                `train_Y`, and, optionally, `train_Yvar`.
            task_feature: Deprecated and allowed only for backward
                compatibility; ignored.

        Returns:
            A dict of keyword arguments that can be used to initialize a `SingleTaskGP`,
            with keys `train_X`, `train_Y`, and, optionally, `train_Yvar`.
        """
        if task_feature is not None:
            warnings.warn(
                "`task_feature` is deprecated and will be ignored. In the "
                "future, this will be an error.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().construct_inputs(training_data=training_data)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# -----------------------------------------------------
# VARIATIONAL GPs 
# -----------------------------------------------------

class VariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.num_outputs = 1
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        # self.model.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(dist)


class VariationalIndependentMultitaskGPModel(ApproximateGP):
    def __init__(self, inducing_points, num_tasks):
        
        self.num_outputs = num_tasks

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[0], 
            batch_shape=torch.Size([num_tasks]),
            )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, 
                inducing_points, 
                variational_distribution, 
                learn_inducing_locations=False
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_tasks]))
        )
        
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([num_tasks])) # rank=0 for no cross-task cov


    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        dist = self.likelihood(self(X))
        return GPyTorchPosterior(dist)
    

    import warnings



import torch
from linear_operator.operators import RootLinearOperator
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy

class IndependentMultitaskVariationalStrategy(_VariationalStrategy):
    """
    IndependentMultitaskVariationalStrategy wraps an existing
    :obj:`~gpytorch.variational.VariationalStrategy` to produce vector-valued (multi-task)
    output distributions. Each task will be independent of one another.

    The output will either be a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal` distribution
    (if we wish to evaluate all tasks for each input) or a :obj:`~gpytorch.distributions.MultivariateNormal`
    (if we wish to evaluate a single task for each input).

    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of task_dim.
    :param int task_dim: (Default: -1) Which batch dimension is the task dimension
    """

    def __init__(self, base_variational_strategy, num_tasks, task_dim=-1):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.task_dim = task_dim
        self.num_tasks = num_tasks

    @property
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=-1)

    def __call__(self, x, task_indices=None, prior=False, **kwargs):
        r"""
        See :class:`LMCVariationalStrategy`.
        """
        function_dist = self.base_variational_strategy(x, prior=prior, **kwargs)

        if task_indices is None:
            # Every data point will get an output for each task
            if (
                self.task_dim > 0
                and self.task_dim > len(function_dist.batch_shape)
                or self.task_dim < 0
                and self.task_dim + len(function_dist.batch_shape) < 0
            ):
                return MultitaskMultivariateNormal().from_repeated_mvn(function_dist, num_tasks=self.num_tasks)
            else:
                function_dist = MultitaskMultivariateNormal.from_repeated_mvn(
                    function_dist, 
                    num_tasks=self.num_tasks,
                    #task_dim=self.task_dim
                    )
                assert function_dist.event_shape[-1] == self.num_tasks
                return function_dist

        else:
            # Each data point will get a single output corresponding to a single task

            if self.task_dim > 0:
                raise RuntimeError(f"task_dim must be a negative indexed batch dimension: got {self.task_dim}.")
            num_batch = len(function_dist.batch_shape)
            task_dim = num_batch + self.task_dim

            # Create a mask to choose specific task assignment
            shape = list(function_dist.batch_shape + function_dist.event_shape)
            shape[task_dim] = 1
            task_indices = task_indices.expand(shape).squeeze(task_dim)

            # Create a mask to choose specific task assignment
            task_mask = torch.nn.functional.one_hot(task_indices, num_classes=self.num_tasks)
            task_mask = task_mask.permute(*range(0, task_dim), *range(task_dim + 1, num_batch + 1), task_dim)

            mean = (function_dist.mean * task_mask).sum(task_dim)
            covar = (function_dist.lazy_covariance_matrix * RootLinearOperator(task_mask[..., None])).sum(task_dim)
            return MultivariateNormal(mean, covar)
