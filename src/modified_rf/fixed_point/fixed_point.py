# Copyright 2026 Raul Mohedano and Erik Velasco-Salido (Vision Modeling Group, CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import torch
from torch import nn
import torch.nn.functional as F

import torchdeq
from torchdeq import get_deq



######################################################
######################################################
### AUXILIARY FUNCTIONS
######################################################
######################################################


######################################################
### Fixing all variables but the first one ###
######################################################

def _generate_f_reduced_input_variables(f_of_all_vars, *rest_input_vars):
    """
    For a given function "f_of_all_vars" of y and other variables, e.g. x and a number of family parameter sets *theta,
    block all the other parameters but y for the given values.
    WARNING:
        - The input function "f_of_all_vars" presumes the following input variable order:
            f_of_all_vars(y, *rest_input_vars), wherein theta is a variable number of tensor variables.
        - As an example: for a function "f_of_all_vars(y, x, theta_1, ..., theta_N)", the call
            "_generate_f_reduced_input_variables(f_of_all_vars, x0, , theta_1_0, ..., theta_N_0) would return
          a function of y only with the rest of parameters fixed, respectively, to
            "x0, , theta_1_0, ..., theta_N_0".
    :param f_of_all_vars:
    :param rest_input_vars:
    :return:
    """

    def _f_of_y_only(y):
        f_y = f_of_all_vars(y, *rest_input_vars)
        return f_y

    return _f_of_y_only


######################################################
### Extraction of full Jacobian matrix for a certain operation
######################################################

def full_jacobian_matrix(f_x, x):
    """
    Calculates the Jacobian matrix for the output (tensor) "f_x" which is related, through some function, to "x".

    *WARNING*: If either "f_x" or "x" is (or both are) 1D then the resulting Jacobian is a vector;
    in that case the returned tensor is a vector = a tensor of dimension 1.

    Parameters
    ----------
    f_x : int or float
        Description of f_x
    x : torch.Tensor
        Description of x
    Returns
    -------
    torch.Tensor
        Description of the returned Jacobian matrix
    """
    list_of_gradients = []
    for i in range(f_x.numel()):
        x.grad = None
        f_x_i = torch.take(f_x, torch.tensor(i))
        f_x_i.backward(retain_graph=True)
        list_of_gradients.append(x.grad.detach())
    # print(f"list_of_gradients = {list_of_gradients}")
    jacobian_matrix = torch.column_stack(list_of_gradients)
    # print(f"Jacobian matrix = {jacobian_matrix}")
    if jacobian_matrix.size()[0] in [0, 1]:
        jacobian_matrix = torch.flatten(jacobian_matrix)
    # print(jacobian_matrix)
    return jacobian_matrix


######################################################
######################################################
### Models/layers
######################################################
######################################################


class FixedPointLayer(nn.Module):
    """
    This layer implements the fixed point operation corresponding to
    $$\\mathbf{y}^* = f_{\\theta}(\\mathbf{x}, \\mathbf{y}^*)$$
    providing the output $\\mathbf{y}^*$ for a given input $\\mathbf{x}$,
    wherein $\\theta$ corresponds to the parameter(s) ruling the behavior of $f(·; \\theta)$.
    The function f will have to be defined in a manner that:
        - takes the arguments $\\theta$ in the appropriate manner,
        - outputs the same dimensionality as $\\mathbf{y}$.
    The number of input dimensions will not be explicitly specified, since it will be provided, but must be consistent
    with the requirements of the function $f_{\\theta}(·, ·)$.

    Fixed point operations and their corresponding backward calculations are performed, or can be performed, using the \
    library `TorchDEQ <https://torchdeq.readthedocs.io/en/latest/get_started.html#quick-start>`_, which supports the \
    following solvers: ``'anderson'``, ``'broyden'``, ``'fixed_point_iter'`` (default), and \
    ``'simple_fixed_point_iter'`` for both the forward and backward fixed point calculations of the implicit derivation.

        - WARNING (1): ``'fixed_point_iter'`` does, exactly, what its name suggests; that means that
          it is not accelerated and that it only reaches fixed points which are **attracting fixed points**
          and from initial points $\\mathbf{y}_0$ which are in the basin of attraction of the point
          (see `Wiki <https://en.wikipedia.org/wiki/Fixed-point_iteration#Attracting_fixed_points>`_).
          However, ``anderson`` and ``broyden`` methods appear to find the root
          of $\\mathbf{g}(\\mathbf{y})=\\mathbf{f}(y)-\\mathbf{y}$
          and seem to work well in more general settings. This impression, however, must be taken only as an indication
          and not as a rule.

    Certain of the solvers, namely ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, and ``'anderson'``, \
    allow for a parameter ``tau`` ($\\tau$), which servers as a dampening factor for the calculation. For the case of \
    ``'fixed_point_iter'`` and ``'simple_fixed_point_iter'`` it implies the following damped iteration:

        $$\\mathbf{y}^{(n)} = (1-\\tau) \\, \\mathbf{y}^{(n-1)} +
        \\tau \\, f_{\\theta}(\\mathbf{x}, \\mathbf{y}^{(n-1)}) \\; .$$

    Regarding the allowed solvers: this class :py:class:`.FixedPointLayer` \
    is both a class on its own and a parent/interface class. \
    As a parent/interface class, it allows for forward fixed point methods different from the solvers included by \
    the library TorchDEQ (that is, from ``'anderson'``, ``'broyden'``, ``'fixed_point_iter'`` (default), and \
    ``'simple_fixed_point_iter'``): however, in those cases, which are managed by the `forward()` method of the child \
    class and are managed by generated the fixed point solution by that other method and input to the \
    :py:meth:`.FixedPointLayer.forward` method as the initial solution, this class :py:class:`.FixedPointLayer` \
    still needs one of the "canonical" solvers as a "surrogate" solver, which is only symbolically used. \
    The backward fixed point calculation is always performed by one of the canonical solvers.

    Parameters
    ----------
    f : ~collections.abc.Callable
        Function `f(y, x; theta)` (notice that this is the presumed order for its variables), wherein

            - `f` returns a tensor of the same dimensionality of the tensor `y`
            - `x` is a tensor
            - `theta` represents the parameters ruling the shape of the function `f`
            - and the fixed point problem is defined as `y* = f(y*,x; theta)`.
    theta0 : torch.Tensor or dict[torch.Tensor] or list[torch.Tensor]
        Function parameters theta for `f(y,x; theta)`.
        The only requirement that the defined function `f` accepts the used formatting, said `theta0` can be any of
            - a tensor
            - a dictionary of named tensors
            - a list of tensors
        If the argument ``theta_randomly_initialised`` is set to ``True`` its initial content is randomized (see below).
        (*Note*: `theta` can be a tensor, or a list or a dict of tensors: it is important that they are tensors
        because they need to be registered as ``torch.nn.Parameter()`` of the module so they can be automatically
        addressed in the optimization process of the training, and only individual tensors can be registered.
        Different procedure will be followed for each case (tensor, list of tensors, dict of tensors).
    random_initialization_theta : bool, optional
        If ``True`` the exact values of the provided structure ``theta0`` are unimportant: `theta` is composed by
        initialising randomly the exact values of its composing tensors from a uniform distribution [0,1).
        Default: ``False``
    y0 : torch.Tensor, optional
        Initial value for y for the fixed point algorithm aiming at finding `y* = f(y*,x; theta)`.
        Not required. Used to have a fixed initial point for all the fixed
        point iterations. The :py:meth:`.forward` method would in any case allow input of other initial points
        at evaluation.
    random_initialization_y0 : bool, optional
        If ``True`` the exact values of the provided structure ``theta0`` are unimportant: `theta` is composed by
        initialising randomly the exact values of its composing tensors from a uniform distribution [0,1).
        Default: ``False``
    batched_fixed_point : bool, optional
        It indicates whether the calculation in :py:meth:`.forward` of the fixed point  for an input batch
        of $B$ elements (that is, for a 4D input) should operate on each one of its $B$ individual 3D entries
        (if ``False``) or, on the contrary (if ``True``), on the 4D input as one single item.
        Default: ``True``
    ift : bool, optional
    hook_ift : bool, optional
    f_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, and ``'broyden'``, \
        or other value if the forward point calculation is managed by an external forward solver: \
        in the latter case one of the canonical solvers must be used as a surrogate solver.
        It indicates the type of fixed point problem algorithm for the forward fixed point calculation, \
        whose underlying solver functions for the canonical solvers are, respectively, \
        :py:func:`~torchdeq.solver.fp_iter.fixed_point_iter`, \
        :py:func:`~torchdeq.solver.fp_iter.simple_fixed_point_iter`, \
        :py:func:`~torchdeq.solver.anderson.anderson_solver`, and :py:func:`~torchdeq.solver.broyden.broyden_solver`.
        Default: ``'fixed_point_iter'``
    f_max_iter : int, optional
        Maximum number of iterations for the solver really used \
        (see definition of the argument ``f_solver`` regarding when this value corresponds to the actual solver).
        Default: ``51``
    f_tol : float
        Stop condition for the forward fixed point calculation for the solver really used \
        (see definition of the argument ``f_solver`` regarding when this value corresponds to the actual solver).
        Default: ``1e-5``
    f_tau : float, optional
        Float in the range $(0.0, 1.0]$, or None. For the solver algorithms allowing so, \
        i.e. ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, and ``'anderson'``, \
        it servers as a dampening factor for the calculation. For the case of \
        ``'broyden'``, however, no ``f_tau`` is used.
        Default: ``1.0`` (``None`` for ``'broyden'`` solver).
    surrogate_f_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``, or ``None``. \
        (see definition of the argument ``f_solver`` regarding when this value corresponds to the actual solver).
        Type of fixed point problem algorithm for the forward fixed point calculation, whose underlying solver \
        functions are, respectively, :py:func:`~torchdeq.solver.fp_iter.fixed_point_iter`, \
        :py:func:`~torchdeq.solver.fp_iter.simple_fixed_point_iter`, \
        :py:func:`~torchdeq.solver.anderson.anderson_solver`, and :py:func:`~torchdeq.solver.broyden.broyden_solver`.
        Default: ``None``
    surrogate_f_max_iter : int, optional
        Maximum number of iterations for the surrogate solver \
        (see definition of the argument ``f_solver`` regarding when this value corresponds to the actual solver).
        Default: ``None``
    surrogate_f_tol : float
        Stop condition for the forward fixed point calculation for the surrogate solver \
        (see definition of the argument ``f_solver`` regarding when this value corresponds to the actual solver).
        Default: ``None``
    surrogate_f_tau : float, optional
        Float in the range $(0.0, 1.0]$ (see the definition of ``f_tau``).
        Default: ``None``
    b_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the fixed point calculation \
        used in the gradient calculation of the IFT. Default: ``'fixed_point_iter'``
    b_max_iter : int, optional
        Default: ``40``
    b_tol : float
        Stop condition for the backward fixed point calculation. Default: ``1e-6``
    abs_error_threshold : float, optional
        The absolute error threshold, measured as the norm of the error between f(x) and x, \
        for the fixed point calculation. Default: ``1e-5``
    """

    def __init__(self,
                 f=None, theta0=None, random_initialization_theta=False,
                 y0=None, random_initialization_y0=False,
                 batched_fixed_point=True,
                 ift=True, hook_ift=False,
                 f_solver='fixed_point_iter', f_max_iter=51, f_tol=1e-5, f_tau=1.0,
                 surrogate_f_solver=None, surrogate_f_max_iter=51, surrogate_f_tol=1e-5, surrogate_f_tau=1.0,
                 b_solver='fixed_point_iter', b_max_iter=40, b_tol=1e-6,
                 abs_error_threshold=1e-5,
                 ):

        super().__init__()

        ### INITIALIZATION AND REGISTRATION OF THE COMPONENTS OF THE IMPLICIT FUNCTION
        #
        # The function f (for y*=f(x,y*; theta)), there the first argument is y, then x, then theta.
        self._f_function_of_y_x_theta = f
        """
        Function `f(y, x; theta)` of said three variables (notice that this is the presumed order for its variables)
        """
        # Batched fixed point operations or not
        self._batched_fixed_point = batched_fixed_point

        # Abs error threshold for convergence assessment
        self._abs_error_threshold = abs_error_threshold
        """
        Attribute that indicates whether the calculation in :py:meth:`.forward` of the fixed point  for an input batch
        of $B$ elements (that is, for a 4D input) should operate on each one of its $B$ individual 3D entries
        (if ``False``) or, on the contrary (if ``True``), on the 4D input as one single item
        """
        #
        # Theta can be a tensor, or a list or a dict of tensors: it is important that they are tensors because
        # they need to be registered as Parameter() of the module so they can be automatically addressed in the
        # optimization process of the training, and only individual tensors can be registered.
        # Different procedure will be followed for each case (tensor, list of tensors, dict of tensors):
        # First: "theta" is given exactly, or only a structure to be filled randomly is given. If randomly, filled with
        # values from a uniform distribution [0,1) (that is, torch.rand())

        provisional_theta = theta0
        # Register "theta" tensors (randomly initialising if necessary).
        # IMPORTANT: there are special classes ParameterDict and ParameterList, instead of the normal versions!!!
        if isinstance(provisional_theta, torch.Tensor):  # IF TENSOR
            if random_initialization_theta:
                provisional_theta = torch.rand(provisional_theta.size())
            pass
            self._theta = nn.Parameter(provisional_theta, requires_grad=True)
        elif isinstance(provisional_theta, list):  # IF LIST
            self._theta = nn.ParameterList()
            for ind in range(0, len(provisional_theta)):
                if not isinstance(provisional_theta[ind], torch.Tensor):
                    raise TypeError(
                        (f"list does not contain all torch.Tensor: " +
                         f"{type(provisional_theta[ind])} for the {ind}-th element is given."
                         )
                    )
                pass
                if random_initialization_theta:
                    provisional_theta[ind] = torch.rand(provisional_theta[ind].size())
                pass
                self._theta.append(nn.Parameter(provisional_theta[ind], requires_grad=True))
            pass
        elif isinstance(provisional_theta, dict):  # IF DICT
            self._theta = nn.ParameterDict()
            for key in provisional_theta.keys():
                if not isinstance(provisional_theta[key], torch.Tensor):
                    raise TypeError(
                        (f"dict does not contain all torch.Tensor: " +
                         f"{type(provisional_theta[key])} for key {key} is given."
                         )
                    )
                pass
                if random_initialization_theta:
                    provisional_theta[key] = torch.rand(provisional_theta[key].size())
                pass
                self._theta[key] = nn.Parameter(provisional_theta[key], requires_grad=True)
            pass
        else:
            raise TypeError(
                (f"Only torch.Tensor, list (of torch.Tensor), or list (of torch.Tensor) accepted: " +
                 f"{type(self._theta)} is given."
                 )
            )
        pass

        """ Analogous for y0."""
        self.y0 = None
        if y0 is not None:
            self.y0 = y0
            if random_initialization_y0:
                self.y0 = torch.rand(self.y0.size())
            pass
        pass

        ######################################################################################################
        ### INITIALIZATION OF THE OBJECT OF THE CLASS DEQBase
        ######################################################################################################

        ### FORWARD SOLVER

        list_canonical_solvers = ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden']
        list_canonical_solvers_with_tau = ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson']

        self._f_solver = f_solver
        self._f_max_iter = f_max_iter
        self._f_tol = f_tol
        #
        self._f_tau = None
        if self._f_solver in list_canonical_solvers and self._f_solver not in list_canonical_solvers_with_tau:
            # In this case: it does not use f_tau, so no need to check it or store it
            self._f_tau = None
        elif not isinstance(f_tau, (float, int)) or (f_tau <= 0.0) or (f_tau > 1.0):
            raise ValueError(f"'f_tau' must be a scalar in the range (0.0, 1.0]!")
        else:
            self._f_tau = f_tau
        pass

        # "Surrogate" forward solver: this occurs when a child class uses a different forward solver from those
        # allowed/considering in this parent class, i.e. this FixedPointLayer; in such case the child class receives,
        # in its forward pass, an initial value which is directly the "desired solution" of the fixed point problem.
        # In such case "self._f_XXXXX" is the method of the child class, which may or may not be one of the forward
        # solvers allowed by the FixedPointLayer/get_deq function.

        if surrogate_f_solver is None:  # No surrogate solver has been provided
            if f_solver in list_canonical_solvers:
                self._surrogate_f_solver = self._f_solver
                self._surrogate_f_max_iter = self._f_max_iter
                self._surrogate_f_tol = self._f_tol
                self._surrogate_f_tau = self._f_tau
            else:                       # We assign by default a surrogate solver which does nothing
                self._surrogate_f_solver = 'fixed_point_iter'
                self._surrogate_f_max_iter = 1
                self._surrogate_f_tol = self._f_tol
                self._surrogate_f_tau = 0.0
            pass
        elif surrogate_f_solver in list_canonical_solvers:      # A surrogate solver has been provided
            if surrogate_f_max_iter is None or surrogate_f_max_iter is None:
                raise ValueError(f"Surrogate forward solver '{surrogate_f_solver}' requires 'surrogate_f_max_iter' and " +
                                 f"'surrogate_f_tol' to be provided!")
            pass
            self._surrogate_f_solver = surrogate_f_solver
            self._surrogate_f_max_iter = surrogate_f_max_iter
            self._surrogate_f_tol = surrogate_f_tol
            self._surrogate_f_tau = None
            if self._surrogate_f_solver in list_canonical_solvers_with_tau:
                if not isinstance(surrogate_f_tau, (float, int)) or (surrogate_f_tau < 0.0) or (surrogate_f_tau > 1.0): # Now we allow for tau=0.0!
                    raise ValueError(f"'surrogate_f_tau' must be a scalar in the range [0.0, 1.0]!")
                else:
                    self._surrogate_f_tau = surrogate_f_tau
                pass
            pass
        else:
            raise ValueError(f"Surrogate forward solver '{surrogate_f_solver}' not allowed: must be one of " +
                             f"{list_canonical_solvers}!")
        pass

        ### BACKWARD SOLVER

        self._b_solver = b_solver
        self._b_max_iter = b_max_iter
        self._b_tol = b_tol

        ### CREATION OF THE DEQBase OBJECT

        self._deq_manager = get_deq(
            ift=ift, hook_ift=hook_ift,
            f_solver=self._surrogate_f_solver, f_max_iter=self._surrogate_f_max_iter, f_tol=self._surrogate_f_tol,
            b_solver=self._b_solver, b_max_iter=self._b_max_iter, b_tol=self._b_tol
        )

        ### INITIALIZATION OF THE FIXED POINT CALCULATION RESULTS/PARAMETERS
        self._last_forward_warning_above_threshold_y_out = None
        self._last_forward_abs_error_threshold = None
        self._last_forward_info = None
        self._last_forward_batched_fixed_point = None

    def eval(self):
        """
        Set the module in evaluation mode.
        """
        self.training = False
        self._deq_manager.eval()

    def train(self, mode=True):
        """
        Set the module in training mode.
        """
        self.training = True
        self._deq_manager.train()

    @property
    def f_function_of_y_x_theta(self):
        """
        Obtain the current ``f_function_of_y_x_theta`` of the class for fixed-point calculations.

        Returns
        -------
        ~collections.abc.Callable
        """
        return self._f_function_of_y_x_theta

    @f_function_of_y_x_theta.setter
    def f_function_of_y_x_theta(self, new_f):
        """
        Set a new ``f_function_of_y_x_theta`` of the class.

        Parameters
        ----------
        new_f : ~collections.abc.Callable
        Function `new_f(y, x; theta)` (notice that this is the presumed order for its variables), wherein

            - `new_f` returns a tensor of the same dimensionality of the tensor `y`
            - `x` is a tensor
            - `theta` represents the parameters ruling the shape of the function `f`
            - and the fixed point problem is defined as `y* = new_f(y*,x; theta)`.
        """
        self._f_function_of_y_x_theta = new_f

    @property
    def batched_fixed_point(self):
        """
        Obtain whether the current type of fixed point calculation is batched or not.

        Returns
        -------
        bool
        """
        return self._batched_fixed_point

    @batched_fixed_point.setter
    def batched_fixed_point(self, new_batched_fixed_point):
        self._batched_fixed_point = new_batched_fixed_point

    @property
    def f_solver(self):
        """
        Obtain the forward solver of the instance of the class.

        Returns
        -------
        The solver.
        """
        return self._f_solver

    @property
    def surrogate_f_solver(self):
        """
        Obtain the forward solver of the instance of the class.

        Returns
        -------
        The solver.
        """
        return self._surrogate_f_solver

    @property
    def b_solver(self):
        """
        Obtain the backward solver of the instance of the class.

        Returns
        -------
            The solver.
        """
        return self._b_solver

    def forward(self, x, y0=None, batched_fixed_point=None, f_tau=None, abs_error_threshold=None,
                flag_update_last_forward_info=True):
        """
        Fixed-point
        The operation of the method depends on the state :py:attr:`.batched_fixed_point` of the class and of the \
        dimensionality of ``x``, or analogously on the argument ``batched_fixed_point`` overriding it:

        - if :py:attr:`.batched_fixed_point` is ``False`` then each (trailing) 3D element of ``x`` is subject to \
          an independent fixed-point problem;

        - ``x.size(0)`` = ``y0.size(0)`` $> 1$ then the $i$-th element of ``y0`` is regarded the initial point for the \
          $i$-th element of input ``x``.

        **The function additionally provides a warning whenever the fixed point error, ** \
        **measured as the norm of the error between f(x) and x, is above the threshold ** ``abs_error_threshold``. \
        **Such warning is not returned but stored for explicit query, if desired, through the method ** \
        :py:meth:`.get_last_forward_convergence_info`. In fact, \
        the function generates, apart from the returned fixed point, the following data available through
        :py:meth:`.get_last_forward_convergence_info`:

            - the boolean Tensor indicating whether the returned fixed point  (for each input datapoint) was achieved \
              with satisfactory absolute error/convergence;,

            - the abs_error threshold used for the above assessment, 'self._last_forward_abs_error_threshold',

            - the batched-fixed-point flag used for the calculation, 'self._last_forward_batched_fixed_point'; and

            - the info (about trajectories) for the last calculations.

        Parameters
        ----------
        x : torch.Tensor
        y0 : torch.Tensor, optional
            It overrides, if provided, the value provided or initialised in the constructor.
            Default: ``None``
        batched_fixed_point : bool, optional
            It overrides, if provided, the value currently in the corresponding attribute \
            :py:attr:`.batched_fixed_point` of the class, without changing said attribute. \
            If ``None`` the mode currently indicated by the attribute is used. \
            Default: ``None``
        f_tau : float, optional
            Float in the range $(0.0, 1.0]$; if provided (i.e. not ``None``), \
            and for the solver algorithms allowing so, it overrides the value $\\tau$ provided in the constructor (see
            :py:class:`.FixedPointLayer`) during the current evaluation.
            Default: ``None``
        abs_error_threshold : float, optional
            The absolute error threshold, measured as the norm of the error between f(x) and x, \
            for the fixed point calculation. \
            If ``None`` the mode currently indicated by the attribute is used.
            Default: ``None``
        flag_update_last_forward_info : bool, optional
            If ``True`` the information about the last forward calculation is updated, \
            otherwise it is not updated.
            Default: ``True``

        Returns
        -------
        torch.Tensor
            The output of the layer, that is, the fixed point of the layer for the input `x` and
            the (current) parameters `theta` of the layer
        """

        ### SETTING PARAMETERS FOR THE FORWARD SOLVER IF NECESSARY
        current_surrogate_f_solver_kwargs = {}
        if self._surrogate_f_solver in ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson']:
            if f_tau is not None:
                if not isinstance(f_tau, (float,int)) or (f_tau <= 0.0) or (f_tau > 1.0):
                    raise ValueError(f"'f_tau' must be a scalar in the range (0.0, 1.0]!")
                else:
                    current_surrogate_f_solver_kwargs['tau'] = f_tau
                pass
            else:
                current_surrogate_f_solver_kwargs['tau'] = self._surrogate_f_tau
            pass
        pass

        # Set the initial point for the iteration
        if y0 is None:
            y0 = self.y0
        pass
        # If still no initial point: exception!!!
        if y0 is None:
            raise Exception(f"No initial point available: none provided and none available as attribute!")
        pass

        # Check dimensional compatibility between both, and make both 4D, if possible, and otherwise exception
        casted_images = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            casted_images = True
        elif x.ndim != 4:
            raise Exception(f"Input 'x' not 3D or 4D!")
        pass
        if y0.ndim == 3:
            y0 = y0.unsqueeze(0)
        elif y0.ndim != 4:
            raise Exception(f"Initial fixed point not 3D or 4D!")
        pass
        if (y0.size(0) > 1) and (y0.size(0) != x.size(0)):
            raise Exception(f"Incompatible input 'x' {x.size()} and fixed point {y0.size()} sizes!")
        pass

        # Operate differently if batched fixed point or per-sample fixed point:
        y_out = None

        self._last_forward_batched_fixed_point = \
            self._batched_fixed_point if batched_fixed_point is None else batched_fixed_point
        self._last_forward_abs_error_threshold = \
            self._abs_error_threshold if abs_error_threshold is None else abs_error_threshold
        y_out = None
        info = None

        if self._last_forward_batched_fixed_point:
            if y0.size(0) == 1:
                y0 = y0.repeat(tuple([x.size(0)]) + tuple([1, 1, 1]))
            pass
            # Particularize the function f to the current x amd theta (and leave y as variable)
            f_function_of_y_only = _generate_f_reduced_input_variables(
                self._f_function_of_y_x_theta, x, self._theta
            )
            output_y, info = self._deq_manager(f_function_of_y_only, y0, solver_kwargs=current_surrogate_f_solver_kwargs)
            y_out = output_y[-1]
        else:
            y_out = torch.empty(tuple([x.size(0)]) + y0.size()[-3:])
            list_individual_info_i = []
            for ind_in_batch in range(x.size(0)):
                # Particularize the function f to the current x amd theta (and leave y as variable)
                f_function_of_y_only = _generate_f_reduced_input_variables(
                    self._f_function_of_y_x_theta, x[ind_in_batch].unsqueeze(0), self._theta
                )
                # Solve the fixed point using the .forward of the self._deq_manager (DEQBase) object.
                output_y, info_i = self._deq_manager(f_function_of_y_only,
                                                     y0[min(ind_in_batch, y0.size(0) - 1)].unsqueeze(0),
                                                     solver_kwargs=current_surrogate_f_solver_kwargs)
                if ind_in_batch == 0:  # Move the output tensor 'y_out' to the device of 'f_function_of_y_only and '_deq_manager'
                    y_out = y_out.to(output_y[-1].device.type)
                pass
                y_out[ind_in_batch] = output_y[-1][:]
                #
                # Pack the 'info_i' elements into one single 'info'
                if ind_in_batch == 0:
                    info = {}
                    for key in info_i:
                        if key == 'sradius':  # Leave it useless
                            info[key] = 0.0
                        else:
                            field_size = list(info_i[key].size())
                            field_size[0] = x.size(0)
                            info[key] = torch.empty(tuple(field_size),
                                                    dtype=info_i[key].dtype, device=info_i[key].device)
                        pass
                    pass
                pass
                #
                for key in info_i:
                    if key != 'sradius':
                        info[key][ind_in_batch] = info_i[key][:].detach().clone()
                    pass
                pass
            pass
        pass
        #
        if flag_update_last_forward_info:
            # We assess whether all the images in the batch achieved a reasonable absolute error, which
            # we will take as an indication of convergence
            self._last_forward_warning_above_threshold_y_out = info['abs_lowest'] > self._last_forward_abs_error_threshold

            # We also store the 'info' structure of the last calculation
            self._last_forward_info = info
        pass
        #
        # If the input x was 3D, i.e. (C, H, W), and was casted to 4D, i.e. (1, C, H, W), then we un-cast it \
        # and we do the same with 'above_threshold_y_out'. WARNING: 'info' is not un-casted!!!
        if casted_images:
            y_out = y_out[0]
            if flag_update_last_forward_info:
                self._last_forward_warning_above_threshold_y_out = self._last_forward_warning_above_threshold_y_out[0]
            pass
        pass
        #
        # The function generates the output 'y_out', the fixed point, WHICH IS THE ONLY ELEMENT DIRECTLY RETURNED.
        # But it also stores the following data, available if explicitly queried by \
        # the method 'get_last_forward_convergence_info()'.
        # - the boolean Tensor indicating whether fixed point returned (for each input datapoint) was achieved \
        #   ABOVE THE INDICATED 'abs_error_threshold',
        # - the abs_error threshold used for the above assessment, 'self._last_forward_abs_error_threshold',
        # - the batched-fixed-point flag used for the calculation, 'self._last_forward_batched_fixed_point'; and
        # - the info (about trajectories) for the last calculations.

        return y_out

    def get_last_forward_convergence_info(self):
        """
        Get the convergence information for the last :py:meth:`.forward` fixed point calculation. \
        If no calculation has been performed by the object so far ``None`` is returned.

        Returns
        -------
        warning_above_threshold_y_out : torch.Tensor
            Tensor of bools indicating whether the corresponding image has reached a fixed point \
            with an absolute error above the threshold ``abs_error_threshold``
        last_forward_abs_error_threshold : float
            The absolute error threshold used for the fixed point calculation.
        info : torchdeq.SolverStat or list[torchdeq.SolverStat], optional
            Info regarding the optimization process is to be returned as well. \
            The `forward method of torchdeq.DEQIndexing <https://github.com/locuslab/torchdeq/blob/main/torchdeq/core.py#L561>`_ \
            returns a dict-like `SolverStat object <https://github.com/locuslab/torchdeq/blob/main/torchdeq/solver/stat.py#L4>`_ \
            containing information about the optimization process. If the option is ``True``, along with the output of \
            the layer, such ``info`` is returned. \
            **Important**: Said returned ``info`` is one single dictionary SolverStat if *batched_fixed_point* \
            is used, but a list of B SolverStats, each SolverStat corresponding to each input point, if \
            *batched_fixed_point* is not used.
        last_forward_batched_fixed_point : bool
            If batched fixed point calculation was used.
        """

        return (
            copy.deepcopy(self._last_forward_warning_above_threshold_y_out),
            copy.deepcopy(self._last_forward_abs_error_threshold),
            copy.deepcopy(self._last_forward_info),
            copy.deepcopy(self._last_forward_batched_fixed_point)
        )
