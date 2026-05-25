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

import torch

import torchdeq
import torchdeq.solver


####################################################################################
####################################################################################
# SUBSTITUTION OF ``torchdeq.solver.utils.batch_masked_mixing``
####################################################################################
####################################################################################

####################################################################################
# Rewrite
####################################################################################

def my_torchdeq_solver_utils_batch_masked_mixing(mask, mask_var, orig_var):
    """
    This function re-writes the original function :py:func:`torchdeq.solver.utils.batch_masked_mixing` which, \
    whenever any of the entries of the input contained any NaN, let the NaN go through even if the mask ``mask`` \
    would not be such that such value should go through: that behavior causes problems e.g. using Broyden's solver, \
    which as implemented in TorchDEQ might generate NaN values, in the function \
    :py:func:`torchdeq.solver.utils.update_state`.

    The following is exactly the documentation of the original function \
    :py:func:`torchdeq.solver.utils.batch_masked_mixing`: ::

        Applies a mask to ``mask_var`` and the inverse of the mask to ``orig_var``, then sums the result.

        Helper function. First aligns the axes of mask to mask_var.
        Then mixes mask_var and orig_var through the aligned mask.

        Args:
            mask (torch.Tensor): A tensor of shape (B,).
            mask_var (torch.Tensor): A tensor of shape (B, ...) for the mask to select.
            orig_var (torch.Tensor): A tensor of shape (B, ...) for the reversed mask to select.

        Returns:
            torch.Tensor: A tensor resulting from the masked mixture of ``mask_var`` and ``orig_var``.

        Example:
            mask = torch.tensor([True, False])
            mask_var = torch.tensor([[1, 2], [3, 4]])
            orig_var = torch.tensor([[5, 6], [7, 8]])
            result = batch_masked_mixing(mask, mask_var, orig_var)
            result
                tensor([[1, 2],
                        [7, 8]])`

    Parameters
    ----------
    mask : torch.Tensor
        Tensor of bool values ruling the selection of the values from mask_var and orig_var.
    mask_var : int or float or torch.Tensor
        Tensor of values to be selected when the mask is True. If it is a tensor its size must fit that of /
        the variable ``mask``
    orig_var : int or float or torch.Tensor
        Tensor of values to be selected when the mask is True. If it is a tensor its size must fit that of /
        the variable ``mask``

    Returns
    -------
    torch.Tensor
    """

    if torch.is_tensor(mask_var):
        # This check is necessary because the "mask_var" could be a scalar
        batch_size = mask_var.size(0)
        extra_dims = mask_var.ndim - 1
        repetition_dims = (1,) + tuple(mask_var.size()[1:]) if extra_dims > 0 else (1,)
    elif torch.is_tensor(orig_var):
        batch_size = orig_var.size(0)
        extra_dims = orig_var.ndim - 1
        repetition_dims = (1,) + tuple(orig_var.size()[1:]) if extra_dims > 0 else (1,)
    else:
        raise ValueError('Either mask_var or orig_var should be a Pytorch tensor!')
    pass
    #
    aligned_mask = mask.view((batch_size,) + (1,) * extra_dims).repeat(repetition_dims)
    #
    return torch.where(aligned_mask, mask_var, orig_var)


####################################################################################
# Replace
####################################################################################

torchdeq.solver.utils.batch_masked_mixing = my_torchdeq_solver_utils_batch_masked_mixing


####################################################################################
####################################################################################
# SUBSTITUTION OF ``torchdeq.solver.utils.batch_masked_mixing``
####################################################################################
####################################################################################

####################################################################################
# Rewrite
####################################################################################

def my_torchdeq_solver_broyden_line_search(update, x0, g0, g, nstep=0, on=True):
    """
    This function re-writes the original function :py:func:`torchdeq.solver.broyden.line_search` which, \
    whenever any of the entries of the optimization direction `update` is NaN, propagates the NaN to the \
    updated value. Our modification detects when an entry is NaN, in that case, makes that variation zero.

    The following is exactly the documentation of the original function \
    :py:func:`torchdeq.solver.broyden.line_search`: ::

        `update` is the propsoed direction of update.

        Code adapted from scipy.

    Parameters
    ----------
    update : torch.Tensor
    x0 : torch.Tensor
    g0 : torch.Tensor
    g : callable
    nstep : int
    on : bool

    Returns
    -------
    torch.Tensor
    """

    # By construction "update" and "x0" have 2 dimensions: bsz and dim
    # Check which points of the batch have a NaN update
    # Check which points have a NaN update
    nan_update = torch.isnan(update).any(dim=tuple(range(1, update.ndim)))
    update[nan_update] = 0.0

    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_guess = x0 + s * update
        g0_new = g(x_guess)
        phi_new = torchdeq.solver.broyden._safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = torchdeq.solver.broyden.scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or (s is None):
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite
    ###############################


####################################################################################
# Replace
####################################################################################

torchdeq.solver.broyden.line_search = my_torchdeq_solver_broyden_line_search
