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

from math import floor, ceil
import warnings
from abc import ABC, abstractmethod
import inspect

import torch
from torch import nn
import torch.nn.functional as F

from modified_rf.fixed_point import FixedPointLayer

from modified_rf.nn_layers import SMLayer, INRFv2Layer, INRFv1Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer
from modified_rf.nn_layers.utils import (conv2d_adapted,
                                         conv2d_crossdiff,
                                         f_modified_RF)
from modified_rf.nn_layers.utils import (ndim_activation_function_from_1dim_activation_functions,
                                         cast_scalar_like_to_image)

import gc
import modified_rf.memory_handling as memo


######################################################
### INRFv1 operation
######################################################

def inrfv1_function(im, theta,
                    phi_activation='relu', sigma_activation='tanh',
                    m_padding='same', m_padding_mode='zeros', m_groups=1,
                    **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.INRFv1Layer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``'interpolated'``
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``True``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor
        Output of the ibnn_internal implicit function (not fixed point solution, only the computation of the implicit function)
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    inrfv1 = INRFv2Layer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        **kwargs
    )

    for field in inrfv1.theta_copy:
        inrfv1.set_field_in_theta(theta[field], field)
    pass

    return inrfv1(im)


######################################################
### INRFv2 operation
######################################################

def inrfv2_function(im, theta,
                    phi_activation='relu', sigma_activation='tanh',
                    m_padding_mode='zeros', m_groups=1,
                    w_padding_mode='zeros', w_groups=1,
                    **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.INRFv2Layer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
            - ``theta['w']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter of the non-linearities of the implicit bias
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    w_padding_mode : : str, optional
        Always same size of the input int this case, the mode adopts a value among
        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    w_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``'interpolated'``
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``True``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor
        Output of the ibnn_internal implicit function (not fixed point solution, only the computation of the implicit function)
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    inrfv2 = INRFv2Layer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        w_kernel_size=theta['w'].size()[2:], w_padding_mode=w_padding_mode, w_groups=w_groups,
        w_initialization='zeros',
        w_trainable=False,
        **kwargs
    )

    for field in inrfv2.theta_copy:
        inrfv2.set_field_in_theta(theta[field], field)
    pass

    return inrfv2(im)


######################################################
### INRFv2 operation
######################################################

def inrfv3_function(im, theta,
                    phi_activation='relu', sigma_activation='tanh',
                    m_padding_mode='zeros', m_groups=1,
                    w_padding_mode='zeros', w_groups=1,
                    **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.INRFv3Layer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
            - ``theta['w']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter of the non-linearities of the implicit bias
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    w_padding_mode : : str, optional
        Always same size of the input int this case, the mode adopts a value among
        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    w_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``'interpolated'``
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``True``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor
        Output of the ibnn_internal implicit function (not fixed point solution, only the computation of the implicit function)
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    inrfv3 = INRFv3Layer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        w_kernel_size=theta['w'].size()[2:], w_padding_mode=w_padding_mode, w_groups=w_groups,
        w_initialization='zeros',
        w_trainable=False,
        **kwargs
    )

    for field in inrfv3.theta_copy:
        inrfv3.set_field_in_theta(theta[field], field)
    pass


    return inrfv3(im)


######################################################
### ibnn_lite operation
######################################################

def ibnn_lite_function(im, theta,
                    phi_activation='relu', sigma_activation='tanh',
                    m_padding='same', m_padding_mode='zeros', m_groups=1,
                    w_padding_mode='zeros', w_groups=1,
                    **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.IBNNLiteLayer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
            - ``theta['w']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter of the non-linearities of the implicit bias
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    w_padding_mode : : str, optional
        Always same size of the input int this case, the mode adopts a value among
        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    w_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``'interpolated'``
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: ``True``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor
        Output of the ibnn_internal implicit function (not fixed point solution, only the computation of the implicit function)
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    ibnn_lite = IBNNLiteLayer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        w_kernel_size=theta['w'].size()[2:], w_padding_mode=w_padding_mode, w_groups=w_groups,
        w_initialization='zeros',
        w_trainable=False,
        **kwargs
    )

    for field in ibnn_lite.theta_copy:
        ibnn_lite.set_field_in_theta(theta[field], field)
    pass

    return ibnn_lite(im)




######################################################
### ibnn_internal operation
######################################################

def ibnn_function(im, theta, phi_activation='relu', sigma_activation='tanh',
                  m_padding='same', m_padding_mode='zeros', m_groups=1,
                  w_padding_mode='zeros', w_groups=1,
                  f_solver='broyden', b_solver='broyden',
                  f_tol=1e-8, b_tol=1e-6,
                  flag_print_residual=False, flag_print_residual_trace=False, flag_return_residual=False,
                  **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.IBNNInternalLayer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
        Depending on the input (at least ``u`` different than ``None``) it must contain also:
            - ``theta['w']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter of the non-linearities of the implicit bias
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    w_padding_mode : : str, optional
        Always same size of the input int this case, the mode adopts a value among
        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    w_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    batched_fixed_point : bool, optional
        It indicates whether the calculation in :py:meth:`.forward` of the fixed point  for an input batch
        of $B$ elements (that is, for a 4D input) should operate on each one of its $B$ individual 3D entries
        (if ``False``) or, on the contrary (if ``True``), on the 4D input as one single item.
        Default: see :py:class:`.IBNNInternalLayer`
    abs_error_threshold : float, optional
        The absolute error threshold, measured as the norm of the error between f(x) and x, \
        for the fixed point calculation.
        Default: see :py:class:`.IBNNInternalLayer`
    ift : bool, optional
        Default: see :py:class:`.IBNNInternalLayer`
    hook_ift : bool, optional:
        Default: see :py:class:`.IBNNInternalLayer`
    f_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the forward fixed point calculation.
        Default: see :py:class:`.IBNNInternalLayer`
    b_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the fixed point calculation \
        used in the gradient calculation of the IFT.
        Default: see :py:class:`.IBNNInternalLayer`
    f_tol : float
        Stop condition for the forward fixed point calculation.
        Default: see :py:class:`.IBNNInternalLayer`
    b_tol : float
        Stop condition for the backward fixed point calculation.
        Default: see :py:class:`.IBNNInternalLayer`
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: see :py:class:`.IBNNInternalLayer`
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: see :py:class:`.IBNNInternalLayer`
    flag_print_residual : bool, optional
        Default: ``False``
    flag_print_residual_trace : bool, optional
        Default: ``False``
    flag_return_residual : bool, optional
        Default: ``False``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor, float]
        Output of the ibnn_internal implicit function.
        If ``flag_return_residual`` is ``True``, it returns a tuple with the residual image and the maximum residual.
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    ibnn_internal = IBNNInternalLayer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        w_kernel_size=theta['w'].size()[2:], w_padding_mode=w_padding_mode, w_groups=w_groups,
        w_initialization='zeros',
        w_trainable=False,
        f_solver=f_solver, b_solver=b_solver,
        f_tol=f_tol, b_tol=b_tol,
        **kwargs
    )

    for field in ibnn_internal.theta_copy:
        ibnn_internal.set_field_in_theta(theta[field], field)
    pass

    ibnn_im = ibnn_internal(im)

    if flag_print_residual_trace:
        print(f"-----")
    pass

    if flag_print_residual or flag_return_residual:
        # Assess the residual of the fixed point solution
        output_function = f_modified_RF(
            im, theta, u=ibnn_im,
            phi_activation=ibnn_internal.phi_activation, sigma_activation=sigma_activation,
            m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            w_padding_mode=w_padding_mode, w_groups=w_groups,
            **kwargs
        )
        residual_image = ibnn_im - output_function
        max_residual = torch.max(torch.abs(residual_image)).item()
        median_residual = torch.median(torch.abs(residual_image)).item()
        norm_residual = torch.norm(residual_image).item()

        if flag_print_residual:
            print(f"Max residual: {max_residual:.3e} (median: {median_residual:.3e}; norm: {norm_residual:.3e})")
    pass

    if flag_print_residual_trace:
        print(f"Convergence info as trace of the norm of the residual: ", end="")
        print(",    ".join([
            f"{ind}: {v:.1e}" for ind, v in enumerate(
                ibnn_internal.get_last_forward_convergence_info()[2]['abs_trace'].flatten().tolist()
            )
        ]))
        print(f"-----")
    pass

    structure_to_return = ibnn_im if not flag_return_residual else (ibnn_im, residual_image, max_residual)

    return structure_to_return


######################################################
### ibnn_internal operation
######################################################

def ibnnx_function(
    im, theta, phi_activation='relu', sigma_activation='tanh',
    m_padding='same', m_padding_mode='zeros', m_groups=1,
    w_padding_mode='zeros', w_groups=1,
    f_solver='broyden', b_solver='broyden',
    f_tol=1e-8, b_tol=1e-6,
    flag_print_residual=False, flag_print_residual_trace=False, flag_return_residual=False,
    **kwargs):
    """
    Applies the same operation performed by the layer :py:class:`.IBNNLayer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
        Depending on the input (at least ``u`` different than ``None``) it must contain also:
            - ``theta['w']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter of the non-linearities of the implicit bias
            - ``theta['lambda']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the weight of the complete implicit bias, which is considered \
            either a scalar or a vector with the same number of channels as u.
        And optionally, in the latter case, it can also contain some of the following modifiers of basic $\\sigma$ \
        (see :py:func:`.ndim_activation_function_from_1dim_activation_functions`):
            - ``theta['sigma_x_compress']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_stretch']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_x_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
            - ``theta['sigma_y_offset']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'tanh'``
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    w_padding_mode : : str, optional
        Always same size of the input int this case, the mode adopts a value among
        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    w_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    batched_fixed_point : bool, optional
        It indicates whether the calculation in :py:meth:`.forward` of the fixed point  for an input batch
        of $B$ elements (that is, for a 4D input) should operate on each one of its $B$ individual 3D entries
        (if ``False``) or, on the contrary (if ``True``), on the 4D input as one single item.
        Default: see :py:class:`.IBNNLayer`
    abs_error_threshold : float, optional
        The absolute error threshold, measured as the norm of the error between f(x) and x, \
        for the fixed point calculation.
        Default: see :py:class:`.IBNNLayer`
    ift : bool, optional
        Default: see :py:class:`.IBNNLayer`
    hook_ift : bool, optional:
        Default: see :py:class:`.IBNNLayer`
    f_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the forward fixed point calculation.
        Default: see :py:class:`.IBNNLayer`
    b_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the fixed point calculation \
        used in the gradient calculation of the IFT.
        Default: see :py:class:`.IBNNLayer`
    f_tol : float
        Stop condition for the forward fixed point calculation.
        Default: see :py:class:`.IBNNLayer`
    b_tol : float
        Stop condition for the backward fixed point calculation.
        Default: see :py:class:`.IBNNLayer`
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by
        :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity of the underlying calculations
        in the order of $S \\times N^2$ (for $N$ the side of the input image(s) and $S$ the number of interpolation
        points, and disregarding the contribution of the convolution kernel), or using the exact calculation of
        complexity in the order of $N^4$. Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: see :py:class:`.IBNNLayer`
    memory_saving_version : bool, optional
        Relevant for the function :py:func:`.conv2d_crossdiff`.
        Default: see :py:class:`.IBNNLayer`
    flag_print_residual : bool, optional
        Default: ``False``
    flag_print_residual_trace : bool, optional
        Default: ``False``
    flag_return_residual : bool, optional
        Default: ``False``
    **kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information), used internally by
        the function :py:func:`.conv2d_crossdiff` used in turn internally in this function:

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **num_sampling_points** : `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **start_range_std_activation**, **end_range_std_activation** : \
        `float <https://docs.python.org/3/library/stdtypes.html#float>`_ or \
        `int <https://docs.python.org/3/library/stdtypes.html#int>`_, optional

            Range of the standardized activation function `activation_function`, that is, with \
            `x_compress` $=1$ and `x_offset` $=0$. The effective range wherein the `num_sampling_points` \
            samples of the interpolation are drawn are the absolute values \
            $\\textrm{start_range} = \\frac{\\textrm{start_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ \
            and $\\textrm{end_range} = \\frac{\\textrm{end_range_std_activation}}{\\textrm{x_compress}}+\\textrm{x_offset}$ . \
            Default: see :py:func:`.generate_builder_1_order_interpolation` for the default absolute range

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor, float]
        Output of the ibnn_internal implicit function.
        If ``flag_return_residual`` is ``True``, it returns a tuple with the residual image and the maximum residual.
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'
    lambda_type = 'scalar' if theta['lambda'].numel == 1 else 'scalar_per_channel'

    ibnn = IBNNLayer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
        sigma_activation=sigma_activation,
        sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
        sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
        sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
        lambda_type=lambda_type, initial_lambda=0.0, lambda_trainable=False,
        w_kernel_size=theta['w'].size()[2:], w_padding_mode=w_padding_mode, w_groups=w_groups,
        w_initialization='zeros',
        w_trainable=False,
        f_solver=f_solver, b_solver=b_solver,
        f_tol=f_tol, b_tol=b_tol,
        **kwargs
    )

    for field in ibnn.theta_copy:
        ibnn.set_field_in_theta(theta[field], field)
    pass

    ibnnx_im = ibnn(im)

    if flag_print_residual_trace:
        print(f"-----")
    pass

    if flag_print_residual or flag_return_residual:
        # Assess the residual of the fixed point solution
        output_function = f_modified_RF(
            im, theta, u=ibnnx_im,
            phi_activation=ibnn.phi_activation, sigma_activation=sigma_activation,
            m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            w_padding_mode=w_padding_mode, w_groups=w_groups,
            **kwargs
        )
        residual_image = ibnnx_im - output_function
        max_residual = torch.max(torch.abs(residual_image)).item()
        median_residual = torch.median(torch.abs(residual_image)).item()
        norm_residual = torch.norm(residual_image).item()

        if flag_print_residual:
            print(f"Max residual: {max_residual:.3e} (median: {median_residual:.3e}; norm: {norm_residual:.3e})")
    pass

    if flag_print_residual_trace:
        print(f"Convergence info as trace of the norm of the residual: ", end="")
        print(",    ".join([
            f"{ind}: {v:.1e}" for ind, v in enumerate(
                ibnn.get_last_forward_convergence_info()[2]['abs_trace'].flatten().tolist()
            )
        ]))
        print(f"-----")
    pass

    structure_to_return = ibnnx_im if not flag_return_residual else (ibnnx_im, residual_image, max_residual)

    return structure_to_return


######################################################
### SM operation
######################################################

def sm_function(im, theta,
                phi_activation='relu',
                m_padding='same', m_padding_mode='zeros', m_groups=1):
    """
    Applies the same operation performed by the layer :py:class:`.SMLayer`: see the latter for details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements:
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`.
        Default: ``'relu'``
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    m_groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``

    Returns
    -------
    torch.Tensor
        Output of the function
    """

    b_type = 'scalar' if theta['b'].numel == 1 else 'scalar_per_channel'

    sm = SMLayer(
        im.size(-3), theta['m'].size(-4), phi_activation, theta['m'].size()[2:],
        m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
        m_initialization='zeros', m_trainable=False,
        b_type=b_type, initial_b=0.0, b_trainable=False,
        in_size=im.size(),
    )

    for field in sm.theta_copy:
        sm.set_field_in_theta(theta[field], field)
    pass

    return sm(im)