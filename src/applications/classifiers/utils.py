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

from collections import OrderedDict
import copy
import warnings


import numpy as np

import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize

from modified_rf import _dict_conv_like_layers
from modified_rf import processed_constructor_kwargs_for_conv_like_layer
from modified_rf import kernel_size_check_and_reformat_into_tuple, resolve_kernel_size_for_im_size


#########################################################################################
#########################################################################################
#########################################################################################
# PARAMETERIZATIONS FOR THE LAYERS
#########################################################################################
#########################################################################################
#########################################################################################

class PencilOfPlanes(nn.Module):
    """
    This module implements a parameterization for the weights of a :py:class:`~nn.torch.Linear` layer that \
    ensures that the resulting planes of decision (disregarding the bias) are a pencil of planes \
    around a given spine direction; this is achieved by making the weights of the layer \
    always orthogonal to the spine direction, which is achieved by subtracting the projection \
    of the weights onto the spine direction.
    The default spine direction, expressed by `None`, is the vector $\\frac{1}{\\sqrt{N}}\\mathbb{1}_{N}$.

    Parameters
    ----------
    spine_direction : torch.Tensor, optional
        A 1D tensor representing the spine direction. If `None`, the default spine direction is used, \
        which is a vector of ones normalized to have norm 1.
        Default: `None`.
    """

    def __init__(self, spine_direction=None):
        super().__init__()
        assert spine_direction is None or (
                isinstance(spine_direction, torch.Tensor) and spine_direction.ndim == 1), \
            "spine_direction must be a 1D tensor or None"
        # If it is a vector, reformat it to a norm-1 column vector; if it is None, leave it as None (easier to handle)
        if spine_direction is not None:
            self.spine_direction = spine_direction / spine_direction.norm()
        pass
        # Register buffer: it makes sure that the spine direction is not considered a parameter of the model, \
        # but it is still saved and loaded with the model state
        self.register_buffer("spine_direction", spine_direction)

    def forward(self, w):
        parameterized_w = None
        if self.spine_direction is None:
            parameterized_w = w - torch.mean(w, dim=0, keepdim=True)
        else:
            parameterized_w = w - torch.dot(w, self.spine_direction) * self.spine_direction
        pass
        return parameterized_w


#########################################################################################
#########################################################################################
#########################################################################################
# AUXILIARY FUNCTIONS
#########################################################################################
#########################################################################################
#########################################################################################

def _create_standard_FC_blocks(num_features_in, num_features_out,
                               num_layers=1, num_features_intermediate=None,
                               batch_normalization=True, dropout=None,
                               penciled_decision=False, softmax_output=False):
    """
    It creates a list of tuples, where each tuple contains the block name and the :py:class:`torch.nn.Sequential`, \
    corresponding to FC blocks according to the given parameters. And each block is as follows:

    - in general: intermediate blocks contain a nn.Linear layer from `num_features_intermediate` to \
      `num_features_intermediate` features, followed by an activation function (ReLU by default) \
      and a batch normalization layer...
    - if they are not the last: the last block contains only the nn.Linear layer, without any activation \
      function or normalization, leading to the output of the network, which is a tensor of `num_features_out` features;
    - and if they are not the first block: the first block takes `num_features_in` at their input.

    If the flag `penciled_decision` is set to ``True``, then the first block will be parameterized by a \
    :py:class:`.PencilOfPlanes` layer and will have zero offset, \
    which will ensure that the decision planes are a pencil of planes \
    around the given spine direction (1,...,1); it defaults to ``False``,
    i.e. the first block is a usual affine FC layer.

    If the argument `softmax_output` is set as `True` then a final exit layer :py:class:`~torch.nn.Softmax` \
    is added at the end of the list.

    Parameters
    ----------
    num_features_in : int
        Number of features in the input of the first block
    num_features_out : int
        Number of features in the output of the last block
    num_layers : int, optional
        Number of layers in the block, i.e. number of intermediate blocks to be created. \
        Default: 1 (i.e. no intermediate blocks)
    num_features_intermediate : int, optional
        (When `num_layers` > 1) Number of features in the intermediate blocks, no effect if `num_layers` <= 1.
        Default: ``None``
    batch_normalization : bool, optional
        If ``True``, a batch normalization layer is added after each intermediate block.
        Default: ``True``
    dropout : float, optional
        If provided, it will be used as the dropout probability after each intermediate block.
        Default: ``None`` (i.e. no dropout)
    penciled_decision : bool, optional
        Default: ``False``
    softmax_output : bool, optional
        Default: ``False``


    Returns
    -------
    list
        List of 2D tuples composed of name and layer
    """

    ####################################
    # Initial checks
    ####################################

    assert isinstance(num_features_in, int) and num_features_in > 0, \
        f"Invalid 'num_features_in': {num_features_in} found, expected a positive integer."
    assert isinstance(num_features_out, int) and num_features_out > 0, \
        f"Invalid 'num_features_out': {num_features_out} found, expected a positive integer."

    assert isinstance(num_layers, int), \
        f"Invalid 'num_layers': {num_layers}, of type {type(num_layers)}, found; expected a non-negative integer."
    if num_layers > 1:
        assert isinstance(num_features_intermediate, int) and num_features_intermediate > 0, \
            f"Invalid 'num_features_intermediate': {num_features_intermediate} found, " + \
            f"expected a positive integer when 'num_layers' > 1."
    elif num_layers == 1: # In this case, a warning about the parameters that will not be used
        warnings.warn("'num_layers' = 1, so 'num_features_intermediate', 'batch_normalization', and 'dropout' " +
                      "will not be used.")
    else:
        raise Exception(f"Invalid 'num_layers': {num_layers} found, expected a positive (>0) integer.")
    pass

    assert isinstance(batch_normalization, bool), \
        f"Invalid 'batch_normalization': {batch_normalization} found, expected a boolean."
    assert isinstance(dropout, (type(None), float)) and (dropout is None or (0 <= dropout <= 1)), \
        f"Invalid 'dropout': {dropout} found, expected a float in the range [0, 1] or None."

    assert isinstance(penciled_decision, bool), "Invalid 'penciled_decision' parameter, expected a boolean."

    assert isinstance(softmax_output, bool), "Invalid 'softmax_output' parameter, expected a boolean."

    ####################################
    # Creation of the list
    ####################################

    list_of_named_layers = []

    # FC blocks

    for i in range(num_layers):
        #
        ### BLOCK (i):
        #
        flag_last_layer = False if (i < (num_layers - 1)) else True
        num_features_in_block_i = num_features_in if i == 0 else num_features_intermediate
        num_features_out_block_i = num_features_intermediate if not flag_last_layer else num_features_out
        #
        ordered_dict_block_i = OrderedDict()
        if i == 0 and penciled_decision:
            # If it is the first block and penciled decision, use PencilOfPlanes for parameterization:
            # BIAS=FALSE FOR NO OFFSET!
            linear_layer_to_parametrize = nn.Linear(num_features_in_block_i, num_features_out_block_i, bias=False)
            parametrize.register_parametrization(linear_layer_to_parametrize, "weight", PencilOfPlanes())
            ordered_dict_block_i.update({f"fc_layer_{i}": linear_layer_to_parametrize})
        else:
            # Otherwise
            ordered_dict_block_i.update(
                {f"fc_layer_{i}": nn.Linear(num_features_in_block_i, num_features_out_block_i, bias=True)}
            )
        pass
        #
        if not flag_last_layer:
            ordered_dict_block_i.update({f"relu_{i}": nn.ReLU()})
            if batch_normalization:
                ordered_dict_block_i.update({f"norm_{i}": nn.BatchNorm1d(num_features_out_block_i,
                                                                         affine=True, track_running_stats=True)})
            pass
            if dropout is not None:
                ordered_dict_block_i.update({f"dropout_{i}": nn.Dropout(p=dropout)})
            pass
        pass
        #
        # Create and append the block to the list
        list_of_named_layers.append(
            (f"fc_block_{i}", nn.Sequential(OrderedDict(ordered_dict_block_i)))
        )
        #
    pass

    ####################################

    return list_of_named_layers


def _create_standard_head_module(num_features_in, num_features_out,
                                 num_layers=1, num_features_intermediate_layers=None,
                                 batch_normalization=True, dropout=None,
                                 penciled_decision=False, softmax_output=False):
    """
    It creates a :py:class:`~torch.nn.Sequential` containing a :py:class:`~torch.nn.Flatten` layer, \
    the blocks defined by the function \
    :py:func:`._create_standard_FC_blocks` plus an exit layer defined by the indicated flag `softmax_output`: \
    if ``True``, precisely a softmax, but if ``False``, simply the identity.

    If the flag `penciled_decision` is set to ``True``, then the first block will be parameterized by a \
    :py:class:`.PencilOfPlanes` layer and will have zero offset, \
    which will ensure that the decision planes are a pencil of planes \
    around the given spine direction (1,...,1); it defaults to ``False``,
    i.e. the first block is a usual affine FC layer.

    Parameters
    ----------
    num_features_in : int
        Number of features in the input of the first block
    num_features_out : int
        Number of features in the output of the last block
    num_layers : int, optional
        Number of layers in the block, i.e. number of intermediate blocks to be created. \
        Default: 1 (i.e. no intermediate blocks)
    num_features_intermediate_layers : int, optional
        (When `num_layers` > 1) Number of features in the intermediate blocks, no effect if `num_layers` <= 1.
        Default: ``None``
    batch_normalization : bool, optional
        If ``True``, a batch normalization layer is added after each intermediate block
        Default: ``True``
    dropout : float, optional
        If provided, it will be used as the dropout probability after each intermediate block.
        Default: ``None`` (i.e. no dropout)
    penciled_decision : bool, optional
        Default: ``False``
    softmax_output : bool, optional
        Default: ``False``

    Returns
    -------
    torch.nn.Sequential
        Sequential with the blocks defined by the function.
    """

    ####################################
    # Checks are performed in :py:func:`._create_standard_FC_blocks`)
    ####################################
    # Creation of the list
    ####################################

    list_of_named_layers = [('flatten', nn.Flatten())]

    # Only FC blocks
    list_of_named_layers.extend(
        _create_standard_FC_blocks(
            num_features_in=num_features_in, num_features_out=num_features_out,
            num_layers=num_layers, num_features_intermediate=num_features_intermediate_layers,
            batch_normalization=batch_normalization, dropout=dropout,
            penciled_decision=penciled_decision, softmax_output=softmax_output
        )
    )

    # Add the exit layer
    if softmax_output:
        list_of_named_layers.append(("exit", nn.Softmax(dim=-1)))
    else:
        list_of_named_layers.append(("exit", nn.Identity()))
    pass

    ####################################

    return nn.Sequential(OrderedDict(list_of_named_layers))


def _create_standard_block_conv_i(conv_like_type,
                                  in_channels, out_channels,
                                  m_kernel_size, m_independent_channels, w_independent_channels=None,
                                  in_size=None, ind_block=None, maxpool_reduction=1, batch_normalization=True,
                                  **rest_constructor_kwargs):
    """
    Creation of a :py:class:`torch.nn.Sequential` block, with a name `block_conv_i`, where *i* will correspond to \
    the index of the block `ind_block` in the network if provided (or to the string ``'i'`` if the index is not \
    provided.

    Regarding the image sizes: `in_size` is not always compulsory, since it will not be used for \
    filters defined in absolute (pixel) sizes, but \
    it is compulsory if the filters are defined in relative sizes (i.e. floats in the range [0, 1]) and the \
    subsequent functions will in fact complain; however, in both cases, the returned `out_size` can only be \
    generated if the `in_size` is given, otherwise it will be returned as `None`.

    Parameters
    ----------
    conv_like_type : str
        One of ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, or ``'ibnn'``
    in_channels : int
    out_channels : int
    m_kernel_size : tuple[int] (2D) or list[tuple[int]] (2D) or float or int
    m_independent_channels : bool
    w_independent_channels : bool
        Not optional if `conv_like_type` is in [``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``] \
        and ignored otherwise.
    in_size : tuple[int] (2D), optional
        Optional for filters defined in absolute (pixel) sizes, compulsory for filters defined in relative sizes.
        Default: ``None``
    ind_block : int or str, optional
        Index of the block in the network, or a string to be used as the name of the block.
        Default: ``None`` (i.e. the name of the block will be ``'i'``)
    maxpool_reduction : int, optional
        If greater than 1, the indication is the reduction factor used at the end of each and every block indicated \
        in the net specification `conv_block_specification`.
        Default: ``1`` (no reduction)
    batch_normalization : bool, optional
        Default: `True`
    rest_constructor_kwargs
        The rest of the keyword arguments which are regarded as relevant for the definition of the conv-like layer

    Returns
    -------
    block_conv_i : torch.nn.Sequential
    name_block_conv_i : str
    out_size : tuple of int
    """

    ####################################
    # Initial checks
    ####################################

    assert isinstance(conv_like_type, str), \
        f"Invalid 'conv_like_type': {conv_like_type} found, expected a string."
    assert conv_like_type in _dict_conv_like_layers, \
        f"Invalid 'conv_like_type': {conv_like_type} found, expected one of {list(_dict_conv_like_layers.keys())}."

    assert isinstance(in_channels, int) and in_channels > 0, \
        f"Invalid 'in_channels': {in_channels} found, expected a positive integer."
    assert isinstance(out_channels, int) and out_channels > 0, \
        f"Invalid 'out_channels': {out_channels} found, expected a positive integer."

    assert isinstance(m_kernel_size, (tuple, list, float, int)), \
        f"Invalid 'm_kernel_size': {m_kernel_size} found, expected a tuple of 2 integers, a list of tuples of 2 " + \
        f"integers, a float in the range [0, 1], or an integer > 0."

    assert isinstance(m_independent_channels, bool), \
        f"Invalid 'm_independent_channels': {m_independent_channels} found, expected a boolean."

    if conv_like_type in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
        assert isinstance(w_independent_channels, bool), \
            f"Invalid 'w_independent_channels': {w_independent_channels} found, expected a boolean."
    pass

    if in_size is not None:
        assert isinstance(in_size, (tuple, list)) and len(in_size) == 2 \
               and all(isinstance(x, int) and x > 0 for x in in_size), \
            f"Invalid 'in_size': {in_size} found, expected a tuple of 2 positive integers."
    pass

    if ind_block is not None:
        assert isinstance(ind_block, (int, str)), \
            f"Invalid 'ind_block': {ind_block} found, expected an integer or a string."
    pass

    assert isinstance(maxpool_reduction, int) and maxpool_reduction >= 1, \
        f"Invalid 'maxpool_reduction': {maxpool_reduction} found, expected an integer >= 1."

    assert isinstance(batch_normalization, bool), \
        f"Invalid 'batch_normalization': {batch_normalization} found, expected a boolean."

    ####################################
    # Fill the parameters of the convolutional layer
    ####################################
    conv_like_type_layer_i = copy.deepcopy(conv_like_type)
    # If the conv_like_type is not 'sm', check lambda value and if it is 0, set the conv_like_type to 'sm'
    if conv_like_type_layer_i != 'sm':
        if 'initial_lambda' in rest_constructor_kwargs and isinstance(rest_constructor_kwargs['initial_lambda'], list) and len(rest_constructor_kwargs['initial_lambda']) != 1:
            if rest_constructor_kwargs['initial_lambda'][ind_block] == 0.0:
                conv_like_type_layer_i = 'sm'
        pass
    # Fields in the constructor of the layers to leave out: the fields that will be filled individually per layer
    fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'm_kernel_size', 'm_groups', 'w_groups']
    conv_kwargs = processed_constructor_kwargs_for_conv_like_layer(_dict_conv_like_layers[conv_like_type_layer_i],
        fields_leave_out=fields_leave_out,
        dict_explicit_args=None, dict_non_explicit_args=rest_constructor_kwargs,
        flag_add_all_other_args=False if conv_like_type_layer_i == 'sm' else True)
    kwargs_layer_i = copy.deepcopy(conv_kwargs)


    # if conv_like_type_layer_i == 'sm':
    #     # If it is 'sm', we do not need the w_kernel_size, so we remove it
    #     if 'w_kernel_size' in kwargs_layer_i:
    #         del kwargs_layer_i['w_kernel_size']
    #     pass
    #     # And we set the padding to 'same'
    #     kwargs_layer_i['m_padding'] = 'same'

    kwargs_layer_i['in_size'] = in_size
    kwargs_layer_i['in_channels'] = in_channels
    kwargs_layer_i['out_channels'] = out_channels
    kwargs_layer_i['m_kernel_size'] = m_kernel_size
    kwargs_layer_i['m_groups'] = kwargs_layer_i['out_channels'] \
        if m_independent_channels else 1

    if conv_like_type_layer_i in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
        kwargs_layer_i['w_groups'] = kwargs_layer_i['out_channels'] \
            if w_independent_channels else 1
        if 'initial_lambda' in conv_kwargs and isinstance(conv_kwargs['initial_lambda'], list) and len(conv_kwargs['initial_lambda']) != 1:
            kwargs_layer_i['initial_lambda'] = conv_kwargs['initial_lambda'][ind_block]
    pass

    ####################################
    # Create the layer and the whole block
    ####################################

    suffix_to_add = f"{ind_block}" if ind_block is not None else 'i'

    name_block_conv_i = f"block_conv_{suffix_to_add}"
    name_conv_i = f"conv_{suffix_to_add}"
    name_norm_conv_i = f"norm_conv_{suffix_to_add}"
    name_maxpool_conv_i = f"maxpool_conv_{suffix_to_add}"

    # Create the conv layer
    conv_like_layer_i = _dict_conv_like_layers[conv_like_type_layer_i](**kwargs_layer_i)

    # Create the normalization layer
    norm_layer_i = nn.Identity() if not batch_normalization else \
        nn.BatchNorm2d(num_features=out_channels, affine=True, track_running_stats=True)

    # Create the maxpool layer
    maxpool_layer_i = nn.Identity() if maxpool_reduction == 1 else nn.MaxPool2d(kernel_size=maxpool_reduction)

    # And pack in the block
    block_conv_i = nn.Sequential(OrderedDict([
        (name_conv_i, conv_like_layer_i),
        (name_norm_conv_i, norm_layer_i),
        (name_maxpool_conv_i, maxpool_layer_i)
    ]))

    ####################################
    # Calculate the output size of the block:
    # WARNING: IT STRONGLY DEPENDS ON THE 'm_padding'!!!
    ####################################

    out_size = None

    if in_size is not None:
        # Effect of the convolution with m
        out_size = None
        if conv_like_layer_i.m_padding == 'fc':
            out_size = m_kernel_size
        elif conv_like_layer_i.m_padding == 'same':
            out_size = in_size
        elif conv_like_layer_i.m_padding == 'valid':
                out_size = tuple([in_size[i] - conv_like_layer_i.theta_copy['m'].size(i) + 1 for i in [-2, -1]])
        else:
            raise Exception((f"Unknown padding {conv_like_layer_i.m_padding} (only 'fc', 'valid', and 'same' known), " +
                             f"unable to determine out_size in the layer."))
        pass
        # Effect of the maxpooling (if any)
        if maxpool_reduction != 1:
            out_size = tuple([elem // maxpool_reduction for elem in out_size])
        pass
    pass

    ####################################

    return block_conv_i, name_block_conv_i, out_size


def _create_standard_backbone_module(
        in_size, in_channels,
        conv_like_type, conv_like_type_position,
        conv_block_specification, channels_per_conv_layer,
        phi_activation_per_conv_layer, m_kernel_size_per_conv_layer,
        batch_normalization_per_conv_layer=True,
        maxpool_reduction_per_conv_block=1, kernel_maxpool_reduction_per_conv_block=None,
        dilation_maxpool_reduction_per_conv_block=1,
        m_independent_channels=False, w_kernel_size=None, w_independent_channels=True,
        **kwargs):

    """
    It creates a :py:class:`~torch.nn.Sequential` containing :py:class:`~torch.nn.Module` blocks connected \
    sequentially, each one named `block_conv_i` (where $i$ will correspond to the index of the block `ind_block` \
    in the network) and containing, in turn:

    - a number of conv-like layers defined by the $i$-th element of `conv_block_specification`,
    - a maxpooling layer with a reduction factor `maxpool_reduction`, and
    - a batch normalization layer if `batch_normalization` is set to ``True``.

    WARNING: 'm_padding' is set to 'same' for all the convolutional-like layers.

    Parameters
    ----------
    in_size : tuple of int (2D)
    in_channels : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str
    conv_block_specification : tuple|list[int]
    channels_per_conv_layer : tuple|list[int]
    phi_activation_per_conv_layer : tuple|list[ str ]
        Activation functions used in the convolutional-like layers
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ]
    batch_normalization_per_conv_layer : bool or tuple|list[bool], optional
        Default: ``True``
    maxpool_reduction_per_conv_block : int or tuple|list[int], optional
        Default: ``1`` (1 for all blocks)
    kernel_maxpool_reduction_per_conv_block : int or tuple|list[int], optional
        Default: ``None`` (identical to `maxpool_reduction_per_conv_block` for all blocks)
    dilation_maxpool_reduction_per_conv_block : int or tuple|list[int], optional
        Default: ``1`` (1 for all blocks)
    m_independent_channels : bool, optional
        Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float, optional
        Not optional if `conv_like_type` is in [``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``]
    w_independent_channels : bool, optional
        Default: ``True``
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.

    Returns
    -------
    backbone_module : torch.nn.Sequential
    out_size : tuple of int
    out_channels : int
    """

    # Just to make sure that the operations in this function do not alter parameters outside it
    kwargs = copy.deepcopy(kwargs)

    ####################################
    # Parameter checks
    ####################################

    # Check and store input size
    assert isinstance(in_size, (tuple, list)) and len(in_size) == 2 and \
           all(isinstance(x, int) and x > 0 for x in in_size), \
        f"Invalid 'in_size': tuple/list of 2 integers > 0 expected, {in_size} found!."

    # Check and store number of input channels
    assert isinstance(in_channels, int) and in_channels > 0, \
        f"Invalid 'in_channels': integer > 0 expected, {in_channels} found!."

    # Check the flag indicating whether the convolutional-like layer is used everywhere
    conv_like_type_position_allowable_values = ['everywhere', 'first', 'last']
    assert isinstance(conv_like_type_position, str), \
        f"Invalid 'conv_like_type_position': str expected, {conv_like_type_position} found!."
    assert conv_like_type_position in conv_like_type_position_allowable_values, \
        f"Invalid 'conv_like_type_position': {conv_like_type_position} not found in the list of allowable " + \
        f"values, that is, {conv_like_type_position_allowable_values}."

    # Check and adapt the input values
    num_conv_layers = None
    num_conv_blocks = None
    assert isinstance(conv_block_specification, (list, tuple)), \
        f"Invalid 'conv_block_specification': list or tuple expected, {type(conv_block_specification)} found!."
    assert all([isinstance(elem, int) and elem > 0 for elem in conv_block_specification]), \
        f"Invalid 'conv_block_specification': list of integers > 0 expected, {conv_block_specification} found!."
    # This gives the number of total convolutional-like layers in the net
    num_conv_layers = int(np.array(conv_block_specification).sum())
    num_conv_blocks = len(conv_block_specification)

    # If 'batch_normalization_per_conv_layer' \
    # and 'maxpool_reduction_per_conv_block' are defaults or 1D elements,
    # convert them into lists of the appropriate length
    if isinstance(batch_normalization_per_conv_layer, bool):
        batch_normalization_per_conv_layer = [batch_normalization_per_conv_layer] * num_conv_layers
    if isinstance(maxpool_reduction_per_conv_block, int):
        maxpool_reduction_per_conv_block = [maxpool_reduction_per_conv_block] * num_conv_blocks
    # For 'kernel_maxpool_reduction_per_conv_block' check if it must be copied... or to the same
    if kernel_maxpool_reduction_per_conv_block is None:
        kernel_maxpool_reduction_per_conv_block = maxpool_reduction_per_conv_block
    elif isinstance(kernel_maxpool_reduction_per_conv_block, int):
        kernel_maxpool_reduction_per_conv_block = [kernel_maxpool_reduction_per_conv_block] * num_conv_blocks
    pass
    if isinstance(dilation_maxpool_reduction_per_conv_block, int):
        dilation_maxpool_reduction_per_conv_block = [dilation_maxpool_reduction_per_conv_block] * num_conv_blocks

    # Checks!
    _tmp_dict = {'channels_per_conv_layer': channels_per_conv_layer,
                 'phi_activation_per_conv_layer': phi_activation_per_conv_layer,
                 'm_kernel_size_per_conv_layer': m_kernel_size_per_conv_layer,
                 'batch_normalization_per_conv_layer': batch_normalization_per_conv_layer,
                 'maxpool_reduction_per_conv_block': maxpool_reduction_per_conv_block,
                 'kernel_maxpool_reduction_per_conv_block': kernel_maxpool_reduction_per_conv_block,
                 'dilation_maxpool_reduction_per_conv_block': dilation_maxpool_reduction_per_conv_block,
                 }

    for key, value in _tmp_dict.items():
        assert isinstance(value, (list, tuple)), \
            f"Invalid '{key}': list or tuple expected, {type(value)} found!."
        if key in ['maxpool_reduction_per_conv_block', 'kernel_maxpool_reduction_per_conv_block',
                   'dilation_maxpool_reduction_per_conv_block']:
            # Length must fit with the number of blocks
            assert len(value) == num_conv_blocks, \
                (f"Invalid '{key}': list or tuple of length {num_conv_blocks} expected, due to " +
                 f"'conv_block_specification'={conv_block_specification}" +
                 f" {value} found!.")
            assert all([isinstance(elem, int) and elem > 0 for elem in value]), \
                f"Invalid '{key}': list of integers > 0 expected, {value} found!."
        else:
            # Common for all the other keys: length must fit
            assert len(value) == num_conv_layers, \
                (f"Invalid '{key}': list or tuple of length {num_conv_layers} expected, due to " +
                 f"'conv_block_specification'={conv_block_specification}" +
                 f" {value} found!.")
            # Now, the specific types of each key
            if key in ['channels_per_conv_layer']:
                assert all([isinstance(elem, int) and elem > 0 for elem in value]), \
                    f"Invalid '{key}': list of integers > 0 expected, {value} found!."
            elif key == 'm_kernel_size_per_conv_layer':
                pass
                # Further checks later on 'm_kernel_size_per_conv_layer', once it has been stored
            elif key == 'batch_normalization_per_conv_layer':
                assert all([isinstance(elem, bool) for elem in value]), \
                    f"Invalid '{key}': list of booleans expected, {value} found!."
            elif key == 'phi_activation_per_conv_layer':
                assert all([isinstance(elem, str) for elem in value]), \
                    f"Invalid '{key}': list of strings expected, {value} found!."
            pass
        pass
    pass

    # Process 'initial_lambda', if present, specially: if it is a vector it must have as many elements as \
    # conv. layers in the net; if it is a single value, it is expanded to a vector with as many elements as \
    # conv. layers in the net, all with the same value.
    if conv_like_type in ['inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
        if not 'initial_lambda' in kwargs:
            raise Exception(f"Invalid 'initial_lambda': it must be explicitly set to a valid value; " +
                            f"no value, or None, provided.")
        else:
            initial_lambda = kwargs['initial_lambda']
            # We make the 'initial_lambda' always be a vector with 1 lambda per conv-like layer
            # (even if all lambdas are the same)
            new_initial_lambda = None
            if conv_like_type_position in ['everywhere', 'all']:
                if isinstance(initial_lambda, (list, tuple)):
                    assert len(initial_lambda) == num_conv_layers and all(
                        [isinstance(elem, (int, float)) for elem in initial_lambda]), \
                        (f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers}, " +
                         f"as many as conv-like layers, expected, " +
                         f"{initial_lambda} found!.")
                    new_initial_lambda = tuple([float(elem) for elem in initial_lambda])
                elif isinstance(initial_lambda, (int, float)):
                    new_initial_lambda = tuple([float(initial_lambda)] * num_conv_layers)
                else:
                    raise TypeError(f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers}, " +
                                    f"as many as conv-like layers, or scalar expected, " +
                                    f"{initial_lambda} found!")
                pass
            elif conv_like_type_position in ['first', 'last'] or num_conv_layers == 1:
                if isinstance(initial_lambda, (int, float)):  # We make it a list of length 1
                    new_initial_lambda = [0.0] * num_conv_layers
                    if conv_like_type_position == 'first':
                        new_initial_lambda[0] = float(initial_lambda)
                    else:
                        new_initial_lambda[-1] = float(initial_lambda)
                    pass
                elif isinstance(initial_lambda, (list, tuple)):
                    assert len(initial_lambda) == num_conv_layers, \
                        (f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers}, " +
                         f"as many as conv-like layers, expected, " +
                         f"{initial_lambda} found!.")
                    assert all([isinstance(elem, (int, float)) for elem in initial_lambda]), \
                        (f"Invalid 'initial_lambda': list/tuple of numbers expected, " +
                         f"{initial_lambda} found!")
                    # Check that only the first or last element is non-zero
                    if conv_like_type_position == 'first':
                        assert all([float(elem) == 0.0 for elem in initial_lambda[1:]]), \
                            (f"Invalid 'initial_lambda': only the first element can be non-zero " +
                             f"if 'conv_like_type_position' is 'first', " +
                             f"{initial_lambda} found!")
                    else:
                        assert all([float(elem) == 0.0 for elem in initial_lambda[:-1]]), \
                            (f"Invalid 'initial_lambda': only the last element can be non-zero " +
                             f"if 'conv_like_type_position' is 'last', " +
                             f"{initial_lambda} found!")
                    pass
                    new_initial_lambda = [float(elem) for elem in initial_lambda]
                else:
                    raise TypeError(f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers}, " +
                                    f"as many as conv-like layers, or scalar expected, " +
                                    f"{initial_lambda} found!")
                # If it is a list of 1 element, expand to a list spaning the total number of conv-like layers
                if isinstance(initial_lambda, (list, tuple)) and len(initial_lambda) == 1:
                    initial_lambda = [float(initial_lambda[0])] * num_conv_layers
            else:
                raise Exception(f"Invalid 'conv_like_type_position': {conv_like_type_position} not recognized!.")
            pass
            #
            # USE THE NEW initial_lambda
            kwargs['initial_lambda'] = new_initial_lambda
        pass
    pass

    ######################################################
    # LAYER PARAMETER SEPARATION:
    # The will generate the required for 'sm' and for the selected conv-like layer
    # to ease their use for populating the backbone of the network
    ######################################################

    # Fields in the constructor of the layers to leave out: the fields that will be filled individually per layer
    fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'phi_activation', 'm_kernel_size', 'm_groups', 'w_groups']

    # Dictionary of explicit args in the network that we want to force
    dict_explicit_args_sm = {'m_padding': 'same'}
    dict_explicit_args_conv_like_layer = copy.deepcopy(dict_explicit_args_sm)
    if conv_like_type in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
        dict_explicit_args_conv_like_layer['w_kernel_size'] = w_kernel_size
    pass

    # Dictionary, but for the fields in the list 'fields_leave_out', for the SMLayers
    sm_kwargs = processed_constructor_kwargs_for_conv_like_layer(
        _dict_conv_like_layers['sm'],
        fields_leave_out=fields_leave_out,
        dict_explicit_args=dict_explicit_args_sm, dict_non_explicit_args=kwargs,
        flag_add_all_other_args=False
    )

    # Dictionary, but for the fields in the list 'fields_leave_out', for the selected conv-like layer
    conv_like_layer_kwargs = processed_constructor_kwargs_for_conv_like_layer(
        _dict_conv_like_layers[conv_like_type],
        fields_leave_out=fields_leave_out,
        dict_explicit_args=dict_explicit_args_conv_like_layer, dict_non_explicit_args=kwargs,
        flag_add_all_other_args=True
    )

    #############################################################################################
    # Creation of the module, by creating its blocks
    #############################################################################################

    ordered_dict_backbone_module = OrderedDict()

    # Note: in "channels_per_conv_layer" and "m_kernel_size_per_conv_layer" values are given for absolute layer numbers!
    abs_ind_conv_layer = 0
    total_num_layers = sum(conv_block_specification)

    out_channels_previous_layer = in_channels
    out_size_previous_layer = in_size

    for ind_block, conv_block_i_specification in enumerate(conv_block_specification):
        #
        ordered_dict_block_i = OrderedDict()
        #
        ##################################################
        # Create and add the conv-like layers in the block
        ##################################################
        #
        for rel_ind_conv_layer in range(conv_block_i_specification):
            #
            # Type of the conv-like layer at block `ind_block`, layer `rel_ind_conv_layer`,
            # and selection of tis corresponding kwargs
            conv_like_type_i = 'sm'
            if (abs_ind_conv_layer == 0 and conv_like_type_position == 'first') or \
                    (abs_ind_conv_layer == total_num_layers-1 and conv_like_type_position == 'last') or \
                    conv_like_type_position == 'everywhere':
                # No 'sm':
                if 'initial_lambda' in kwargs and kwargs['initial_lambda'][abs_ind_conv_layer] != 0.0:
                    conv_like_type_i = conv_like_type
            pass

            # Sizes of layer at block `ind_block`, layer `rel_ind_conv_layer`
            in_size_layer_i = out_size_previous_layer
            in_channels_layer_i = out_channels_previous_layer


            # Get the kwargs of the corresponding conv-like layer and complete them using the size and channel info
            kwargs_layer_i = copy.deepcopy(dict_explicit_args_sm) if conv_like_type_i == 'sm' \
                else copy.deepcopy(conv_like_layer_kwargs)
            kwargs_layer_i['in_size'] = in_size_layer_i
            kwargs_layer_i['in_channels'] = in_channels_layer_i
            kwargs_layer_i['out_channels'] = channels_per_conv_layer[abs_ind_conv_layer]
            kwargs_layer_i['m_kernel_size'] = \
                resolve_kernel_size_for_im_size(
                    kernel_size=m_kernel_size_per_conv_layer[abs_ind_conv_layer],
                    im_size=in_size_layer_i,
                    make_odd=True
                )
            kwargs_layer_i['phi_activation'] = phi_activation_per_conv_layer[abs_ind_conv_layer]
            kwargs_layer_i['m_groups'] = kwargs_layer_i['in_channels'] if m_independent_channels else 1
            if conv_like_type_i in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
                kwargs_layer_i['w_groups'] = kwargs_layer_i['out_channels'] if w_independent_channels else 1
                kwargs_layer_i['initial_lambda'] = kwargs['initial_lambda'][abs_ind_conv_layer]
            pass

            # Create the layer
            layer_conv_i = _dict_conv_like_layers[conv_like_type_i](**kwargs_layer_i)
            
            # Name the layer and add it to the ordered dict
            name_conv_i = f"conv_{ind_block}_{rel_ind_conv_layer}"
            ordered_dict_block_i.update({name_conv_i: layer_conv_i})
            
            # Update the output size and channels
            out_channels_previous_layer = kwargs_layer_i['out_channels']
            out_size_previous_layer = kwargs_layer_i['in_size'] # Because of the padding='same'
            #
            ##################################################
            # Create and add the batch normalization layer, if requested
            ##################################################
            #
            name_norm_conv_i = f"norm_{ind_block}_{rel_ind_conv_layer}"
            layer_norm_conv_i = nn.BatchNorm2d(num_features=out_channels_previous_layer,
                                               affine=True, track_running_stats=True) \
                if batch_normalization_per_conv_layer[abs_ind_conv_layer] \
                else nn.Identity()
            ordered_dict_block_i.update({name_norm_conv_i: layer_norm_conv_i})
            #
            # Advance the absolute conv layer index and update the output size and channels
            abs_ind_conv_layer += 1
        pass
        #
        ##################################################
        # Create and add maxpooling layer, if requested
        ##################################################
        #
        name_maxpool_conv_i = f"maxpool_{ind_block}"
        stride_maxpool_reduction_for_block_i = maxpool_reduction_per_conv_block[ind_block]
        kernel_maxpool_reduction_for_block_i = kernel_maxpool_reduction_per_conv_block[ind_block]
        dilation_maxpool_reduction_for_block_i = dilation_maxpool_reduction_per_conv_block[ind_block]
        layer_maxpool_conv_i = \
            nn.MaxPool2d(
                kernel_size=kernel_maxpool_reduction_for_block_i,
                stride=stride_maxpool_reduction_for_block_i,
                dilation=dilation_maxpool_reduction_for_block_i)\
            if (stride_maxpool_reduction_for_block_i is not None) and (
                    (stride_maxpool_reduction_for_block_i > 1) or (dilation_maxpool_reduction_for_block_i > 1) or \
                    (kernel_maxpool_reduction_for_block_i > 1)
            ) else nn.Identity()
        ordered_dict_block_i.update({name_maxpool_conv_i: layer_maxpool_conv_i})

        # Update the output size and channels
        out_channels_previous_layer = out_channels_previous_layer
        d = dilation_maxpool_reduction_for_block_i
        k = kernel_maxpool_reduction_for_block_i
        s = stride_maxpool_reduction_for_block_i
        out_size_previous_layer = tuple([int( (elem - d*(k-1) - 1)/s + 1 ) for elem in out_size_previous_layer])

        ##################################################
        # Pack and add the block
        ##################################################
        #
        name_block_conv_i = f"block_conv_{ind_block}"
        module_block_conv_i = nn.Sequential(ordered_dict_block_i)
        ordered_dict_backbone_module.update({name_block_conv_i: module_block_conv_i})
    pass

    #############################################################################################
    # Pack all the blocks into a single module
    #############################################################################################

    backbone_module = nn.Sequential(ordered_dict_backbone_module)
    out_size = out_size_previous_layer
    out_channels = out_channels_previous_layer

    return backbone_module, out_size, out_channels
