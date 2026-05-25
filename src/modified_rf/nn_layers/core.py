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
import copy

import torch
from torch import nn
import torch.nn.functional as F

import modified_rf.nn_layers as nnl
from modified_rf.fixed_point import FixedPointLayer

from modified_rf.nn_layers.utils import f_modified_RF, cast_scalar_like_to_image

import gc
import modified_rf.memory_handling as memo




def processed_constructor_kwargs_for_conv_like_layer(
    ModifiedRFLayerClass: type,
    fields_leave_out=None, dict_explicit_args=None, dict_non_explicit_args=None, flag_add_all_other_args=False
):
    """
    This function fills a dictionary of arguments required/accepted for the constructor of the convolution-like layer \
    type `ModifiedRFLayerClass` (subclass of :py:class:`.ModifiedRFLayer`) \
    using the entries in the provided dictionary `dict_explicit_args` and `dict_non_explicit_args` \
    taking into account the following aspects: *(1)* the list of explicit parameters that the layer accepts, \
    among which some compulsory (with no default values) and some others optional and with default value; \
    *(2)* a set of unkwnown acceptable *kwargs* for the layer; and (3) a list of `fields_leave_out` which have been \
    explicitly requested out from the resulting dictionary.list of fit the potential parameters accepted by the layer. \
    The fields/keys in `fields_leave_out` \
    are not included at all in the resulting dictionary of arguments.
    Regarding the separation into `dict_explicit_args` and `dict_non_explicit_args`: \
    the fields of the latter correspond to \
    the unkwnown acceptable *kwargs* of the layer, which might or might not be present; they appear separated from \
    `dict_explicit_args` so as to allow the inclusion of all fields in `dict_non_explicit_args`, identified or not, \
    depending on the value of `flag_add_all_other_args` (if ``True``, all fields in `dict_non_explicit_args` are addded; \
    if ``False``, only those fields that are explicitly listed in the constructor of the convolution-like layer \
    type `ModifiedRFLayerClass` are added).

    Parameters
    ----------
    ModifiedRFLayerClass : type, class inheriting from :py:class:`.ModifiedRFLayer`
    fields_leave_out : list[str] or tuple[str], optional
        List of fields that we do not want to include in the dictionary of arguments. \
        Default: ``None``
    dict_explicit_args : dict, optional
        Dictionary with the arguments that we want to push into the constructor of the layer.
        Default: ``None``
    dict_desired_args : dict, optional
        Dictionary with the arguments that we want to push into the constructor of the layer.
        Default: ``None``
    flag_add_all_other_args : bool, optional
        Default: ``False``

    Returns
    -------
    dict
        Dictionary (*kwargs*) with the arguments that would generate a functional object of the layer type \
        `modifiedRFLayerClass`.
    """

    ########################################################
    # Initial checks
    ########################################################

    assert issubclass(ModifiedRFLayerClass, nnl.ModifiedRFLayer), \
        f"The class '{ModifiedRFLayerClass.__name__}' is not a subclass of 'ModifiedRFLayer'!"

    if fields_leave_out is None:
        fields_leave_out = []
    elif isinstance(fields_leave_out, str):
        fields_leave_out = [fields_leave_out]
    elif isinstance(fields_leave_out, tuple):
        fields_leave_out = list(fields_leave_out)
    pass

    assert isinstance(fields_leave_out, list) and all([isinstance(f, str) for f in fields_leave_out]), \
        f"The argument 'fields_leave_out' must be a list of strings, but it is {type(fields_leave_out)}!"

    if dict_explicit_args is None:
        dict_explicit_args = {}
    pass
    assert isinstance(dict_explicit_args, dict), \
        f"The argument 'dict_explicit_args' must be a dictionary, but it is {type(dict_explicit_args)}!"

    if dict_non_explicit_args is None:
        dict_non_explicit_args = {}
    pass
    assert isinstance(dict_non_explicit_args, dict), \
        f"The argument 'dict_non_explicit_args' must be a dictionary, but it is {type(dict_non_explicit_args)}!"

    assert isinstance(flag_add_all_other_args, bool), \
        f"The argument 'flag_add_all_other_args' must be a boolean, but it is {type(flag_add_all_other_args)}!"

    ########################################################
    # Create the dictionary of arguments
    ########################################################

    # Initialize the dictionary of arguments... to be filled
    conv_like_layer_final_args = {}

    # Potential arguments of the layer, which might present
    conv_like_layer_potential_args = ModifiedRFLayerClass.constructor_default_values(only_not_none=False)
    # Arguments of the layer that have a default
    conv_like_layer_default_args = ModifiedRFLayerClass.constructor_default_values(only_not_none=True)

    # We "hard"copy the dictionary "dict_non_explicit_args" for convenience: so we can use pop() without modifying \
    # the original dictionary to know what we have already used and what we have not
    dict_non_explicit_args = copy.deepcopy(dict_non_explicit_args)

    # Hierarchically fill the potential arguments of the Layer: "forced", present here in kwargs, or default
    for key in conv_like_layer_potential_args:
        if key in fields_leave_out:
            # Ignore
            pass
        elif key in dict_explicit_args:
            conv_like_layer_final_args[key] = copy.deepcopy(dict_explicit_args[key])
        elif key in dict_non_explicit_args:
            conv_like_layer_final_args[key] = dict_non_explicit_args.pop(key, None)  # This time we remove them from 'kwargs'
        elif key in conv_like_layer_default_args:
            conv_like_layer_final_args[key] = copy.deepcopy(conv_like_layer_default_args[key])
        pass
    pass

    # If 'flag_add_all_other_args' is True and there are elements in 'dict_non_explicit_args' left, we include them
    if flag_add_all_other_args:
        for key in dict_non_explicit_args:
            conv_like_layer_final_args[key] = dict_non_explicit_args[key]
        pass
    pass

    # Fix some incompatibilities that we might have observed in the parameters
    m_padding = conv_like_layer_final_args.get('m_padding', None)
    if m_padding=='fc':
        for key in ['m_padding_mode']:
            if key in conv_like_layer_final_args:
                conv_like_layer_final_args[key] = None
    elif m_padding == 'valid':
        for key in ['m_padding_mode']:
            if key in conv_like_layer_final_args:
                conv_like_layer_final_args[key] = None
    pass

    return conv_like_layer_final_args


######################################################


def m_and_b_dimensioning(in_channels, out_channels,
                         m_kernel_size, m_padding='same', m_groups=1,
                         b_type='scalar_per_channel', in_size=None):
    """
    This function creates the "empty" (in the sense that the values therein are unimportant; the size of the \
    created structures is) tensors `m` and `b` for the given parameters.

    This function, previously part of the constructor of each :py:class:`ModifiedRFLayerClass` child class, has been \
    "externalized" since its logic has "grown" since the consideration of the `m_padding` options \
    ``'same'``/``'valid'``, on the one hand, and ``'fc'``, on the other: since both cases are addressed quite \
    differently this process has been separated into a function with such purpose.

    The main difference between the pseudo-convolutional layers ``'same'``/``'valid'`` and the fully-connected \
    layer ``'fc'`` is the following: for the former, `m_kernel_size` represents, precisely, the spatial size of the \
    desired filter mask (which, along with `in_channels`, `out_channels`, and the knowledge of the padding type, \
    would help deduce the output size of the function; however, for ``'fc'``, the provided `m_kernel_size` \
    represents actually the intended spatial extent of the output of the layer, and then it can be equated to an \
    equivalent argument `out_size`, which, together with the (in this case) required argument `in_size` \
    allows to deduce the size to the matrices of the linear transform if necessary.

    Parameters
    ----------
    in_channels, out_channels : int
    m_kernel_size : int or float or 2D list[int|float] or 2D tuple[int|float]
        **Important:** \
        When `m_padding` corresponds to ``'same'``/``'valid'``, then the layer is pseudo-convolutional \
        and the argument `m_kernel_size` corresponds to the **spatial extent of the filter mask**. \
        However, when `m_padding` corresponds to ``'fc'``, then the layer is full-connected \
        and the argument `m_kernel_size` corresponds to the **spatial extent of the resulting image**.
    m_padding : str, optional
        Value among ``‘same’``, `‘valid’``, and ``‘fc’``. The options ``‘same’`` and `‘valid’`` correspond to a \
        convolutional affine transform and correspond to the usual definitions for padding used e.g. in the \
        function :py:class:`torch.nn.Conv2d`: see :py:func:`.conv2d_adapted` for a detailed explanation. \
        However, the option ``‘fc’`` corresponds to a fully-convolutional \
        affine transform, which renders the rest of options related to $\\mathbf{M}$ and $\\mathbf{b}$ meaningless: \
        in particular the arguments ``m_kernel_size``, ``m_padding_mode``, ``m_groups``, and ``b_type`` are simply \
        not used in such case (and set to defaults for the ``‘fc’`` case).
        Default: ``‘same’``
    m_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    b_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``, or ``None``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). \
        In the special case where the affine transformation given by $\\mathbf{M}$ and $\\mathbf{b}$ is \
        fully connected its value is forced to ``None``, meaning that every output pixel has a different bias.
        Default: None for `m_padding` ``'fc'``, ``'scalar_per_channel'`` otherwise
    in_size : int or list[int] or tuple[int]
        Integer, interpreted corresponding to data in the form of a column vector, of 2D tuples \
        (2D at least; trailing dimensions do not matter for the involved processing regarding only \
        spatial extent) indicating the size of the input image/data to the layer. This argument is not necessary if \
        `m_kernel_size` indicates a convolutional transformation and \
        the kernel sizes for *m* and *w* are provided as absolute int values; however it becomes necessary if \
        any of them is provided as relative (e.g. floats $0 \\leq x \\leq 1$) to the input size, and \
        in the case of ``m_padding`` ``'fc'``.
        Default: ``None``

    Returns
    -------
    m : torch.Tensor
        Initial `m`
    b : torch.Tensor
        Initial `b`
    """

    #####
    # Initial checks
    #####

    assert isinstance(in_channels, int) and in_channels >= 1, \
        f"'in_channels' must be an integer >= 1 but got {in_channels}."
    assert isinstance(out_channels, int) and out_channels >= 1, \
        f"'out_channels' must be an integer >= 1 but got {out_channels}."

    # 'm_kernel_size' will be checked for each condition separately, below

    assert isinstance(m_padding, str), \
        f"'m_padding' must be a string: {type(m_padding)} found."
    if not m_padding in ['fc', 'valid', 'same']:
        raise Exception(f"'m_padding' must be one of: 'same', 'valid', and 'fc': {m_padding_mode} found.")
    pass

    assert isinstance(m_groups, int), f"'m_groups' must be an integer but got {type(m_groups)}."
    assert m_groups >= 1, f"'m_groups' must be >= 1 but got {m_groups}."
    # Check divisibility of both 'in_channels' and 'out_channels' by 'm_groups'
    if in_channels % m_groups != 0:
        raise Exception(f"'in_channels' {in_channels} must be divisible by 'm_groups' {m_groups} but is not.")
    if out_channels % m_groups != 0:
        raise Exception(f"'out_channels' {out_channels} must be divisible by 'm_groups' {m_groups} but is not.")

    if m_padding == 'fc':
        if b_type is not None: # Issue a warning and set it to None
            warnings.warn(f"'b_type' different from None ({b_type}) for 'm_padding'='fc': None forced.")
            b_type = None
    else:
        assert b_type in ['scalar', 'scalar_per_channel'], \
            f"'b_type' must be one of ['scalar', 'scalar_per_channel'] " + \
            f"when 'm_padding'='{m_padding}', but got '{b_type}'."
    pass

    c_in_per_group = in_channels // m_groups
    c_out = out_channels

    #####
    # Create the structures 'm' and 'b' differently for the
    #####

    if m_padding == 'fc':
        ############################################################################################################
        # For 'fc', fully-connected
        ############################################################################################################
        # In this case: 'in_size' encodes the spatial size of the input of the image,
        #               'm_kernel_size' encodes the spatial size of the output image of the layer
        ############################################################################################################
        # We use, for convenience, the name 'out_size' inside this function
        out_size = copy.deepcopy(m_kernel_size)

        assert in_size is not None and out_size is not None, \
            f"Both 'in_size' and 'out_size' are required when 'm_padding'='fc'."
        # Check and process the input and output sizes
        original_dict_in_out_size = {'in': copy.deepcopy(in_size),
                                     'out': copy.deepcopy(out_size)}
        dict_in_out_size = {'in': copy.deepcopy(in_size),
                            'out': copy.deepcopy(out_size)}
        for key in dict_in_out_size:
            assert dict_in_out_size[key] is not None, \
                f"'{key}_size' must be provided when 'm_padding'='fc': however, None has been provided."
            if isinstance(dict_in_out_size[key], int): # If only one int, it will be understood as a column vector of that size
                dict_in_out_size[key] = (dict_in_out_size[key], 1)
            if isinstance(dict_in_out_size[key], list):
                dict_in_out_size[key] = tuple(dict_in_out_size[key])
            assert isinstance(dict_in_out_size[key], tuple) and len(dict_in_out_size[key])==2, \
                f"'{key}_size' must be a 2D tuple (or castable) but but got {original_dict_in_out_size[key]} instead."
            assert all([isinstance(elem, int) for elem in dict_in_out_size[key]]), \
                f"'{key}_size' must be composed, for 'm_padding'='fc', of integers; got {original_dict_in_out_size[key]} instead."
            assert all([elem>0 for elem in dict_in_out_size[key]]), \
                f"'{key}_size' must be composed, for 'm_padding'='fc', of integers > 0; got {original_dict_in_out_size[key]} instead."
        pass
        #
        H_in, W_in = dict_in_out_size['in'][-2], dict_in_out_size['in'][-1]
        H_out, W_out = dict_in_out_size['out'][-2], dict_in_out_size['out'][-1]
        #
        # Define 'm' and 'b' according to these values
        m = torch.empty((c_out, c_in_per_group, H_out*W_out, H_in*W_in))
        b = torch.empty((c_out, H_out, W_out))
    else:
        ############################################################################################################
        # For 'valid'/'same', convolutional
        ############################################################################################################
        # In this case: 'in_size' is not necessary, and it is not used,
        #               'm_kernel_size' encodes the spatial extent of the image
        ############################################################################################################
        # Reformat to help checks (not resolving yet for relative filters)
        m_kernel_size = nnl.kernel_size_check_and_reformat_into_tuple(m_kernel_size)
        assert len(m_kernel_size) == 2, \
            f"'m_kernel_size' must be of length 2, {m_kernel_size} found."
        # And resolve the (spatial) kernel size if necessary
        m_kernel_size = nnl.resolve_kernel_size_for_im_size(m_kernel_size, im_size=in_size, make_odd=False)
        #
        # Define 'm' and 'b' according to these values
        m = torch.empty((c_out, c_in_per_group, m_kernel_size[-2], m_kernel_size[-1]))
        b = torch.empty((c_out, 1, 1)) if b_type=='scalar_per_channel' else torch.empty((1, 1, 1))
    pass

    return m, b


######################################################
######################################################
# MODELS/LAYERS
######################################################
######################################################

######################################################
# Abstract Layer ModifiedRFLayer
######################################################

class ModifiedRFLayer(ABC):
    """
    This class aims at unifying all the *getter* and *setter* methods of all the INRF versions and ibnn_internal, \
    respecting the requirements of Autograd for gradient/not-gradient. Said getters and setters are mostly \
    based on the assumption that there is an attribute dictionary 'self._theta' with the fields 'm', 'b', 'lambda', 'w'.

    The 'forward' will have to be rewritten by each child.
    """

    def __init__(self, **kwargs):
        # super().__init__()
        # print(f"---> Constructor of ModifiedRFLayer!")

        ### No more **kwargs arguments than expected
        list_allowable_kwargs = ['calculation_mode', 'memory_saving_version',
                                 'nonuniform_sampling', 'num_sampling_points',
                                 'start_range_std_activation', 'end_range_std_activation', 'interpolation_type',
                                 'extrapolation_type']

        # Empty dictionary for the kwargs in the above list, no more
        accepted_kwargs = {}
        for key in kwargs.keys():
            if key not in list_allowable_kwargs:
                # Warning instead of Exception
                warnings.warn(f"The keyword argument '{key}' is not in the list of allowable arguments!")
            else:
                accepted_kwargs[key] = kwargs[key]
            pass
        pass

        # Remove 'calculation_mode' and 'memory_saving_version' for a special treatment
        self._calculation_mode = None
        if 'calculation_mode' in accepted_kwargs:
            self._calculation_mode = accepted_kwargs['calculation_mode']
            del accepted_kwargs['calculation_mode']
        pass
        self._memory_saving_version = None
        if 'memory_saving_version' in accepted_kwargs:
            self._memory_saving_version = accepted_kwargs['memory_saving_version']
            del accepted_kwargs['memory_saving_version']
        pass

        # Rest of "fixed" kwargs, as attribute
        self._fixed_kwargs = accepted_kwargs

    pass

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        This function, **meant to be called before starting the training, independently of** \
        **the initialized values, to avoid zero gradients for some trainable parameters** \
        **depending on their starting value**, introduces random variations to the initialized \
        values.

        This function introduces randomness to the components of $\\Theta$ set as trainable, and only \
        to those: the rest of components stay intact. Regarding the component to which randomness \
        is introduced, the default mode of the function (``additive``=``True``) adds randomness \
        to the current value of the corresponding component; complete randomness around zero values \
        is also supported  (``additive``=``False``).

        Parameters
        ----------
        distribution : str or dict[str], optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            If the provided value is a string the same standard distribution is applied to all trainable fields \
            of $\\Theta$; if instead ``gain`` is a dictionary each key must contain the distribution \
            of the corresponding key of $\\Theta$. \
            Default: ``'normal'``
        gain : int or float or dict[int or float], optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            If the provided value is a scalar the same gain is applied to all trainable fields of $\\Theta$; \
            if instead ``gain`` is a dictionary each key must contain the scalar gain of the corresponding \
            key of $\\Theta$. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        # Flags "same distribution/gain for all keys or not?"
        common_distribution = isinstance(distribution, str)
        common_gain = isinstance(gain, float) or isinstance(gain, int)

        # Dictionary with the standard random generators to use (since the uniform is a bit modified wrt. torch.rand)
        dict_standard_random_generator = {}
        dict_standard_random_generator['uniform'] = \
            lambda s, device=None: torch.rand(size=s, device=device) - 0.5
        dict_standard_random_generator['normal'] = \
            lambda s, device=None: torch.randn(size=s, device=device)

        for key in self._theta:
            if self._theta_trainable[key]:
                #
                normalized_random_generator_key = dict_standard_random_generator[distribution] if common_distribution \
                    else dict_standard_random_generator[distribution[key]]
                gain_key = gain if common_gain else gain[key]
                #
                noise_component = gain_key * normalized_random_generator_key(self._theta[key].size(),
                                                                             device=self._theta[key].device)
                #
                self._theta[key].requires_grad = False
                self._theta[key][:] = self._theta[key][:] + noise_component[:] if additive else noise_component[:]
                self._theta[key].requires_grad = True
            pass
        pass
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def dict_fields_to_log_at_current_time(self):
        """
        The function returns a dictionary comprising the following fields (depending on the type of model, which \
        is indicated below):

        - For all the layer types:
            - `'layer_type'`: e.g. 'SMLayer', 'INRFv1Layer', ..., 'IBNNLiteLayer', 'IBNNInternalLayer', 'IBNNLayer'
            - `'phi_activation'`,
            - `'m_kernel_size'`, `'m_groups'`, `'m_padding'`, `'m_padding_mode'`, `'m_initialization'`, `'m_trainable'`
            - `'b_type'`, `'initial_b'`, `'b_trainable'`
        - Only for the layers of the INRF family and ibnn_internal:
            - `'sigma_activation'`
            - `'sigma_x_compress'`, `'sigma_y_stretch'`, `'sigma_x_offset'`, `'sigma_y_offset'`
            - `'sigma_x_compress_trainable'`, `'sigma_y_stretch_trainable'`, `'sigma_x_offset_trainable'`, `'sigma_y_offset_trainable'`
            - `'w_kernel_size'`, `'w_groups'`, `'w_padding'`, `'w_padding_mode'`, `'w_initialization'`, `'w_trainable'`
            - `'lambda_type'`, `'initial_lambda'`, `'lambda_trainable'`
        - Only for ibnn_internal:
            - `'batched_fixed_point'`
            - `'f_solver'`, `'b_solver'`

        Returns
        -------
        dict
        """
        #
        dict_fields_to_log = {}
        #
        ##########
        # PARAMETERS FOR EVERY TYPE OF LAYER
        ##########
        #
        dict_fields_to_log['layer_type'] = type(self).__name__
        #
        dict_fields_to_log['phi_activation'] = self._phi_activation
        #
        dict_fields_to_log['m_kernel_size'] = tuple(self.theta_copy['m'].size()[-2:])
        # dict_fields_to_log['m_kernel_height'] = self.theta_copy['m'].size(-2)
        # dict_fields_to_log['m_kernel_width'] = self.theta_copy['m'].size(-1)
        dict_fields_to_log['m_groups'] = self.m_groups
        dict_fields_to_log['m_padding'] = self.m_padding
        dict_fields_to_log['m_padding_mode'] = self.m_padding_mode
        dict_fields_to_log['m_trainable'] = self.get_trainable('m')
        dict_fields_to_log['m_initialization'] = self._m_initialization
        #
        dict_fields_to_log['b_type'] = \
            'scalar' if (self.theta_copy['b'].numel() == 1) else 'scalar_per_channel'
        dict_fields_to_log['initial_b'] = self.theta0['b'].item() \
            if (self.theta0['b'].numel() == 1) else tuple(self.theta0['b'].flatten().tolist())
        dict_fields_to_log['b_trainable'] = self.get_trainable('b')
        #
        ##########
        # PARAMETERS FOR INRFs and ibnn_internal
        ##########
        #
        if isinstance(self, (INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer)):
            #
            dict_fields_to_log['sigma_activation'] = self._sigma_activation
            #
            list_sigma_modifier_keys = ['sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset']
            for key in list_sigma_modifier_keys:
                dict_fields_to_log[key] = self.theta_copy[key].item() if self.theta_copy[key].numel() == 1 \
                    else tuple(self.theta_copy[key].tolist())
                dict_fields_to_log[key + '_trainable'] = self.get_trainable(key)
            pass
            #
            dict_fields_to_log['lambda_type'] = \
                'scalar' if (self.theta_copy['lambda'].numel() == 1) else 'scalar_per_channel'
            dict_fields_to_log['initial_lambda'] = self.theta0['lambda'].item() \
                if (self.theta0['lambda'].numel() == 1) else tuple(self.theta0['lambda'].flatten().tolist())
            dict_fields_to_log['lambda_trainable'] = self.get_trainable('lambda')
            #
            if isinstance(self, (INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer)):
                dict_fields_to_log['w_kernel_size'] = tuple(self.theta_copy['w'].size()[-2:])
                # dict_fields_to_log['w_kernel_height'] = self.theta_copy['w'].size(-2)
                # dict_fields_to_log['w_kernel_width'] = self.theta_copy['w'].size(-1)
                dict_fields_to_log['w_groups'] = self.w_groups
                dict_fields_to_log['w_padding'] = self.w_padding
                dict_fields_to_log['w_padding_mode'] = self.w_padding_mode
                dict_fields_to_log['w_trainable'] = self.get_trainable('w')
                dict_fields_to_log['w_initialization'] = self._w_initialization
                #
                if isinstance(self, (IBNNInternalLayer, IBNNLayer)):
                    dict_fields_to_log['batched_fixed_point'] = self.batched_fixed_point
                    dict_fields_to_log['f_solver'] = self.f_solver
                    dict_fields_to_log['b_solver'] = self.b_solver
                pass
            #
        pass
        #
        return dict_fields_to_log

    def get_extra_state(self):
        """
        Store all the attributes of the network that are not part of the trainable \
        :py:class:`torch.nn.parameter.Parameter` of the network, since those will be stored \
        by :py:class:`torch.nn.Module` itself.

        The function returns, exactly, the dictionary returned by the function \
        :py:meth:`.dict_fields_to_log_at_current_time` of this class: refer to it for details about the fields.

        See also :py:meth:`.set_extra_state` for the counterpart of this method.

        Returns
        -------
        dict
        """

        return self.dict_fields_to_log_at_current_time()

    def set_extra_state(self, extra_state_dict):
        """
        Load into an existing object the state `extra_state_dict` of a previous object of the class stored according \
        to the format in :py:meth:`.get_extra_state` \
        (i.e. :py:meth:`.dict_fields_to_log_at_current_time`).

        In fact: not in all cases the fields in `extra_state_dict` will correspond to the object attempting to load \
        them; this would happen, for instance, when the layer types of both objects are not compatible, since \
        not all layer types accept the same parameters, or when the current object has been initialized with \
        filter sizes not compatible with those stored. This function:

        - provides a warning whenever there are parameters in `extra_state_dict` that would not be loaded because \
          they do not apply or whenever there are parameters of the current objects not to be overridden by \
          `extra_state_dict` because they did not exist in the past object; and

        - **Should it fail (Exception) when filter sizes between both are simply not coincident? It does not for now.**
        """

        ###############################################
        # Compare the 'layer_type' of the current object with that of the past object, and check if things are valid
        ###############################################

        # Order of layers, considering that the attributes of each layer are contained into the following layers
        ordered_list_layers = ['SMLayer', 'INRFv1Layer', 'INRFv2Layer', 'INRFv3Layer', 'IBNNLiteLayer', 'IBNNInternalLayer', 'IBNNLayer']

        current_layer_type = type(self).__name__
        past_layer_type = extra_state_dict['layer_type']

        index_current_layer_type = ordered_list_layers.index(current_layer_type)
        index_past_layer_type = ordered_list_layers.index(past_layer_type)

        # We take the current state dict for comparison with the input 'extra_state_dict'
        current_extra_state_dict = self.dict_fields_to_log_at_current_time()
        if index_current_layer_type < index_past_layer_type:
            print((f"Data from a layer of type '{past_layer_type}' to be loaded in a layer " +
                   f"of type '{current_layer_type}': a number of parameters will not be loaded. To wit:"))
            for key in extra_state_dict:
                if key not in list(current_extra_state_dict.keys()):
                    pass
                    print(f"| {key} ", end="")
                pass
            pass
            print(" ")
        pass

        ###############################################
        # Load the attributes!
        # This block is practically mirrored from :py:meth:`.dict_fields_to_log_at_current_time()`
        ###############################################

        if index_past_layer_type >= ordered_list_layers.index('SMLayer'):
            ##########
            # PARAMETERS FOR EVERY TYPE OF LAYER
            ##########
            #
            # 'layer_type'  Type: not copied, intrinsic of the current layer
            #
            if self.phi_activation != extra_state_dict['phi_activation']:
                print(f"Activation 'phi' changed to '{self.phi_activation}' from '{extra_state_dict['phi_activation']}'!")
            pass
            #self._phi_activation = extra_state_dict['phi_activation']
            #
            # 'm_kernel_size' calculated directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            #
            assert self._m_groups == extra_state_dict['m_groups'], \
                f"Requested groups of 'm' change from {self._m_groups} to {extra_state_dict['m_groups']}: not allowed!"
            #
            assert self._m_padding == extra_state_dict['m_padding'], \
                f"Requested groups of 'w' change from {self._m_padding} to {extra_state_dict['m_padding']}: not allowed!"
            #
            self._m_padding_mode = extra_state_dict['m_padding_mode']
            self.set_trainable('m', extra_state_dict['m_trainable'])
            #
            self._m_initialization = extra_state_dict['layer_type']
            # 'b_type' calculated directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            # 'initial_b' obtained directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            self.set_trainable('b', extra_state_dict['b_trainable'])
        pass

        if (index_past_layer_type >= ordered_list_layers.index('INRFv1Layer')) and \
                (index_current_layer_type >= ordered_list_layers.index('INRFv1Layer')):
            ##########
            # PARAMETERS FOR LAYERS 'INRFv1Layer', 'INRFv2Layer', 'INRFv3Layer', 'IBNNLiteLayer', 'IBNNInternalLayer', 'IBNNLayer'
            ##########
            #
            if self._sigma_activation != extra_state_dict['sigma_activation']:
                print(f"Activation 'sigma' changed from '{self._sigma_activation}' to '{extra_state_dict['sigma_activation']}'!")
            pass
            self._sigma_activation = extra_state_dict['sigma_activation']
            #
            list_sigma_modifier_keys = ['sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset']
            for key in list_sigma_modifier_keys:
                # 'sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset' is obtained directly
                # from 'self._theta', which must have been loaded by 'load_state_dict()'
                self.set_trainable(key, extra_state_dict[key + '_trainable'])
            pass
            #
            # 'lambda_type' calculated directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            # 'initial_lambda' obtained directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            self.set_trainable('lambda', extra_state_dict['lambda_trainable'])
            #
        pass

        if (index_past_layer_type >= ordered_list_layers.index('INRFv2Layer')) and \
                (index_current_layer_type >= ordered_list_layers.index('INRFv2Layer')):
            ##########
            # PARAMETERS FOR LAYERS 'INRFv2Layer', 'INRFv3Layer', 'IBNNLiteLayer', 'IBNNInternalLayer', 'IBNNLayer'
            ##########
            #
            # 'w_kernel_size' calculated directly from 'self._theta', which must have been loaded by 'load_state_dict()'
            #
            assert self._w_groups == extra_state_dict['w_groups'], \
                f"Requested groups of 'w' changed from {self._w_groups} to {extra_state_dict['w_groups']}: not allowed!"
            #
            assert self._w_padding == extra_state_dict['w_padding'], \
                f"Requested groups of 'w' changed from {self._w_padding} to {extra_state_dict['w_padding']}: not allowed!"
            #
            if self._w_padding_mode != extra_state_dict['w_padding_mode']:
                print(f"'w_padding_mode' changed from {self._w_padding_mode} to {extra_state_dict['w_padding_mode']}!")
            pass
            self._w_padding_mode = extra_state_dict['w_padding_mode']
            #
            self.set_trainable('w', extra_state_dict['w_trainable'])
            self._w_initialization = extra_state_dict['layer_type']
            #
        pass

        if (index_past_layer_type >= ordered_list_layers.index('IBNNInternalLayer')) and \
                (index_current_layer_type >= ordered_list_layers.index('IBNNInternalLayer')):
            ##########
            # PARAMETERS FOR LAYER 'IBNNInternalLayer' and 'IBNNLayer'
            ##########
            #
            self._batched_fixed_point = extra_state_dict['batched_fixed_point']
            # 'f_solver' and 'b_solver' cannot be changed in the layer FixedPointLayer... for now!
            #
        pass
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()


    @staticmethod
    @abstractmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        # This cumbersome way of defining a static method for all the inheriting classes is due to the fact that, \
        # since the method is static, the self argument is not passed, and the method defined in the parent class \
        # cannot directly access the attributes of the child class.
        raise Exception(f"Abstract-like forward method needing rewriting.")

    pass

    @staticmethod
    def _constructor_default_values_given_class(query_class, query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class `query_class`\
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When `query_args` is provided, only the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``)
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_class : class
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        signature_parameters = inspect.signature(query_class).parameters
        #
        list_arguments_of_interest = None
        if query_args is None:
            pass
        elif isinstance(query_args, (list, tuple)):
            list_arguments_of_interest = list(query_args)
        elif isinstance(query_args, str):
            list_arguments_of_interest = [query_args]
        else:
            raise Exception(f"Unknown type '{type(query_args)}' for 'arg_name'!")
        pass

        #
        dict_default_values = {}
        for key in signature_parameters:
            if key not in ['self', 'kwargs']:
                if list_arguments_of_interest is None or key in list_arguments_of_interest:
                    default_value = signature_parameters[key].default
                    default_value = None if default_value is inspect._empty else default_value
                    if (not only_not_none) or (default_value is not None):
                        dict_default_values[key] = default_value
                    pass
            pass
        pass

        return dict_default_values

    pass

    @abstractmethod
    def _update_underlying_f(self):
        raise Exception(f"Abstract-like forward method needing rewriting.")

    pass

    @abstractmethod
    def forward(self):
        raise Exception(f"Abstract-like forward method needing rewriting.")

    pass

    @property
    def calculation_mode(self):
        """
        Obtain the current ``calculation_mode`` of the class \
        (relevant for the function :py:func:`.conv2d_crossdiff`; see its documentation for further details).

        Returns
        -------
        str
        """
        return self._calculation_mode

    @property
    def memory_saving_version(self):
        """
        Obtain the current ``memory_saving_version`` of the class \
        (relevant for the function :py:func:`.conv2d_crossdiff`; see its documentation for further details).

        Returns
        -------
        str
        """
        return self._memory_saving_version

    @property
    def m_groups(self):
        """
        Obtain the number 'groups' of the filter 'theta['m']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._m_groups

    @property
    def w_groups(self):
        """
        Obtain the number 'groups' of the filter 'theta['w']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._w_groups

    @property
    def m_padding(self):
        """
        Obtain the number 'padding' of the filter 'theta['m']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._m_padding

    @property
    def w_padding(self):
        """
        Obtain the number 'padding' of the filter 'theta['w']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._w_padding

    @property
    def m_padding_mode(self):
        """
        Obtain the number 'padding_mode' of the filter 'theta['m']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._m_padding_mode

    @property
    def w_padding_mode(self):
        """
        Obtain the number 'padding_mode' of the filter 'theta['w']' of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._w_padding_mode

    def get_trainable(self, key):
        """
        Obtain whether 'theta[key]' is currently trainable or blocked.

        Parameters
        ----------
        key: str

        Returns
        -------
        bool
        """
        answer = None if key not in list(self._theta_trainable.keys()) else self._theta_trainable[key]
        return answer

    @property
    def m_trainable(self):
        """
        Obtain whether 'theta['m']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['m']

    @property
    def b_trainable(self):
        """
        Obtain whether 'theta['b']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['b']

    @property
    def lambda_trainable(self):
        """
        Obtain whether 'theta['lambda']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['lambda']

    @property
    def w_trainable(self):
        """
        Obtain whether 'theta['w']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['w']

    @property
    def sigma_x_compress_trainable(self):
        """
        Obtain whether 'theta['sigma_x_compress']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['sigma_x_compress']

    @property
    def sigma_y_stretch_trainable(self):
        """
        Obtain whether 'theta['sigma_y_stretch']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['sigma_y_stretch']

    @property
    def sigma_x_offset_trainable(self):
        """
        Obtain whether 'theta['sigma_x_offset']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['sigma_x_offset']

    @property
    def sigma_y_offset_trainable(self):
        """
        Obtain whether 'theta['sigma_y_offset']' is currently trainable or blocked.

        Returns
        -------
        bool
        """
        return self._theta_trainable['sigma_y_offset']

    @property
    def theta0(self):
        """
        Obtain 'theta0'; in fact, a detached copy of 'theta'.

        Returns
        -------
        dict
        """
        copied_theta0 = {}
        for key in self._theta0:
            copied_theta0[key] = self._theta0[key].detach().clone()
        pass

        return copied_theta0

    @property
    def theta_copy(self):
        """
        Obtain 'theta'; in fact, a detached copy of 'theta'.

        Returns
        -------
        dict
        """
        copied_theta = {}
        for key in self._theta:
            copied_theta[key] = self._theta[key].detach().clone()
        pass

        return copied_theta

    @property
    def phi_activation(self):
        """
        Obtain the current ``phi_activation`` of the class.

        Returns
        -------
        ~collections.abc.Callable or list[~collections.abc.Callable]
        """
        return self._phi_activation

    @property
    def sigma_activation(self):
        """
        Obtain the current ``sigma_activation`` of the class \
        (relevant for the function :py:func:`.conv2d_crossdiff`; see its documentation for further details).

        Returns
        -------
        str
        """
        return self._sigma_activation

    ##########################################################
    # SETTERS for TRAINABLE
    ##########################################################

    @calculation_mode.setter
    def calculation_mode(self, new_calculation_mode):
        """
        Set the ``calculation_mode`` of the class \
        (relevant for the function :py:func:`.conv2d_crossdiff`; see its documentation for further details). \
        ``None`` would imply that the default calculation mode that the downstream function would be used.

        Parameters
        ----------
        new_calculation_mode : str
        """
        self._calculation_mode = new_calculation_mode
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @memory_saving_version.setter
    def memory_saving_version(self, new_memory_saving_version):
        """
        Set the ``memory_saving_version`` of the class \
        (relevant for the function :py:func:`.conv2d_crossdiff`; see its documentation for further details). \
        ``None`` would imply that the default memory saving version that the downstream function would be used.

        Parameters
        ----------
        new_memory_saving_version : str
        """
        self._memory_saving_version = new_memory_saving_version
        # Warning: only used if...
        if self._calculation_mode != 'interpolated':
            warnings.warn((f"'memory_saving_version' is only relevant for 'calculation_mode' = 'interpolated'; " +
                           f"current value is, however, {self._calculation_mode} !"))
        pass
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def set_trainable(self, key_to_modify, trainable=None):
        """
        Set whether 'theta[key_to_modify]' is currently trainable or blocked.

        Parameters
        ----------
        key_to_modify: str
        trainable : bool
        """

        # print(f"---> Method 'set_trainable' of ModifiedRFLayer, 'key_to_modify' = {key_to_modify}!")

        first_key_to_modify = key_to_modify
        key_to_copy_to = None
        if hasattr(self, '_inrf_version') and self._inrf_version == 1 and (key_to_modify in ['m', 'w']):
            warnings.warn(f"For INRFv1 the modifications regarding 'theta['m']' and 'theta['w']' affect both.")
            # print(f"'self._inrf_version' = {self._inrf_version}")
            first_key_to_modify = 'm'
            key_to_copy_to = 'w'
        pass

        if trainable is not None:
            self._theta_trainable[first_key_to_modify] = trainable
            self._theta[first_key_to_modify].requires_grad = True if self._theta_trainable[first_key_to_modify] \
                else False
        pass
        #
        if key_to_copy_to is not None:
            self._theta_trainable[key_to_copy_to] = self._theta_trainable[first_key_to_modify]
            self._theta[key_to_copy_to].requires_grad = True if self._theta_trainable[key_to_copy_to] \
                else False
        pass
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @m_trainable.setter
    def m_trainable(self, new_m_trainable):
        """
        Set whether 'theta['m']' is currently trainable or blocked.
        """
        self.set_trainable('m', trainable=new_m_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @b_trainable.setter
    def b_trainable(self, new_b_trainable):
        """
        Set whether 'theta['b']' is currently trainable or blocked.
        """
        self.set_trainable('b', trainable=new_b_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @w_trainable.setter
    def w_trainable(self, new_w_trainable):
        """
        Set whether 'theta['w']' is currently trainable or blocked.
        """
        self.set_trainable('w', trainable=new_w_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @lambda_trainable.setter
    def lambda_trainable(self, new_lambda_trainable):
        """
        Set whether 'theta['lambda']' is currently trainable or blocked.
        """
        self.set_trainable('lambda', trainable=new_lambda_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @sigma_x_compress_trainable.setter
    def sigma_x_compress_trainable(self, new_trainable):
        """
        Set whether 'theta['sigma_x_compress']' is currently trainable or blocked.
        """
        self.set_trainable('sigma_x_compress', trainable=new_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @sigma_y_stretch_trainable.setter
    def sigma_y_stretch_trainable(self, new_trainable):
        """
        Set whether 'theta['sigma_y_stretch']' is currently trainable or blocked.
        """
        self.set_trainable('sigma_y_stretch', trainable=new_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @sigma_x_offset_trainable.setter
    def sigma_x_offset_trainable(self, new_trainable):
        """
        Set whether 'theta['sigma_x_offset']' is currently trainable or blocked.
        """
        self.set_trainable('sigma_x_offset', trainable=new_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    @sigma_y_offset_trainable.setter
    def sigma_y_offset_trainable(self, new_trainable):
        """
        Set whether 'theta['sigma_y_offset']' is currently trainable or blocked.
        """
        self.set_trainable('sigma_y_offset', trainable=new_trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def set_m(self, new_m_kernel, trainable=None):
        """
        It substitutes the current 'theta['m']' of the layer for the filter mask newly introduced.
        Two aspects must be taken into account:

        - the size, both spatially and channel-wise, of the newly set 'theta['m']' must be coincident with the \
          original filter: the coincidence is checked and an exception is raised when not fulfilled; this restriction \
          has been set to avoid errors between sizes of previously defined layers;
        - the state ``m_trainable`` remains the same it was before the function, unless stated explicitly \
          with the argument ``trainable``.

        Parameters
        ----------
        new_m_kernel : torch.Tensor
        trainable : bool
        """
        self.set_filter_mask_in_theta(new_m_kernel, 'm', trainable=trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def set_w(self, new_w_kernel, trainable=None):
        """
        It substitutes the current 'theta['w']' of the layer for the filter mask newly introduced.
        Two aspects must be taken into account:

        - the size, both spatially and channel-wise, of the newly set 'theta['w']' must be coincident with the \
          original filter: the coincidence is checked and an exception is raised when not fulfilled; this restriction \
          has been set to avoid errors between sizes of previously defined layers;
        - the state ``w_trainable`` remains the same it was before the function, unless stated explicitly \
          with the argument ``trainable``.

        Parameters
        ----------
        new_w_kernel : torch.Tensor
        trainable : bool
        """
        self.set_filter_mask_in_theta(new_w_kernel, 'w', trainable=trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def set_b(self, new_b, trainable=None):
        """
        It substitutes the current 'theta['b']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['b']' must be coincident with the \
          original value; at least the number of elements, even if the are provided in a different format \
          e.g. a float instead of a 1-element tensor;
        - the state ``b_trainable`` remains the same it was before the function, unless stated explicitly \
          with the argument ``trainable``.

        Parameters
        ----------
        new_b : torch.Tensor
        trainable : bool
        """

        self.set_scalar_like_in_theta(new_b, 'b', trainable=trainable)
        #
        # And make sure that the changes apply to the underlying f
        self._update_underlying_f()

    def set_lambda(self, new_lambda, trainable=None):
        """
        It substitutes the current 'theta['lambda']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['lambda']' must be coincident with the \
          original value; at least the number of elements, even if the are provided in a different format \
          e.g. a float instead of a 1-element tensor;
        - the state ``lambda_trainable`` remains the same it was before the function, unless stated explicitly \
          with the argument ``trainable``.

        Parameters
        ----------
        new_lambda : torch.Tensor
        trainable : bool
        """
        self.set_scalar_like_in_theta(new_lambda, 'lambda', trainable=trainable)

    def set_sigma_x_compress(self, new_value, trainable=None):
        """
        It substitutes the current 'theta['sigma_x_compress']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['sigma_x_compress']' \
          must be coincident with the original value; at least the number of elements, even if \
          they are provided in a different format e.g. a float instead of a 1-element tensor;
        - the state of trainability of 'sigma_x_compress' remains the same it was before the function, \
          unless stated explicitly with the argument ``trainable``.

        Parameters
        ----------
        new_value : torch.Tensor
        trainable : bool
        """
        self.set_scalar_like_in_theta(new_value, 'sigma_x_compress', trainable=trainable)

    def set_sigma_y_stretch(self, new_value, trainable=None):
        """
        It substitutes the current 'theta['sigma_y_stretch']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['sigma_y_stretch']' \
          must be coincident with the original value; at least the number of elements, even if \
          they are provided in a different format e.g. a float instead of a 1-element tensor;
        - the state of trainability of 'sigma_y_stretch' remains the same it was before the function, \
          unless stated explicitly with the argument ``trainable``.

        Parameters
        ----------
        new_value : torch.Tensor
        trainable : bool
        """
        self.set_scalar_like_in_theta(new_value, 'sigma_y_stretch', trainable=trainable)

    def set_sigma_x_offset(self, new_value, trainable=None):
        """
        It substitutes the current 'theta['sigma_x_offset']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['sigma_x_offset']' \
          must be coincident with the original value; at least the number of elements, even if \
          they are provided in a different format e.g. a float instead of a 1-element tensor;
        - the state of trainability of 'sigma_x_offset' remains the same it was before the function, \
          unless stated explicitly with the argument ``trainable``.

        Parameters
        ----------
        new_value : torch.Tensor
        trainable : bool
        """
        self.set_scalar_like_in_theta(new_value, 'sigma_x_offset', trainable=trainable)

    def set_sigma_y_offset(self, new_value, trainable=None):
        """
        It substitutes the current 'theta['sigma_x_compress']' of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['sigma_y_offset']' \
          must be coincident with the original value; at least the number of elements, even if \
          they are provided in a different format e.g. a float instead of a 1-element tensor;
        - the state of trainability of 'sigma_y_offset' remains the same it was before the function, \
          unless stated explicitly with the argument ``trainable``.

        Parameters
        ----------
        new_value : torch.Tensor
        trainable : bool
        """
        self.set_scalar_like_in_theta(new_value, 'sigma_y_offset', trainable=trainable)

    def set_filter_mask_in_theta(self, new_filter_mask, key_to_modify, trainable=None):
        """
        It substitutes the current 'theta['m']' or 'theta['w']' of the layer for a new value. \
        **WARNING:** If INRFv1 it makes sure that the correct 'm'-'w' links are kept.

        To be taken into account:

        - the size, both spatially and channel-wise, of the newly set 'theta[key_to_modify]' must be coincident with \
          the original value; at least the number of elements, even if the are provided in a different format \
          e.g. a float instead of a 1-element tensor;
        - the state *trainable* of the indicated key remains the same it was before the function, \
          unless stated explicitly with the argument ``trainable``.

        Parameters
        ----------
        new_filter_mask : torch.Tensor
        key_to_modify: 'm' or 'w'
        trainable : bool
        """

        first_key_to_modify = key_to_modify
        key_to_copy_to = None
        if hasattr(self, '_inrf_version') and self._inrf_version == 1 and (key_to_modify in ['m', 'w']):
            warnings.warn(f"For INRFv1 the modifications regarding 'theta['m']' and 'theta['w']' affect both.")
            first_key_to_modify = 'm'
            key_to_copy_to = 'w'
        pass

        if trainable is not None:
            self._theta_trainable[first_key_to_modify] = trainable
        pass

        # Check the size of the newly suggested tensor, comparing to the previous one
        if new_filter_mask.size() != self._theta[first_key_to_modify].size():
            raise Exception((
                    f"The size of the newly introduced 'new_filter_mask', {new_filter_mask.size()}, is not coincident " +
                    f"with the size of the originally set filter 'theta['{new_filter_mask}']', " +
                    f"{self._theta[first_key_to_modify].size()}"
            ))
        pass
        #
        self._theta[first_key_to_modify].requires_grad = False
        self._theta[first_key_to_modify][:] = new_filter_mask[:]
        self._theta[first_key_to_modify].requires_grad = self._theta_trainable[first_key_to_modify]
        #
        if key_to_copy_to is not None:
            self._theta[key_to_copy_to] = self._theta[first_key_to_modify]
            self._theta_trainable[key_to_copy_to] = self._theta_trainable[first_key_to_modify]
        pass

    def set_scalar_like_in_theta(self, new_scalar_like, key_to_modify, trainable=None):
        """
        It substitutes the current 'theta['b']', 'theta['lambda']' (or other scalar-like value in 'theta' in potential \
        future modifications) of the layer for a new value.

        - the size, both spatially and channel-wise, of the newly set 'theta['lambda']' must be coincident with the \
          original value; at least the number of elements, even if the are provided in a different format \
          e.g. a float instead of a 1-element tensor;
        - the state ``lambda_trainable`` remains the same it was before the function, unless stated explicitly \
          with the argument ``trainable``.

        Parameters
        ----------
        new_scalar_like : torch.Tensor
        key_to_modify: 'b' or 'lambda'
        trainable : bool
        """

        if trainable is not None:
            self._theta_trainable[key_to_modify] = trainable
        pass

        # Check the size of the newly suggested 'lambda', compared to the previous one
        numel_new_scalar_like = 0
        new_scalar_like_as_tensor = None
        if isinstance(new_scalar_like, float) or isinstance(new_scalar_like, int):
            numel_new_scalar_like = 1
            new_scalar_like_as_tensor = torch.Tensor([new_scalar_like])
        elif isinstance(new_scalar_like, list) or isinstance(new_scalar_like, tuple):
            numel_new_scalar_like = len(new_scalar_like)
            new_scalar_like_as_tensor = torch.Tensor(new_scalar_like[:])
        elif isinstance(new_scalar_like, torch.Tensor):
            numel_new_scalar_like = new_scalar_like.numel()
            new_scalar_like_as_tensor = new_scalar_like.flatten()
        pass
        if numel_new_scalar_like != self._theta[key_to_modify].numel():
            raise Exception((
                    f"The size of the newly introduced 'new_w_kernel', of type {type(new_scalar_like)}, is " +
                    f"{numel_new_scalar_like} elements; it is not coincident " +
                    f"with the size of the originally set 'theta['lambda']', of size {self._theta[key_to_modify].size()}"
            ))
        pass

        new_scalar_like_as_tensor_and_compatible_size = new_scalar_like_as_tensor.view(
            self._theta[key_to_modify].size()
        )

        self._theta[key_to_modify].requires_grad = False
        self._theta[key_to_modify][:] = new_scalar_like_as_tensor_and_compatible_size[:]
        self._theta[key_to_modify].requires_grad = self._theta_trainable[key_to_modify]

    def set_field_in_theta(self, new_in_theta, key_to_modify, trainable=None):
        """
        It substitutes the current 'theta['m']', 'theta['w']', 'theta['b']', 'theta['lambda']', depending on the \
        given `key_to_modify`, for its new value indicated by `new_in_theta`.

        The function calls internally :py:meth:`.set_filter_mask_in_theta` or :py:meth:`.set_scalar_like_in_theta`, \
        respectively, depending on the provided `key_to_modify`; therefore their respective documentation apply.


        Parameters
        ----------
        new_in_theta : torch.Tensor
        key_to_modify: 'm', 'b', 'w' or 'lambda'
        trainable : bool
        """

        if key_to_modify in ['m', 'w']:
            self.set_filter_mask_in_theta(new_in_theta, key_to_modify, trainable=trainable)
        elif key_to_modify in ['b', 'lambda',
                               'sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset']:
            self.set_scalar_like_in_theta(new_in_theta, key_to_modify, trainable=trainable)
        else:
            raise Exception(f"Unknown key '{key_to_modify}' provided: only 'm', 'b', 'w' or 'lambda' accepted!")
        pass


######################################################
# ibnn_internal Layer
######################################################


class IBNNInternalLayer(ModifiedRFLayer, FixedPointLayer):
    """
    Layer implementing the *Implicitly-Biased Neural Network* (ibnn_internal) layer.

    In greater detail: the later obtains the fixed point solution \
    $\\mathbf{U}^* = \\mathbf{f}_{\\Theta}(\\mathbf{I}, \\mathbf{U})$, wherein \
    $\\Theta = \\big(\\mathbf{M}, \\mathbf{b}, \\mathbf{\\Omega}, \\mathbf{\\lambda} \\big)$ \
    represents a set of training parameters ruling the specific behavior of the function

    $$\\mathbf{f}_{\\Theta}(\\mathbf{I}, \\mathbf{U})(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{A} (\\mathbf{I}))(\\mathbf{p})
    - \\mathbf{\\lambda}
    \\big( \\mathbf{\\Omega} \\ast \\mathbf{\\sigma} ( \\mathbf{U}-\\mathbf{U}[\\mathbf{p}] ) \\big) (\\mathbf{p})
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
    $$

    implemented through the corresponding particularization of the function :py:func:`.f_modified_RF`. \
    And, analogous to the latter, it accepts general affine transform $\\mathbf{A} (\\mathbf{I}))$ in the form \
    of a fully-connected plus bias but also convolutions: in that case the layer operation can be written as

    $$\\mathbf{f}_{\\Theta}(\\mathbf{I}, \\mathbf{U})(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{M} \\ast \\mathbf{I})(\\mathbf{p})
    - \\mathbf{b}
    - \\mathbf{\\lambda}
    \\big( \\mathbf{\\Omega} \\ast \\mathbf{\\sigma} ( \\mathbf{U}-\\mathbf{U}[\\mathbf{p}] ) \\big) (\\mathbf{p})
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    Input $\\mathbf{I}$ and output/hidden signals $\\mathbf{U}$ do not need to have the same size, but to be compatible:
    that is, the size of $\\mathbf{U}$ will correspond to that of the result of the affine transform \
    $\\mathbf{A} (\\mathbf{I})$, which will be a general convolution if ``m_padding`` is ``'fc'``, \
    or a convolution, to the result of $\\mathbf{M} \\ast \\mathbf{I}$, \
    which depends on the size of $\\mathbf{M}$ and the parameter ``m_padding``, \
    which can be ``'valid'`` or ``'same'``.

    Input $\\mathbf{I}$ and output/hidden signals $\\mathbf{U}$ \
    can also differ in their respective number of channels: \
    the number of channels of the output of $\\mathbf{A} (\\mathbf{I})$, given in fact by $\\mathbf{M}$,
    simply needs to be coincident with the number
    of channels of $\\mathbf{U}$, which has to be the same of the output of
    $\\mathbf{\\Omega} \\!\\ast\\! \\mathbf{\\sigma}(\\mathbf{U})$, given in fact by $\\mathbf{\\Omega}$.
    That is:
    the respective sizes of $\\mathbf{M}$ and $\\mathbf{\\Omega}$ will be
    $(C_U, \\frac{C_I}{\\mathrm{groups}_M}, H_A, W_A)$ and $(C_U, \\frac{C_U}{\\mathrm{groups}_U}, H_U, W_U)$.

    The initialization of the parameters \
    $\\Theta = \\big(\\mathbf{M}, \\mathbf{b}, \\mathbf{\\Omega}, \\mathbf{\\lambda}, \\Theta_{\\sigma} \\big)$ \
    of the network will be performed deterministically according to the value of the respective arguments \
    ``m_initialization``, ``initial_b``, ``w_initialization``, ``initial_lambda``, \
    ``sigma_x_compress``, ``sigma_y_stretch``, ``sigma_x_offset``, and ``sigma_y_offset=0.0``: as far as the \
    initialization of the filter masks for $M$ and $\\Omega$ is concerned refer to the description of the \
    arguments of the constructors below. The randomization/random initialization required prior to the training \
    can be achieved using the method :py:meth:`~ModifiedRFLayer.random_initialization` of the parent class \
    :py:class:`ModifiedRFLayer`.

    **IMPORTANT NOTE (1):**  $\\mathbf{M}$ and $\\mathbf{b}$ **performing a fully-connected (linear) operation,** \
    **instead of a convolutional operation, has been included, which corresponds to** `m_padding`=``‘fc’``. \
    In such case certain arguments of the constructor are directly ignored (e.g. `b_type`), and \
    `m_kernel_size` **will correspond to the spatial extent of the image resulting from** $\\mathbf{A} (\\mathbf{I})$.
    The main difference between the pseudo-convolutional layers ``'same'``/``'valid'`` and the fully-connected \
    layer ``'fc'`` is the following: for the former, `m_kernel_size` represents, precisely, the spatial size of the \
    desired filter mask (which, along with `in_channels`, `out_channels`, and the knowledge of the padding type, \
    would help deduce the output size of the function; however, for ``'fc'``, `in_size` and `out_size` must be given and \
    then the size of the linear transform matrices can be deduced if necessary.

    **IMPORTANT NOTE (2): Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.


    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`, \
        see also :py:func:`.f_modified_RF`
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        Size of the kernel convolving the input image $\\mathbf{I}$, if the network is created as pseudo-convolutional \
        or convolutional-like; said parameter is not necessary nor used when a fully-convolutional (FC) affine \
        transform, instead of a convolution, is selected (see ``m_padding`` for the selection of convolutional vs FC \
        affine transforms. \
        So, for the case of convolutional affine transform: if tuple, $(H,W)$; if scalar S, $(S, S)$ \
        (see above for the implications of each version); if the input value(s) is (are) float 0<x<0 \
        the absolute pixel size of the kernel will be inferred from the argument ``in_size`` which, although optional, \
        becomes compulsory in such case.
    m_padding : str, optional
        Value among ``‘same’``, `‘valid’``, and ``‘fc’``. The options ``‘same’`` and `‘valid’`` correspond to a \
        convolutional affine transform and correspond to the usual definitions for padding used e.g. in the \
        function :py:class:`torch.nn.Conv2d`: see :py:func:`.conv2d_adapted` for a detailed explanation. \
        However, the option ``‘fc’`` corresponds to a fully-convolutional \
        affine transform, which renders the rest of options related to $\\mathbf{M}$ and $\\mathbf{b}$ meaningless: \
        in particular the arguments ``m_padding_mode`` and ``b_type`` are simply \
        not used in such case (and set to defaults for the ``‘fc’`` case).
        Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'`` \
        (see above for the implications of each version).
        See :py:func:`.conv2d_adapted` for a detailed explanation. Default: ``'zeros'``
    m_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Initialization mode for the filter mask $\\mathbf{M}$. If string, the provided value corresponds to the \
        argument ``initialization_type`` of the method :py:func:`.filter_initialization`; if dictionary, \
        the provided keys correspond to ``initialization_type`` and the corresponding keyword arguments \
        of the method :py:func:`.filter_initialization`; if a scalar (float or int) all elements of the filter \
        will be set to the provided value; and if tensor it will need to exactly match the size of requested filter
        size ``m_kernel_size``. \
        Default: ``'zeros'``
    m_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary, \
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    b_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``, or ``None``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). \
        In the special case where the affine transformation is not convolutional (`m_padding`=``‘fc’``) \
        its value will be forced to ``None`` if not so provided and the dimensions will \
        fit the need of the underlying :py:func:`torch.nn.functional.linear`.
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        If ``b_type`` is ``'scalar_per_channel'`` and the provided value is a scalar, that value will correspond \
        to all the dimensions of the vector. \
        Default: ``0.0``
    b_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    in_size : tuple[int], optional
        2D tuple (2D at least; trailing dimensions do not matter for the involved processing regarding only \
        spatial extent) indicating the size of the input image to the layer. This argument is not necessary if \
        the kernel sizes for *m* and *w* are provided as absolute int values; however it becomes necessary if \
        any of them is provided as relative (e.g. floats $0 \\leq x \\leq 1$) to the input size, and \
        in the case of ``m_padding`` ``'fc'``.
        Default: ``None``
    out_size : tuple[int], optional
        2D tuple (2D at least; trailing dimensions do not matter for the involved processing regarding only \
        spatial extent) indicating the size of the output image to the layer. This argument is not necessary unless \
        `m_padding` is ``'fc'`` since in that case it is necessary for setting the fully-connected layer connecting \
        input and output and initializing the corresponding matrices.
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`, \
        see also :py:func:`.f_modified_RF`. \
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifier to the basic sigma activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifier to the basic sigma activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Offset modifiers to the basic activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``False``
    lambda_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
        If ``lambda_type`` is ``'scalar_per_channel'`` and the provided value is a scalar, that value will correspond \
        to all the dimensions of the vector
    lambda_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        Size of the kernel convolving the activated difference of hidden signals. \
        If tuple, $(H,W)$; if scalar S, $(S, S)$. If list/tuple of length $0$ then \
        full-image, uniform product and summation is used. If the input value(s) is (are) float 0<x<0 \
        the absolute pixel size of the kernel will be inferred from the argument ``in_size`` which, although optional, \
        becomes compulsory in such case
    w_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'`` \
        (see above for the implications of each version). \
        In the case of ``w_kernel_size``=`'full'`` no padding is applied so this parameter does not apply. \
        See :py:func:`.conv2d_adapted` for a detailed explanation. Default: ``'zeros'``
    w_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Initialization mode for the filter mask $\\mathbf{\\Omega}$. If string, the provided value corresponds to the \
        argument ``initialization_type`` of the method :py:func:`.filter_initialization`; if dictionary, \
        the provided keys correspond to ``initialization_type`` and the corresponding keyword arguments \
        of the method :py:func:`.filter_initialization`; if a scalar (float or int) all elements of the filter \
        will be set to the provided value; and if tensor it will need to exactly match the size of requested filter
        size ``w_kernel_size``. \
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``False``
    batched_fixed_point : bool, optional
        It indicates whether the calculation in :py:meth:`.forward` of the fixed point  for an input batch
        of $B$ elements (that is, for a 4D input) should operate on each one of its $B$ individual 3D entries
        (if ``False``) or, on the contrary (if ``True``), on the 4D input as one single item.
        Default: ``True``
    f_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, and ``'broyden'``, \
        canonical solvers of the \
        library `TorchDEQ <https://torchdeq.readthedocs.io/en/latest/get_started.html#quick-start>`_, \
        or the values ``'fpgd'`` and ``'pgd'``. The underlying solver functions for the canonical solvers are, \
        respectively, :py:func:`~torchdeq.solver.fp_iter.fixed_point_iter`, \
        :py:func:`~torchdeq.solver.fp_iter.simple_fixed_point_iter`, \
        :py:func:`~torchdeq.solver.anderson.anderson_solver`, and :py:func:`~torchdeq.solver.broyden.broyden_solver`; \
        **the options** ``'fpgd'`` **and** ``'pgd'``, \
        **specific for the ibnn_internal problem and therefore for the forward pass only,** \
        **correspond respectively to the algorithms projected gradient descent and fixed point gradient descent** \
        **described, respectively, by Thomas Batard and Marcelo Bertalmio; both rely on a certain time constant** \
        $\\tau$ **which is needs to be also defined.** Regarding ``'fpgd'``: its implementation is \
        coincident with that of ``'fixed_point_iter'``.
        Default: ``'fixed_point_iter'``
    f_max_iter : int, optional
        Default: ``51``
    f_tol : float
        Stop condition for the forward fixed point calculation. Default: ``1e-5``
    f_tau : float, optional
        Float in the range $(0.0, 1.0]$. For the solver algorithms allowing so, \
        i.e. ``'pgd'``, ``'fixed_point_iter'``/``'fpgd'``, ``'simple_fixed_point_iter'``, ``'anderson'``, \
        it servers as a dampening factor for the calculation (see :py:class:`.FixedPointLayer`).
        Default: ``1.0``
    b_solver : str, optional
        Value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, ``'broyden'``. \
        Type of fixed point problem algorithm for the fixed point calculation \
        used in the gradient calculation of the IFT. Default: ``'fixed_point_iter'``
    b_max_iter : int, optional
        Default: ``40``
    b_tol : float
        Stop condition for the backward fixed point calculation.
        Default: ``1e-6``
    abs_error_threshold : float, optional
        The absolute error threshold, measured as the norm of the error between f(x) and x, \
        for the fixed point calculation. Default: ``1e-5``
    kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.f_modified_RF` (see therein for detailed information about the meaning of each parameter and which
        sub-function makes use of each one):

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'interpolated'``, ``'n4'``. Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; \
            ``True``, non-uniform sampling. Default: see :py:func:`.generate_builder_1_order_interpolation`

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
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=True,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 batched_fixed_point=True,
                 f_solver='fixed_point_iter', f_max_iter=50, f_tol=1e-5, f_tau=0.1,
                 b_solver='fixed_point_iter', b_max_iter=40, b_tol=1e-6,
                 abs_error_threshold=1e-5,
                 **kwargs):

        ###
        # Call the constructor "__init__" of the super class 'ModifiedRFLayer' taking care of the kwargs
        ###
        ModifiedRFLayer.__init__(self, **kwargs)

        # For convenience: create the list with the names of the modifier fields for the activation sigma
        list_sigma_modifier_keys = ['sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset']

        #######################################
        # INITIAL CHECKS REGARDING COMPATIBILITIES OF CHANNEL NUMBERS AND IMAGE DIMENSIONS
        #######################################

        ###
        # Scalar vs scalar-per-channel types: valid parameters?
        ###

        # for type_param in [b_type, lambda_type]:
        #     if not type_param in ['scalar_per_channel', 'scalar']:
        #         raise Exception((f"Valid values for 'b_type' and 'lambda_type' " +
        #                          f"are 'scalar_per_channel', 'scalar': '{type_param}' provided!"))
        #     pass
        # pass
        # Now 'b_type' is checked by the function 'm_and_b_dimensioning'
        for type_param in [lambda_type]:
            if not type_param in ['scalar_per_channel', 'scalar']:
                raise Exception((f"Valid values for 'lambda_type' " +
                                 f"are 'scalar_per_channel', 'scalar': '{type_param}' provided!"))
            pass
        pass

        ###
        # Check that 'initial_lambda' and 'w_kernel_size', compulsory but defaulting None for practical reasons,
        # are not None!
        ###

        aux_dict = {'initial_lambda': initial_lambda, 'w_kernel_size': w_kernel_size}
        for key in aux_dict:
            if aux_dict[key] is None:
                raise Exception(f"Argument '{key}' (different from None) is compulsory!")
            pass
        pass

        ###
        # Initial checks regarding compatibility of channel number vs groups vs image dimensions
        ###

        # And fix 'w_padding' since no other value is allowed
        dict_kernel_size = {'m': m_kernel_size,
                            'w': w_kernel_size}
        w_padding = 'same'
        dict_padding = {'m': m_padding,
                        'w': w_padding}
        dict_padding_mode = {'m': None if m_padding=='fc' else m_padding_mode,
                             'w': w_padding_mode}
        dict_groups = {'m': m_groups,
                       'w': w_groups}
        dict_kernel_out_in_channels = {'m': (out_channels, in_channels),
                                       'w': (out_channels, out_channels)}

        # Check/adapt/resolve the kernel size when m,b/w is convolutional
        for key in dict_kernel_size: # 'm' will be now checked by the function 'm_and_b_dimensioning'
            if dict_padding[key] in ['fc']:
                pass
            elif dict_padding[key] in ['same', 'valid']:
                # In this case the kernel size must be provided
                if dict_kernel_size[key] is None:
                    raise Exception(
                        (f"No kernel size for '{key}', or 'None', is provided! (Reminder: " +
                         f"full-image, uniform kernels are noted as '[]' or '()'!)")
                    )
                pass
                #
                # Reformat to help checks (not resolving yet for relative filters)
                dict_kernel_size[key] = nnl.kernel_size_check_and_reformat_into_tuple(dict_kernel_size[key])
                if len(dict_kernel_size[key]) == 0 and key == 'm':
                    raise Exception(f"Full-image, uniform kernels requested for '{key}: only valid for 'w'!")
                pass
                #
                # And resolve the (spatial) kernel size if necessary
                dict_kernel_size[key] = nnl.resolve_kernel_size_for_im_size(
                    dict_kernel_size[key], im_size=in_size, make_odd=False
                )
            else:
                raise Exception(
                    (f"Valid values for '{key}_padding' are " + ("'fc, " if key == 'm' else "") +
                     f"'same', and 'valid'; '{dict_padding[key]}' provided!"))
            pass
            #
        pass

        # Check the validity of the groups
        for key in ['m', 'w']:
            if (dict_kernel_out_in_channels[key][0] % dict_groups[key] != 0) or \
                    (dict_kernel_out_in_channels[key][1] % dict_groups[key] != 0):
                raise Exception((f"Both 'in_channels'={dict_kernel_out_in_channels[key][1]} and " +
                                 f"'out_channels'={dict_kernel_out_in_channels[key][0]} of '{key}' " +
                                 f"must be divisible by 'm_groups'={dict_groups[key]}!"))
            pass
        pass

        # Create templates (only size is important) for M and b according to the function 'm_and_b_dimensioning'!!!
        # The function 'm_and_b_dimensioning' additionally checks the compatibility of the different arguments
        template_m, template_b = m_and_b_dimensioning(
            in_channels=in_channels, out_channels=out_channels,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_groups=m_groups,
            b_type=b_type, in_size=in_size
        )

        #######################################
        # STORE PARAMETERS THAT WILL BE RELEVANT LATER ON AS ATTRIBUTES
        #######################################

        self._phi_activation = phi_activation

        self._m_kernel_size = dict_kernel_size['m']
        self._m_groups = dict_groups['m']
        self._m_padding = dict_padding['m']
        self._m_padding_mode = m_padding_mode

        self._m_initialization = m_initialization if isinstance(m_initialization, (str, dict)) else 'custom'

        self._b_type = {None if m_padding=='fc' else b_type}

        self._w_kernel_size = dict_kernel_size['w']
        self._w_groups = dict_groups['w']
        self._w_padding = dict_padding['w']
        self._w_padding_mode = w_padding_mode

        self._w_initialization = w_initialization if isinstance(m_initialization, (str, dict)) else 'custom'

        self._lambda_type = lambda_type

        # self._sigma_activation_f = nnl.ndim_activation_function_from_1dim_activation_functions(sigma_activation)
        self._sigma_activation = sigma_activation

        self._calculation_mode = kwargs.pop('calculation_mode', None)
        self._memory_saving_version = kwargs.pop('memory_saving_version', None)

        #######################################
        # Define an initial "_theta0" (kept as record) and transfer it to the dynamically optimized "_theta"
        ###
        # "_theta" is initialized deterministically as "_theta0" according to some desired rules. Later, and before
        # training, random disturbances can be introduced with the appropriate method of the class
        #######################################

        # Initial and "dynamic" parameter set
        self._theta0 = {}

        dict_initialization = {'m': m_initialization,
                               'w': w_initialization}

        for key in ['m', 'w']:
            #
            #####
            # Set the correct size for m/w and create an empty matrix for it
            #####
            #
            tensor_dims = template_m.size() if key == 'm' else \
                (dict_kernel_out_in_channels[key][0],
                 int(dict_kernel_out_in_channels[key][1] / dict_groups[key])) + dict_kernel_size[key]
            self._theta0[key] = torch.zeros(tensor_dims)
            #
            #####
            # Initialize its content
            #####
            #
            if isinstance(dict_initialization[key], str):
                self._theta0[key] = nnl.filter_initialization(
                    tensor_dims, initialization_type=dict_initialization[key],
                    im_size=in_size, groups=dict_groups[key],
                    dtype=torch.float32
                )
            elif isinstance(dict_initialization[key], dict):
                self._theta0[key] = nnl.filter_initialization(
                    tensor_dims,
                    im_size=in_size, groups=dict_groups[key],
                    **dict_initialization[key], dtype=torch.float32
                )
            elif isinstance(dict_initialization[key], (int, float)):
                self._theta0[key] = torch.empty(tensor_dims)
                self._theta0[key][:] = dict_initialization[key]
            elif isinstance(dict_initialization[key], torch.Tensor):
                if dict_initialization[key].size() != tensor_dims:
                    raise Exception((
                            f"The size of the provided tensor for '{key}', {dict_initialization[key].size()}, " +
                            f"is not compatible with the requested size {tensor_dims}")
                    )
                self._theta0[key] = torch.empty(tensor_dims)
                self._theta0[key][:] = dict_initialization[key]
                pass
            pass
        pass

        # For 'b' and 'lambda':
        dict_type = {'b': b_type,
                     'lambda': lambda_type}
        dict_initial_value = {'b': initial_b,
                              'lambda': initial_lambda}
        for key in ['b', 'lambda']:
            #
            # Set the correct size for m/w and create an empty matrix for it
            tensor_dims = None
            if key == 'b':
                self._theta0[key] = torch.empty(template_b.size())
            else:
                if dict_type[key] == 'scalar_per_channel':
                    self._theta0[key] = torch.empty(out_channels, 1, 1)
                else:  # 'scalar'
                    self._theta0[key] = torch.empty(1)
                pass
            pass
            #
            if isinstance(dict_initial_value[key], int) or isinstance(dict_initial_value[key], float):
                self._theta0[key][:] = dict_initial_value[key]
            elif isinstance(dict_initial_value[key], torch.Tensor):
                self._theta0[key][:] = dict_initial_value[key][:]
            else:
                raise Exception((f"The selected initialization value for '{key}' is not an accepted type " +
                                 f"float, int, or torch.Tensor," +
                                 f"but the following: {type(dict_initial_value[key])}"))
            pass
            #
        pass

        # For the sigma modifiers:
        dict_sigma_modifier_values = {
            'sigma_x_compress': sigma_x_compress, 'sigma_y_stretch': sigma_y_stretch,
            'sigma_x_offset': sigma_x_offset, 'sigma_y_offset': sigma_y_offset
        }
        for key in dict_sigma_modifier_values:
            if isinstance(dict_sigma_modifier_values[key], int) or isinstance(dict_sigma_modifier_values[key], float):
                self._theta0[key] = torch.Tensor([dict_sigma_modifier_values[key]])
            elif isinstance(dict_sigma_modifier_values[key], torch.Tensor) or \
                    isinstance(dict_sigma_modifier_values[key], list) or \
                    isinstance(dict_sigma_modifier_values[key], tuple):
                self._theta0[key] = torch.Tensor(dict_sigma_modifier_values[key])
            else:
                raise Exception((f"The selected initialization value for '{key}' is not an accepted type " +
                                 f"float, int, torch.Tensor, list, or tuple" +
                                 f"but the following: {type(dict_sigma_modifier_values[key])}"))
        pass

        #######################################
        # CREATE THE (HIDDEN) ATTRIBUTE "_theta_trainable",
        # WHICH IS A DICTIONARY OF BOOLS MIRRORING THE FIELDS OF "_theta"
        #######################################

        self._theta_trainable = {}
        for key in self._theta0:
            self._theta_trainable[key] = False
        pass
        self._theta_trainable['m'] = m_trainable
        self._theta_trainable['b'] = b_trainable
        self._theta_trainable['w'] = w_trainable
        self._theta_trainable['lambda'] = lambda_trainable
        self._theta_trainable['sigma_x_compress'] = sigma_x_compress_trainable
        self._theta_trainable['sigma_y_stretch'] = sigma_y_stretch_trainable
        self._theta_trainable['sigma_x_offset'] = sigma_x_offset_trainable
        self._theta_trainable['sigma_y_offset'] = sigma_y_offset_trainable


        #######################################
        # FINALLY call the constructor "super.__init__" of the super class FixedPointLayer, which:
        # - creates the parameters "self._theta" from "self.theta0"
        #######################################

        surrogate_f_solver = None
        surrogate_f_max_iter = None
        surrogate_f_tol = None
        surrogate_f_tau = None
        if f_solver=='pgd':
            surrogate_f_solver = 'fixed_point_iter'
            surrogate_f_max_iter = 1    # No extra iterations are allowed
            surrogate_f_tol = f_tol
            surrogate_f_tau = 0.0
        elif f_solver=='fpgd':
            surrogate_f_solver = 'fixed_point_iter'
            surrogate_f_max_iter = f_max_iter   # The 'fixed_point_iter' does everything
            surrogate_f_tol = f_tol
            surrogate_f_tau = f_tau
        elif f_solver in ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden']:
            surrogate_f_solver = f_solver
            surrogate_f_max_iter = f_max_iter  # The 'fixed_point_iter' does everything
            surrogate_f_tol = f_tol
            surrogate_f_tau = f_tau
        else:
            raise Exception(f"Invalid value for 'f_solver': '{f_solver}: " +
                            f"only 'fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden', " +
                            f"'pgd', and 'fpgd' are accepted!")
        pass

        # # The 'f' to use by the fixed point operations will be defined later, not with the constructor, using
        # # the auxiliary method '._update_underlying_f(...)' (see few lines below)
        #
        # # If the values of the forward solver are different from those of the surrogate forward solver, indicate them
        # surrogate_f_kwargs = {}
        #
        # if f_solver=='pgd':
        #     # The constructor of FixedPointLayer will take care of the rest, no need for explicit surrogate definitions
        #     surrogate_f_kwargs = {}
        # elif f_solver=='fpgd': # The surrogate is 'fixed_point_iter' and does everything
        #     surrogate_f_kwargs = {
        #         'surrogate_f_solver': 'fixed_point_iter',
        #         'surrogate_f_max_iter': f_max_iter,
        #         'surrogate_f_tol': f_tol,
        #         'surrogate_f_tau': f_tau
        #     }
        # elif f_solver in ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden']:
        #     surrogate_f_kwargs = {}
        # else:
        #     raise Exception(f"Invalid value for 'f_solver': '{f_solver}: " +
        #                     f"only 'fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden', " +
        #                     f"'pgd', and 'fpgd' are accepted!")
        # pass

        # Other related arguments
        self._batched_fixed_point = batched_fixed_point

        copy_of_theta0 = {}
        for key in self._theta0:
            copy_of_theta0[key] = self._theta0[key].detach().clone()
        pass

        ift = True
        hook_ift = False
        FixedPointLayer.__init__(self,
                                 f=None,
                                 theta0=copy_of_theta0, random_initialization_theta=False,
                                 y0=None,
                                 batched_fixed_point=batched_fixed_point,
                                 ift=ift, hook_ift=hook_ift,
                                 f_solver=f_solver, f_max_iter=f_max_iter, f_tol=f_tol, f_tau=f_tau,
                                 surrogate_f_solver=surrogate_f_solver, surrogate_f_max_iter=surrogate_f_max_iter,
                                 surrogate_f_tol=surrogate_f_tol, surrogate_f_tau=surrogate_f_tau,
                                 b_solver=b_solver, b_max_iter=b_max_iter, b_tol=b_tol,
                                 abs_error_threshold=abs_error_threshold
                                 )

        # And we set the trainable status
        for key in self._theta:
            self._theta[key].requires_grad = self._theta_trainable[key]
        pass

        self._update_underlying_f()

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(IBNNInternalLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)
    pass

    def _update_underlying_f(self):
        def particularized_f_IBNN(u_f, im_f, theta_f):
            return f_modified_RF(
                im_f, theta_f, u=u_f, v=None,
                phi_activation=self._phi_activation, sigma_activation=self._sigma_activation,
                m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups,
                w_padding_mode=self._w_padding_mode, w_groups=self._w_groups,
                calculation_mode=self._calculation_mode, memory_saving_version=self._memory_saving_version,
                **self._fixed_kwargs
            )

        self._f_function_of_y_x_theta = particularized_f_IBNN

    pass

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Copy parameters and buffers from state_dict into this module and its descendants. It overloads the function \
        of the same name (:py:meth:`~torch.nn.Module.load_state_dict`) in :py:class:`torch.nn.Module` to \
        update the underlying function of the class using the newly loaded parameters using the method \
        :py:meth:`.set_filter_mask_in_theta`.

        As in :py:meth:`~torch.nn.Module.load_state_dict`, if strict is True, \
        then the keys of state_dict must exactly match the keys returned by \
        :py:meth:`~torch.nn.Module.state_dict`.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing parameters and persistent buffers
        strict : bool, optional
            Whether to strictly enforce that the keys in ``state_dict`` match the keys returned by this module's \
            :py:meth:`~torch.nn.Module.state_dict`. Default: ``True``
        assign : bool, optional
            When ``False``, the properties of the tensors in the current module are preserved while when ``True``, \
            the properties of the Tensors in the state dict are preserved. The only exception is the \
            ``requires_grad`` field. Default: ``False``

        Returns
        -------
        missing_keys : NamedTuple
            Keys (str) that are expected by this module but missing from the provided ``state_dict``
        unexpected_keys : NamedTuple
            Keys (str) that are not expected by this module but present in the provided ``state_dict``
        """

        # Load
        incompatible_keys = nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        # and update
        self._update_underlying_f()
        #
        return incompatible_keys

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

    def forward(self, x, y0=None, use_y0_conv2=True, batched_fixed_point=None, f_tau=None, abs_error_threshold=None):
        """
        Fixed-point result u* for u in the implicit function f(u,x;theta), that is, u*=f(u*,x;theta).
        If no initial point ``y0`` is provided a tensor filled with zeros will be used.

        The operation of the method depends on the state :py:attr:`.batched_fixed_point` of the class and of the \
        dimensionality of ``x``: if :py:attr:`.batched_fixed_point` is ``False`` then each (trailing) 3D element \
        of ``x`` is subject to an independent fixed-point problem; otherwise the $i$-th element of ``y0``, if ``y0``
        is 4D, or the only available ``y0`` if ``y0`` is 3D, is regarded the as initial point \
        for the $i$-th element of input ``x``.

        The forward operation depends on the attribute ``self._f_solver`` of the object:

        - With a value among ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``, ``'anderson'``, and ``'broyden'`` \
          the (forward) fixed point calculation is performed using the method :py:meth:`.FixedPointLayer.forward` \
          of the parent class :py:class:`.FixedPointLayer`.

        - With a value of ``'fpgd'`` (fixed point gradient descent) the (forward) fixed point calculation is performed \
          using the method :py:meth:`.FixedPointLayer.forward`, with the method ``'fixed_point_iter'``, \
          of the parent class :py:class:`.FixedPointLayer`.

        - With a value among ``'pgd'`` (projected gradient descent) \
          the calculation of the fixed point is calculated here and simply passed to \
          :py:meth:`.FixedPointLayer.forward` as an initial point to be left unchanged.

        **The function additionally calculates information regarding whether the error of the fixed point calculation,** \
        **measured as the norm of the error between f(x) and x, is above the threshold** ``abs_error_threshold``. \
        **Such info is not returned but stored for explicit query, if desired, through the method** \
        :py:meth:`.get_last_forward_convergence_info`, **which returns:**

            - the boolean Tensor indicating whether the returned fixed point  (for each input datapoint) was achieved \
              with satisfactory absolute error/convergence;,

            - the abs_error threshold used for the above assessment, 'self._last_forward_abs_error_threshold',

            - the batched-fixed-point flag used for the calculation, 'self._last_forward_batched_fixed_point'; and

            - the info (about trajectories) for the last calculations.

        Parameters
        ----------
        x : torch.Tensor
        y0 : torch.Tensor, optional
            The fixed point ``y0`` can be either a 3D tensor or a 4D tensor with as many elements as the tensor ``x``
        use_y0_conv2 : bool, optional
            If ``True`` the value of current $\\Phi(\\mathbf{M} \\ast \\mathbf{I}-\\mathbf{b})$ is used as \
            the initial point. If ``False``, a zero tensor is used as initial. Default: ``True``
        batched_fixed_point : bool, optional
            It overrides, if provided, the value currently in the corresponding attribute \
            :py:attr:`.batched_fixed_point` of the class, without changing said attribute. \
            If ``None`` the mode currently indicated by the attribute is used. \
            Default: ``None``
        f_tau : float, optional
            Float in the range $(0.0, 1.0]$; if provided (i.e. not ``None``), \
            and for the solver algorithms allowing so, it overrides the value $\\tau$ provided in the constructor (see
            :py:class:`.IBNNInternalLayer`) during the current evaluation. \
            If ``None`` the value already stored in the class is used.
            Default: ``None``
        abs_error_threshold : float, optional
            The absolute error threshold, measured as the norm of the error between f(x) and x, \
            for the fixed point calculation. \
            If ``None`` the mode currently indicated by the attribute is used.
            Default: ``None``

        Returns
        -------
        y_out : torch.Tensor
            The output of the layer, that is, the fixed point of the layer for the input `x` and
            the (current) parameters `theta` of the layer
        """
        if y0 is None:
            # Calculation of the initial point y0 if not provided, according to the requested settings 'use_y0_conv2':
            # if true then the output of the standard model is used as the initial solution.
            # The size of the zero initial output image is calculated from the size of 'x':
            if use_y0_conv2:
                theta_y0 = {}
                for key in ['m', 'b']:
                    theta_y0[key] = self._theta[key].detach().clone()
                pass

                y0 = f_modified_RF(
                    im=x.detach().clone(), theta=theta_y0, u=None, v=None,
                    phi_activation=self._phi_activation,
                    m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups
                )
            else:
                y0 = torch.zeros((self._theta['m'].size(-4), x.size(-2), x.size(-1))) if self._m_padding == 'same' \
                    else torch.zeros((self._theta['m'].size(-4),
                                      x.size(-2) - self._theta['m'].size(-2) + 1,
                                      x.size(-1) - self._theta['m'].size(-1) + 1))
            pass
        pass

        y_out = None

        ##########################################################################################################
        # Calculation of the forward pass and evaluation of the error, depending on the forward solver
        ##########################################################################################################

        if self._f_solver in ['fixed_point_iter', 'simple_fixed_point_iter', 'anderson', 'broyden'] or \
                self._f_solver == 'fpgd':
            ### Use of the forward point solvers implemented/integrated in FixedPointLayer
            # Fixed point gradient descent or projected gradient descent
            y_out =  FixedPointLayer.forward(
                self, x, y0=y0, batched_fixed_point=batched_fixed_point, f_tau=f_tau, abs_error_threshold=abs_error_threshold
            )
        elif self._f_solver == 'pgd':
            # Projected gradient descent, performed here and passed as initial point to the forward method of the
            # parent class FixedPointLayer

            # Parameters for the specific forward pass
            current_f_tau = self._f_tau if f_tau is None else f_tau
            self._last_forward_batched_fixed_point = \
                self._batched_fixed_point if batched_fixed_point is None else batched_fixed_point
            self._last_forward_abs_error_threshold = \
                self._abs_error_threshold if abs_error_threshold is None else abs_error_threshold

            # Detach 'x' and 'theta'
            theta_detached = self.theta_copy
            x_detached = x.detach().clone()

            # Check dimensional compatibility between x and y0: make both 4D. And create also y_n:
            #
            casted_images = False
            #
            if x_detached.ndim == 3:
                x_detached = x_detached.unsqueeze(0)
                casted_images = True
            elif x_detached.ndim != 4:
                raise Exception(f"Input 'x' not 3D or 4D!")
            pass
            #
            if y0.ndim == 3:
                y0 = y0.unsqueeze(0)
            elif y0.ndim != 4:
                raise Exception(f"Initial fixed point not 3D or 4D!")
            pass
            #
            # Expand 'y0' to the same size (regarding num. images in the batch) as 'x'
            if (y0.size(0) > 1) and (y0.size(0) != x.size(0)):
                raise Exception(f"Incompatible input 'x' {x.size()} and fixed point {y0.size()} sizes!")
            elif (y0.size(0) == 1) and (x.size(0) > 1):
                y0 = y0.expand((x.size(0), -1, -1, -1))
            pass

            # Define the function without the 'phi' activation and with 'theta' fixed but detached, and
            # define, separated, the activation
            def particularized_f_IBNN_without_phi_activation(u_f, x_f):
                return f_modified_RF(
                    x_f, theta_detached, u=u_f, v=None,
                    phi_activation='identity', sigma_activation=self._sigma_activation,
                    m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups,
                    w_padding_mode=self._w_padding_mode, w_groups=self._w_groups,
                    calculation_mode=self._calculation_mode, memory_saving_version=self._memory_saving_version,
                    **self._fixed_kwargs
                )
            # Useless line, only fit to set the device to use
            dummy_eval_particularized_f_IBNN_without_phi_activation = particularized_f_IBNN_without_phi_activation(
                y0, x_detached
            )

            phi_activation = nnl.ndim_activation_function_from_1dim_activation_functions(self._phi_activation)

            # Proceed with the iterations:
            # NOTE: different operation if batch size is used or not

            # We assess whether all the images in the batch achieved a reasonable absolute error, which
            # we will take as an indication of convergence
            self._last_forward_warning_above_threshold_y_out = None
            # We also store the 'info' structure of the last calculation: in this case that will be a tensor with
            # the norm of the residuals
            self._last_forward_info = torch.empty(
                (x.size(0), self._f_max_iter),
                device=dummy_eval_particularized_f_IBNN_without_phi_activation.device
            )
            self._last_forward_warning_above_threshold_y_out = torch.empty(
                (x.size(0),), dtype=torch.bool,
                device=dummy_eval_particularized_f_IBNN_without_phi_activation.device
            )

            y_n = y0.to(dummy_eval_particularized_f_IBNN_without_phi_activation.device)
            if self._last_forward_batched_fixed_point:  # Calculation as a batch
                # Initially, all images have a warning of non-convergence: we will update this as we iterate
                self._last_forward_warning_above_threshold_y_out[:] = True
                for i in range(self._f_max_iter):
                    # Update
                    y_n = phi_activation(
                        (1-current_f_tau) * y_n + \
                        current_f_tau * particularized_f_IBNN_without_phi_activation(y_n, x_detached)
                    )
                    # Fixed-point residual?
                    residual_n = y_n - self._f_function_of_y_x_theta(y_n, x_detached, theta_detached)
                    norm_each_residual_im = torch.norm(torch.flatten(residual_n, 1), p=2, dim=1)
                    if torch.all(norm_each_residual_im < self._f_tol):
                        self._last_forward_info[:, i:] = norm_each_residual_im.unsqueeze(1).expand(-1, self._f_max_iter-i)
                        break
                    else:
                        self._last_forward_info[:, i] = norm_each_residual_im[:]
                    pass
                pass
                # Finally assess if the absolute error is above the threshold of warning
                self._last_forward_warning_above_threshold_y_out[:] = \
                    norm_each_residual_im[:] >= self._last_forward_abs_error_threshold
            else:
                self._last_forward_warning_above_threshold_y_out[:] = False
                for b in range(x.size(0)):
                    norm_each_residual_im_b = 0.0
                    for i in range(self._f_max_iter):
                        y_n[b] = phi_activation(
                            (1-current_f_tau) * y_n[b] + \
                            current_f_tau * particularized_f_IBNN_without_phi_activation(y_n[b], x_detached[b])
                        )
                        # Fixed-point residual?
                        residual_n_b = y_n[b] - self._f_function_of_y_x_theta(y_n[b], x_detached[b], theta_detached)
                        norm_residual_im_b = torch.norm(residual_n_b).item()
                        if norm_each_residual_im_b < self._f_tol:
                            self._last_forward_info[b, i:] = norm_residual_im_b
                            break
                        else:
                            self._last_forward_info[b, i] = norm_residual_im_b
                        pass
                    pass
                    # Finally assess if the absolute error is above the threshold of warning
                    if norm_each_residual_im_b >= self._last_forward_abs_error_threshold:
                        self._last_forward_warning_above_threshold_y_out[b] = True
                pass
            pass

            # Return the output
            y_out_pgd = y_n

            # If the input x was 3D, i.e. (C, H, W), and was casted to 4D, i.e. (1, C, H, W), then we un-cast it \
            # and we do the same with 'above_threshold_y_out'
            if casted_images:
                y_out_pgd = y_out_pgd[0]
                self._last_forward_warning_above_threshold_y_out = self._last_forward_warning_above_threshold_y_out[0]
            pass

            # Still, now, we call the parent class to perform the fixed point calculation with this point as initial!
            y_out = FixedPointLayer.forward(
                self, x, y0=y_out_pgd, batched_fixed_point=self._last_forward_batched_fixed_point,
                f_tau=None, abs_error_threshold=self._last_forward_abs_error_threshold
            )

        pass

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




class INRFFamilyLayer(ModifiedRFLayer, nn.Module):
    """
    Layer implementing the common aspects of the different versions of the family \
    *Intrinsically Non-Linear Receptive Field* (INRF) neural networks.

    In general, the function implemented by all the versions of the family corresponds to the expression

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{\\Omega} \\ast
    \\mathbf{\\sigma} \\big( \\mathbf{V} - \\mathbf{U}[\\mathbf{p}] \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    However, each version of the family differs in the identity and meaning of the components $\\mathbf{\\Omega}$, \
    $\\mathbf{U}$ and $\\mathbf{V}$ of the above expression. To wit:

    .. list-table:: Schematic table of the differences between the different versions of INRF
        :widths: 15 15 15 15 15
        :header-rows: 1
        :stub-columns: 1

        *   -
            - INRFv1
            - INRFv2
            - INRFv3
            - ibnn_lite
        *   - $\\mathbf{\\Omega}$
            - $\\mathbf{M}$
            - $\\mathbf{\\Omega}$ ($\\neq \\mathbf{M}$)
            - $\\mathbf{\\Omega}$ ($\\neq \\mathbf{M}$)
            - $\\mathbf{\\Omega}$ ($\\neq \\mathbf{M}$)
        *   - $\\mathbf{U}$
            - $\\mathbf{I}$
            - $\\mathbf{I}$
            - $\\mathbf{M} \\ast \\mathbf{I} - \\mathbf{b}$
            - $\\mathbf{M} \\ast \\mathbf{I} - \\mathbf{b}$
        *   - $\\mathbf{V}$
            - $\\mathbf{I}$
            - $\\mathbf{I}$
            - $\\mathbf{I}$
            - $\\mathbf{M} \\ast \\mathbf{I} - \\mathbf{b}$

    The trainable parameters of the network are, thus, \
    $\\Theta = \\big(\\mathbf{M}, \\mathbf{b}, \\mathbf{\\Omega}, \\mathbf{\\lambda} \\big)$ where in the case \
    INRFv1 the parameter $\\mathbf{\\Omega}$ in fact tied to $\\mathbf{M}$. In all cases the functions are implemented \
    through the corresponding particularization of the function :py:func:`.f_modified_RF`. \
    For specific classes corresponding to each version refer, respectively, to \
    :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, and :py:class:`.IBNNLiteLayer`.

    **Argument restriction depending on the INRF version:**

    Note that, due to the differences of the components $\\mathbf{\\Omega}$, \
    $\\mathbf{U}$ and $\\mathbf{V}$ in the difference versions and the resulting sizes of the involved convolutions,
    each version poses different restrictions for the involved filters and their behaviors:

    - **v1** requires no restrictions;
    - **v2** **locks** ``m_padding`` **and** ``w_padding`` **to ``'same'``** to ensure size consistency; \
    it additionally requires that $\\mathbf{M}$ and $\\mathbf{\\Omega}$ share their number of output channels \
    and their presumed number of input channels (*groups* left aside);
    - **v3** **locks** ``m_padding`` **and** ``w_padding`` **to ``'same'``** to ensure size consistency; \
    it additionally requires that both $\\mathbf{M}$ and $\\mathbf{\\Omega}$ have as many output channels \
    as input channels (*groups* left aside) and both are coincident; and
    - **v4** **locks** ``w_padding`` **to ``'same'``**; it additionally requires \
    that $\\mathbf{M}$ and $\\mathbf{\\Omega}$ share their number of output channels and \
    that $\\mathbf{\\Omega}$ has as many input channels (*groups* left aside) as output channels.

    Input combinations failing to fulfil such requirements raise an exception.

    Input $\\mathbf{I}$ and output can differ in their respective number of channels: \
    the output of $\\mathbf{M} \\!\\ast\\! \\mathbf{I}$, given in fact by $\\mathbf{M}$,
    simply needs to be coincident with the number
    of channels of the output of $\\mathbf{\\Omega} \\!\\ast\\! \\mathbf{\\sigma}(\\cdot)$, \
    given in fact by $\\mathbf{\\Omega}$.

    The initialization of the parameters \
    $\\Theta = \\big(\\mathbf{M}, \\mathbf{b}, \\mathbf{\\Omega}, \\mathbf{\\lambda} \\big)$ of the network \
    will be performed deterministically according to the value of the respective arguments \
    ``m_initialization``, ``initial_b``, ``w_initialization``, ``initial_lambda``, \
    ``sigma_x_compress``, ``sigma_y_stretch``, ``sigma_x_offset``, and ``sigma_y_offset=0.0``: as far as the \
    initialization of the filter masks for $M$ and $\\Omega$ is concerned refer to the description of the \
    arguments of the constructors below. The randomization/random initialization required prior to the training \
    can be achieved using the method :py:meth:`~ModifiedRFLayer.random_initialization` of the parent class \
    :py:class:`ModifiedRFLayer`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    version : int
        Accepted versions: ``1``, ``2``, ``3``, and ``4`` (see above for the implications of each version)
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`, \
        see also :py:func:`.f_modified_RF`
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        Size of the kernel convolving the input image $\\mathbf{I}$. If tuple, $(H,W)$; if scalar S, $(S, S)$ \
        (see above for the implications of each version). If the input value(s) is (are) float 0<x<0 \
        the absolute pixel size of the kernel will be inferred from the argument ``in_size`` which, although optional, \
        becomes compulsory in such case
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. See :py:func:`.conv2d_adapted` for a detailed explanation.
        Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'`` \
        (see above for the implications of each version).
        See :py:func:`.conv2d_adapted` for a detailed explanation. Default: ``'zeros'``
    m_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Initialization mode for the filter mask $\\mathbf{M}$. If string, the provided value corresponds to the \
        argument ``initialization_type`` of the method :py:func:`.filter_initialization`; if dictionary, \
        the provided keys correspond to ``initialization_type`` and the corresponding keyword arguments \
        of the method :py:func:`.filter_initialization`; if a scalar (float or int) all elements of the filter \
        will be set to the provided value; and if tensor it will need to exactly match the size of requested filter
        size ``m_kernel_size``. \
        Default: ``'zeros'``
    m_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary, \
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    b_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        If ``b_type`` is ``'scalar_per_channel'`` and the provided value is a scalar, that value will correspond \
        to all the dimensions of the vector.
        Default: ``0.0``
    b_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    in_size : tuple[int], optional
        2D tuple (2D at least; trailing dimensions do not matter for the involved processing regarding only \
        spatial extent) indicating the size of the input image to the layer. This argument is not necessary if \
        the kernel sizes for *m* and *w* are provided as abslute int values; however it becomes necessary if \
        any of them is provided as relative (e.g. floats $0 \\leq x \\leq 1$) to the input size. \
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`, \
        see also :py:func:`.f_modified_RF`. \
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifier to the basic sigma activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifier to the basic sigma activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int orm float] or torch.Tensor, optional
        Offset modifiers to the basic activation function, potentially trainable: see \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions`. Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``False``
    lambda_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
        If ``lambda_type`` is ``'scalar_per_channel'`` and the provided value is a scalar, that value will correspond \
        to all the dimensions of the vector
    lambda_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        Size of the kernel convolving the activated difference of hidden signals. \
        If tuple, $(H,W)$; if scalar S, $(S, S)$. If list/tuple of length $0$ then \
        full-image, uniform product and summation is used. If the input value(s) is (are) float 0<x<0 \
        the absolute pixel size of the kernel will be inferred from the argument ``in_size`` which, although optional, \
        becomes compulsory in such case
    w_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'`` \
        (see above for the implications of each version).
        See :py:func:`.conv2d_adapted` for a detailed explanation. Default: ``'zeros'``
    w_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Initialization mode for the filter mask $\\mathbf{\\Omega}$. If string, the provided value corresponds to the \
        argument ``initialization_type`` of the method :py:func:`.filter_initialization`; if dictionary, \
        the provided keys correspond to ``initialization_type`` and the corresponding keyword arguments \
        of the method :py:func:`.filter_initialization`; if a scalar (float or int) all elements of the filter \
        will be set to the provided value; and if tensor it will need to exactly match the size of requested filter
        size ``w_kernel_size``. \
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps \
        (see above for the implications of each version).
        Default: ``False``
    kwargs : optional
        These keyword refer to the following arguments of the function \
        :py:func:`.f_modified_RF` (see therein for detailed information about the meaning of each parameter and which
        sub-function makes use of each one):

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'interpolated'``, ``'n4'``. Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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
    """

    def __init__(self, version,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=True,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 **kwargs):

        ###
        # Call the constructor "__init__" of the super class 'nn.Module' (the abstract class is not even defined):
        ###
        nn.Module.__init__(self)

        ###
        # Call the constructor "__init__" of the super class 'ModifiedRFLayer' taking care of the kwargs
        ###
        ModifiedRFLayer.__init__(self, **kwargs)

        ### Version
        if version in [1, 2, 3, 4]:
            self._inrf_version = version
        else:
            raise Exception(f"Introduced version {version} not allowed: only 1 to 4 accepted!")
        pass
        self._version = version

        #######################################
        # INITIAL CHECKS REGARDING COMPATIBILITIES OF CHANNEL NUMBERS AND IMAGE DIMENSIONS
        # REMEMBER: VERSION-DEPENDENT!
        #######################################

        ###
        # Scalar vs scalar-per-channel types: valid parameters?
        ###

        for type_param in [b_type, lambda_type]:
            if not type_param in ['scalar_per_channel', 'scalar']:
                raise Exception((f"Valid values for 'b_type' and 'lambda_type' " +
                                 f"are 'scalar_per_channel', 'scalar': '{type_param}' provided!"))
            pass
        pass

        ###
        # Initial checks regarding compatibility of channel number vs groups vs image dimensions
        ###

        # These structures are version-dependent, we will go one-by-one filling them
        # and making some additional checks

        ###
        # Check that 'initial_lambda' and 'w_kernel_size', compulsory but defaulting None for practical reasons,
        # are not None!
        ###

        if initial_lambda is None:
            raise Exception(f"Argument 'initial_lambda' (different from None) is compulsory!")
        pass

        if (self._inrf_version == 1) and (w_kernel_size is not None):
            raise Exception((f"For INRF version {self._inrf_version} 'w_kernel_size' must NOT BE provided: " +
                             f"{w_kernel_size} found!"))
        elif (self._inrf_version != 1) and (w_kernel_size is None):
            raise Exception((f"For INRF version {self._inrf_version} 'w_kernel_size' must BE provided: " +
                             f"{w_kernel_size} found!"))
        pass

        ###
        # Initial checks regarding compatibility of channel number vs groups vs image dimensions
        ###

        w_padding = None
        dict_kernel_size = {}
        dict_groups = {}
        dict_kernel_out_in_channels = {}
        #
        if self._inrf_version == 1:
            # No restrictions: same filter in both sides
            # In/out channels?
            dict_kernel_out_in_channels['m'] = (out_channels, in_channels)
            dict_kernel_out_in_channels['w'] = dict_kernel_out_in_channels['m']
            dict_groups['m'] = m_groups
            dict_groups['w'] = dict_groups['m']
            w_padding = m_padding
            # Only checks: no 'w_kernel_size' is provided
            dict_kernel_size['m'] = m_kernel_size
            dict_kernel_size['w'] = dict_kernel_size['m']
        elif self._inrf_version == 2:
            # In/out channels?
            dict_kernel_out_in_channels['m'] = (out_channels, in_channels)
            dict_kernel_out_in_channels['w'] = (out_channels, in_channels)
            dict_groups['m'] = m_groups
            dict_groups['w'] = w_groups
            # Spatial padding?
            w_padding = 'same'
            if (m_padding != 'same') or (w_padding != 'same'):
                raise Exception(
                    (f"For INRF version {self._inrf_version} both 'm_padding' and 'w_padding' must be 'same'; however" +
                     f"'m_padding' = {m_padding} and 'w_padding' = {w_padding} have been found.")
                )
            pass
            dict_kernel_size['m'] = m_kernel_size
            dict_kernel_size['w'] = w_kernel_size
        elif self._inrf_version == 3:
            # In/out channels?
            dict_kernel_out_in_channels['m'] = (out_channels, in_channels)
            dict_kernel_out_in_channels['w'] = (out_channels, in_channels)
            dict_groups['m'] = m_groups
            dict_groups['w'] = w_groups
            if dict_kernel_out_in_channels['m'][0] != dict_kernel_out_in_channels['m'][1]:
                raise Exception(
                    (f"For INRF version {self._inrf_version} " +
                     f"'in_channels' = {dict_kernel_out_in_channels['m'][1]} " +
                     f"and 'out_channels' = {dict_kernel_out_in_channels['m'][0]}  must be coincident due to " +
                     "the cross-calculation between M*I and I and between M*I and W*(M*I-I)!")
                )
            pass
            # Spatial padding?
            w_padding = 'same'
            if (m_padding != 'same') or (w_padding != 'same'):
                raise Exception(
                    (f"For INRF version {self._inrf_version} both 'm_padding' and 'w_padding' must be 'same'; however" +
                     f"'m_padding' = {m_padding} and 'w_padding' = {w_padding} have been found.")
                )
            pass
            dict_kernel_size['m'] = m_kernel_size
            dict_kernel_size['w'] = w_kernel_size
        else:  # if self._inrf_version == 4
            # In/out channels?
            dict_kernel_out_in_channels['m'] = (out_channels, in_channels)
            dict_kernel_out_in_channels['w'] = (out_channels, out_channels)
            dict_groups['m'] = m_groups
            dict_groups['w'] = w_groups
            # Spatial padding?
            w_padding = 'same'
            dict_kernel_size['m'] = m_kernel_size
            dict_kernel_size['w'] = w_kernel_size
        pass
        #
        for key in dict_kernel_size:
            #
            if dict_kernel_size[key] is None:
                raise Exception(
                    (f"No kernel size for '{key}', or 'None', is provided! (Reminder: " +
                     f"full-image, uniform kernels are noted as '[]' or '()'!)")
                )
            pass
            #
            # Reformat to help checks (not resolving yet for relative filters)
            dict_kernel_size[key] = nnl.kernel_size_check_and_reformat_into_tuple(dict_kernel_size[key])
            #
            if len(dict_kernel_size[key]) == 0 and key == 'm':
                raise Exception(f"Full-image, uniform kernels requested for '{key}: only valid for 'w'!")
            pass
            #
            if len(dict_kernel_size[key]) == 0 or isinstance(dict_kernel_size[key][0], float):
                if in_size is None:
                    raise Exception((f"Relative kernel sizes, particularly {dict_kernel_size[key]}, " +
                                     f"are provided for '{key}', but 'in_size' is not provided!"))
                elif len(in_size) < 2:
                    raise Exception((f"Relative kernel sizes are provided for '{key}', " +
                                     f"but 'in_size' is not (or does not contain) a 2D tuple!"))
                pass
            pass
            #
            # And resolve the (spatial) kernel size if necessary
            dict_kernel_size[key] = nnl.resolve_kernel_size_for_im_size(
                dict_kernel_size[key], im_size=in_size, make_odd=False
            )
        pass

        for key in ['m', 'w']:
            if (dict_kernel_out_in_channels[key][0] % dict_groups[key] != 0) or \
                    (dict_kernel_out_in_channels[key][1] % dict_groups[key] != 0):
                raise Exception((f"Both 'in_channels'={dict_kernel_out_in_channels[key][1]} and " +
                                 f"'out_channels'={dict_kernel_out_in_channels[key][0]} of '{key}' " +
                                 f"must be divisible by '{key}_groups'={dict_groups[key]}!"))
        pass

        #######################################
        # STORE PARAMETERS THAT WILL BE RELEVANT LATER ON AS ATTRIBUTES
        #######################################

        # self._phi_activation_f = nnl.ndim_activation_function_from_1dim_activation_functions(phi_activation)
        self._phi_activation = phi_activation

        self._m_kernel_size = dict_kernel_size['m']
        self._m_groups = dict_groups['m']
        self._m_padding = m_padding
        self._m_padding_mode = m_padding_mode

        self._m_initialization = m_initialization if isinstance(m_initialization, str) else 'custom'

        self._b_type = b_type

        self._w_kernel_size = dict_kernel_size['w']
        self._w_groups = dict_groups['w']
        self._w_padding = w_padding
        self._w_padding_mode = w_padding_mode

        self._w_initialization = w_initialization if isinstance(w_initialization, str) else 'custom'

        self._lambda_type = lambda_type

        # self._sigma_activation_f = nnl.ndim_activation_function_from_1dim_activation_functions(sigma_activation)
        self._sigma_activation = sigma_activation

        self._calculation_mode = kwargs.pop('calculation_mode', None)
        self._memory_saving_version = kwargs.pop('memory_saving_version', None)

        #######################################
        # Define an initial "_theta0" (kept as record) and transfer it to the dynamically optimized "_theta"
        ###
        # "_theta" is initialized deterministically as "_theta0" according to some desired rules. Later, and before
        # training, random disturbances can be introduced with the appropriate method of the class
        #######################################

        # Initial and "dynamic" parameter set
        self._theta0 = {}

        # Deterministic initialization of "_theta0" according to the provided arguments

        # For 'm' and 'w': We have "dict_groups", "dict_kernel_out_in_channels", and "dict_kernel_size" from before,
        # we define also...
        dict_initialization = {'m': m_initialization,
                               'w': w_initialization}
        for key in ['m', 'w']:
            if (self._inrf_version == 1) and (key == 'w'):
                self._theta0['w'] = self._theta0['m']
            else:
                tensor_dims = (dict_kernel_out_in_channels[key][0],
                               int(dict_kernel_out_in_channels[key][1] / dict_groups[key])) + dict_kernel_size[key]
                if isinstance(dict_initialization[key], str):
                    self._theta0[key] = nnl.filter_initialization(
                        tensor_dims, initialization_type=dict_initialization[key],
                        im_size=in_size, groups=dict_groups[key],
                        dtype=torch.float32
                    )
                elif isinstance(dict_initialization[key], dict):
                    self._theta0[key] = nnl.filter_initialization(
                        tensor_dims,
                        im_size=in_size, groups=dict_groups[key],
                        **dict_initialization[key], dtype=torch.float32
                    )
                elif isinstance(dict_initialization[key], (int, float)):
                    self._theta0[key] = torch.empty(tensor_dims)
                    self._theta0[key][:] = dict_initialization[key]
                elif isinstance(dict_initialization[key], torch.Tensor):
                    if dict_initialization[key].size() != tensor_dims:
                        raise Exception((
                                f"The size of the provided tensor for '{key}', {dict_initialization[key].size()}, " +
                                f"is not compatible with the requested size {tensor_dims}")
                        )
                    self._theta0[key] = torch.empty(tensor_dims)
                    self._theta0[key][:] = dict_initialization[key]
                    pass
                pass
            pass
        pass

        # For 'b' and 'lambda':
        dict_type = {'b': b_type,
                     'lambda': lambda_type}
        dict_initial_value = {'b': initial_b,
                              'lambda': initial_lambda}
        for key in ['b', 'lambda']:
            #
            if dict_type[key] == 'scalar_per_channel':
                self._theta0[key] = torch.empty(out_channels, 1, 1)
            else:  # 'scalar'
                self._theta0[key] = torch.empty(1)
            pass
            #
            if isinstance(dict_initial_value[key], int) or isinstance(dict_initial_value[key], float):
                self._theta0[key][:] = dict_initial_value[key]
            elif isinstance(dict_initial_value[key], torch.Tensor):
                self._theta0[key][:] = dict_initial_value[key][:]
            else:
                raise Exception((f"The selected initialization value for '{key}' is not an accepted type " +
                                 f"float, int, or torch.Tensor," +
                                 f"but the following: {type(dict_initial_value[key])}"))
            pass
            #
        pass

        # For the sigma modifiers:
        dict_sigma_modifier_values = {
            'sigma_x_compress': sigma_x_compress, 'sigma_y_stretch': sigma_y_stretch,
            'sigma_x_offset': sigma_x_offset, 'sigma_y_offset': sigma_y_offset
        }
        for key in dict_sigma_modifier_values:
            if isinstance(dict_sigma_modifier_values[key], int) or isinstance(dict_sigma_modifier_values[key], float):
                self._theta0[key] = torch.Tensor([dict_sigma_modifier_values[key]])
            elif isinstance(dict_sigma_modifier_values[key], torch.Tensor) or \
                    isinstance(dict_sigma_modifier_values[key], list) or \
                    isinstance(dict_sigma_modifier_values[key], tuple):
                self._theta0[key] = torch.Tensor(dict_sigma_modifier_values[key])
            else:
                raise Exception((f"The selected initialization value for '{key}' is not an accepted type " +
                                 f"float, int, torch.Tensor, list, or tuple" +
                                 f"but the following: {type(dict_sigma_modifier_values[key])}"))
        pass

        #######################################
        # CREATE THE (HIDDEN) ATTRIBUTE "_theta_trainable",
        # WHICH IS A DICTIONARY OF BOOLS MIRRORING THE FIELDS OF "_theta"
        #######################################

        self._theta_trainable = {}
        for key in self._theta0:
            self._theta_trainable[key] = False
        pass
        self._theta_trainable['m'] = m_trainable
        self._theta_trainable['b'] = b_trainable
        if (self._inrf_version == 1):
            if w_trainable != self._theta_trainable['m']:
                warnings.warn((
                        f"For INRFv1 'theta['w']' mirrors 'theta['m']': 'theta['w']' will be set as " +
                        f"trainable={self._theta_trainable['m']} like 'theta['m']' is, " +
                        f"contrary to the selection for 'w_trainable'."))
            pass
            self._theta_trainable['w'] = self._theta_trainable['m']
        else:
            self._theta_trainable['w'] = w_trainable
        pass
        self._theta_trainable['lambda'] = lambda_trainable
        self._theta_trainable['sigma_x_compress'] = sigma_x_compress_trainable
        self._theta_trainable['sigma_y_stretch'] = sigma_y_stretch_trainable
        self._theta_trainable['sigma_x_offset'] = sigma_x_offset_trainable
        self._theta_trainable['sigma_y_offset'] = sigma_y_offset_trainable

        #######################################
        # FINALLY WE CREATE THE DYNAMIC "_theta" with their ".requires_grad" according to "_theta_trainable"
        #######################################

        self._theta = nn.ParameterDict()
        for key in self._theta0:
            self._theta[key] = nn.Parameter(self._theta0[key].clone(), requires_grad=self._theta_trainable[key])
        pass

        #######################################
        # Make all versions consistent by creating corresponding expressions for U and V
        #######################################

        self._update_underlying_f()

    pass

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(INRFFamilyLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass

    def _update_underlying_f(self):

        def f_id(im_f, theta_f):
            return im_f

        pass

        def f_mi_b(im_f, theta_f):
            return f_modified_RF(
                im_f, theta_f,
                phi_activation='identity',
                m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups
            )

        pass

        f_U_inrf_version_i = f_id
        f_V_inrf_version_i = lambda x1, x2: None
        if self._version == 3:
            f_U_inrf_version_i = f_mi_b
            f_V_inrf_version_i = f_id
        elif self._version == 4:
            f_U_inrf_version_i = f_mi_b
            f_V_inrf_version_i = lambda x1, x2: None
        elif self._version not in [1, 2]:
            raise Exception(f"Introduced version {self._version} not allowed: only 1 to 4 accepted!")
        pass

        def f_INRF_version_i(im_f, theta_f):
            return f_modified_RF(
                im_f, theta_f, u=f_U_inrf_version_i(im_f, theta_f), v=f_V_inrf_version_i(im_f, theta_f),
                phi_activation=self._phi_activation, sigma_activation=self._sigma_activation,
                m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups,
                w_padding_mode=self._w_padding_mode, w_groups=self._w_groups,
                calculation_mode=self._calculation_mode, memory_saving_version=self._memory_saving_version,
                **self._fixed_kwargs
            )

        self._f_INRF_version_i = f_INRF_version_i

    pass

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Copy parameters and buffers from state_dict into this module and its descendants. It overloads the function \
        of the same name (:py:meth:`~torch.nn.Module.load_state_dict`) in :py:class:`torch.nn.Module` to \
        update the underlying function of the class using the newly loaded parameters using the method \
        :py:meth:`.set_filter_mask_in_theta`.

        As in :py:meth:`~torch.nn.Module.load_state_dict`, if strict is True, \
        then the keys of state_dict must exactly match the keys returned by \
        :py:meth:`~torch.nn.Module.state_dict`.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing parameters and persistent buffers
        strict : bool, optional
            Whether to strictly enforce that the keys in ``state_dict`` match the keys returned by this module's \
            :py:meth:`~torch.nn.Module.state_dict`. Default: ``True``
        assign : bool, optional
            When ``False``, the properties of the tensors in the current module are preserved while when ``True``, \
            the properties of the Tensors in the state dict are preserved. The only exception is the \
            ``requires_grad`` field. Default: ``False``

        Returns
        -------
        missing_keys : NamedTuple
            Keys (str) that are expected by this module but missing from the provided ``state_dict``
        unexpected_keys : NamedTuple
            Keys (str) that are not expected by this module but present in the provided ``state_dict``
        """

        # Load
        incompatible_keys = nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        # and update
        self._update_underlying_f()
        #
        return incompatible_keys

    @property
    def version(self):
        """
        Obtain the version of the current INRFFamilyLayer object.

        Returns
        -------
        int
        """
        return self._inrf_version

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            The output of the layer.
        """

        ################
        # Check whether operations should be pushed into GPU
        ################
        computation_device = 'cpu'
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # If here, 'cuda' is available.
            # Check if 'theta' is defined so, and raise error if any key in 'theta' disagrees with the rest
            for key in self._theta:
                # print(f"'Device of 'theta['{key}']' = {self._theta[key].device.type}")
                if self._theta[key].device.type == 'cpu' and computation_device == 'cuda':
                    raise Exception(
                        f"Certain keys of 'theta' (e.g. '{key}') are 'cpu' while others 'cuda': inconsistent!")
                else:
                    computation_device = self._theta[key].device.type
                pass
            pass
        pass
        # print(f"'Device after checks = {computation_device}")
        ################
        if x.device.type != computation_device:
            x = x.to(computation_device)
        pass

        return self._f_INRF_version_i(x, self._theta)




######################################################
# ibnn_internal Layer
######################################################


class IBNNLayer(IBNNInternalLayer):
    """
    Layer implementing the *Implicitly-Biased Neural Network* (ibnn_internal) layer having the activation but with the \
    application of the (external) activation function **after** the fixed point calculation: that is the reason \
    why it is called $\\textrm{ibnn_internal}\\textbf{X}$, for "$\\textrm{ibnn_internal with e}\\textbf{X}\\textrm{ternal activation}$". \
    The layer is using :py:class:`.IBNNInternalLayer` with an identity function as \
    its `phi_activation` and a modified forward pass where the selected `phi_activation` is applied to its result. \
    All the arguments of the constructor of this class :py:class:`.IBNNInternalLayer` and its method correspond \
    totally to those of its parent class :py:class:`.IBNNInternalLayer`.

    Regarding its arguments: see :py:class:`.IBNNInternalLayer`!
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=True,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 batched_fixed_point=True,
                 f_solver='fixed_point_iter', f_max_iter=50, f_tol=1e-5, f_tau=0.1,
                 b_solver='fixed_point_iter', b_max_iter=40, b_tol=1e-6,
                 abs_error_threshold=1e-5,
                 **kwargs):

        #######################################
        # It constructs and object of the parent class... but without phi
        #######################################

        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            phi_activation='identity',  # Identity function as phi activation
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            w_kernel_size=w_kernel_size, w_padding_mode=w_padding_mode, w_groups=w_groups,
            w_initialization=w_initialization,
            w_trainable=w_trainable,
            batched_fixed_point=batched_fixed_point,
            f_solver=f_solver, f_max_iter=f_max_iter, f_tol=f_tol, f_tau=f_tau,
            b_solver=b_solver, b_max_iter=b_max_iter, b_tol=b_tol,
            abs_error_threshold=abs_error_threshold,
            **kwargs)

        # And externaly creates the activation function to apply after the fixed point calculation
        self._extra_phi_activation = phi_activation

        # And... we simply need to overload the methods of the parent class relevant to the activation function:
        # in particular, and MAINLY, we need to adapt the method 'forward' to apply the activation function.
    pass

    def forward(self, x, y0=None, use_y0_conv2=True, batched_fixed_point=None, f_tau=None, abs_error_threshold=None):
        """
        Analogous to the method :py:meth:`.IBNNInternalLayer.forward`.
        """
        #
        # The output for the identity:
        y_out = super().forward(
            x=x, y0=y0, use_y0_conv2=use_y0_conv2,
            batched_fixed_point=batched_fixed_point, f_tau=f_tau, abs_error_threshold=abs_error_threshold
        )
        #
        # And we apply the activation function
        phi_activation_fun = nnl.ndim_activation_function_from_1dim_activation_functions(self._extra_phi_activation)
        #
        return phi_activation_fun(y_out)
    pass

    @property
    def phi_activation(self):
        """
        Obtain the current ``phi_activation`` of the class.

        Returns
        -------
        ~collections.abc.Callable or list[~collections.abc.Callable]
        """
        return self._extra_phi_activation
    pass

    def dict_fields_to_log_at_current_time(self):
        """
        Analogous role to the method :py:meth:`.IBNNInternalLayer.dict_fields_to_log_at_current_time`.
        """
        # Take the fields to log from the parent class
        dict_fields_to_log = super().dict_fields_to_log_at_current_time()
        dict_fields_to_log['phi_activation'] = self._extra_phi_activation
        #
        return dict_fields_to_log
    pass

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Analogous to the method :py:meth:`.IBNNInternalLayer.constructor_default_values`.
        """

        return ModifiedRFLayer._constructor_default_values_given_class(IBNNLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)
    pass




class IBNNLiteLayer(INRFFamilyLayer):
    """
    Layer implementing a simplified version of the IBNN layer (:py:class:`.IBNNLayer) consisting in a single \
    calculation of the fixed point iteration required by the complete IBNN layer so its operation is made, \
    consequently, purely forward.

    The layer corresponds to the operation

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{\\Omega} \\ast
    \\mathbf{\\sigma} \\big(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I} \\!-\\! \\mathbf{b}) -
    (\\mathbf{M} \\! \\ast \\!\\mathbf{I} \\!-\\! \\mathbf{b})[\\mathbf{p}]
    \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits completely from :py:class:`.INRFFamilyLayer`, sharing all methods and attributes with it. \
    :py:class:`.IBNNLiteLayer` can in fact be seen as its mere alias of :py:class:`.INRFFamilyLayer` with \
    ``version`` = ``4`` and comprising uniquely the constructor arguments relevant for the former: therefore, for \
    a detailed description of argument, attribute, and method description, refer to :py:class:`.INRFFamilyLayer`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    m_padding : str, optional
        Default: ``‘same’``
    m_padding_mode : str, optional
        Default: ``'zeros'``
    m_groups : int, optional
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``'zeros'``
    m_trainable : bool, optional
        Default: ``True``
    b_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        Default: ``0.0``
    b_trainable : bool, optional
        Default: ``True``
    in_size : tuple[int], optional
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Default: ``False``
    lambda_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
    lambda_trainable : bool, optional
        Default: ``True``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    w_padding_mode : str, optional
        Default: ``'zeros'``
    w_groups : int, optional
        Default: ``1``
    w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Default: ``False``
    **kwargs : optional

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=True,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 **kwargs):
        #
        version = 4
        #
        ###
        # Call the constructor "super.__init__" of the super class:
        ###
        super().__init__(
            version=version,
            in_channels=in_channels, out_channels=out_channels,
            phi_activation=phi_activation,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            w_kernel_size=w_kernel_size, w_padding_mode=w_padding_mode,
            w_groups=w_groups, w_initialization=w_initialization, w_trainable=w_trainable,
            **kwargs
        )

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(IBNNLiteLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass




class INRFv1Layer(INRFFamilyLayer):
    """
    Layer implementing the *version v1* of the
    *Intrinsically Non-Linear Receptive Field* (INRF) neural network layer, that is,

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{M} \\ast
    \\mathbf{\\sigma} \\big( \\mathbf{I} - \\mathbf{I}[\\mathbf{p}] \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits completely from :py:class:`.INRFFamilyLayer`, sharing all methods and attributes with it. \
    :py:class:`.INRFv1Layer` can in fact be seen as its mere alias of :py:class:`.INRFFamilyLayer` with \
    ``version`` = ``1`` and comprising uniquely the constructor arguments relevant for the former: therefore, for \
    a detailed description of argument, attribute, and method description, refer to :py:class:`.INRFFamilyLayer`.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    m_padding : str, optional
        Default: ``‘same’``
    m_padding_mode : str, optional
        Default: ``'zeros'``
    m_groups : int, optional
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``'zeros'``
    m_trainable : bool, optional
        Default: ``True``
    b_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        Default: ``0.0``
    b_trainable : bool, optional
        Default: ``True``
    in_size : tuple[int], optional
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Default: ``False``
    lambda_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
    lambda_trainable : bool, optional
        Default: ``False``
    **kwargs : optional

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=False,
                 **kwargs):
        #
        version = 1
        #
        ###
        # Call the constructor "super.__init__" of the super class:
        ###
        super().__init__(
            version=version,
            in_channels=in_channels, out_channels=out_channels,
            phi_activation=phi_activation,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            **kwargs
        )

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(INRFv1Layer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass


class INRFv2Layer(INRFFamilyLayer):
    """
    Layer implementing the *version v2* of the
    *Intrinsically Non-Linear Receptive Field* (INRF) neural network layer, that is,

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{\\Omega} \\ast
    \\mathbf{\\sigma} \\big( \\mathbf{I} - \\mathbf{I}[\\mathbf{p}] \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits completely from :py:class:`.INRFFamilyLayer`, sharing all methods and attributes with it. \
    :py:class:`.INRFv2Layer` can in fact be seen as its mere alias of :py:class:`.INRFFamilyLayer` with \
    ``version`` = ``2`` and comprising uniquely the constructor arguments relevant for the former: therefore, for \
    a detailed description of argument, attribute, and method description, refer to :py:class:`.INRFFamilyLayer`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    m_padding_mode : str, optional
        Default: ``'zeros'``
    m_groups : int, optional
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``'zeros'``
    m_trainable : bool, optional
        Default: ``True``
    b_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        Default: ``0.0``
    b_trainable : bool, optional
        Default: ``True``
    in_size : tuple[int], optional
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Default: ``False``
    lambda_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
    lambda_trainable : bool, optional
        Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    w_padding_mode : str, optional
        Default: ``'zeros'``
    w_groups : int, optional
        Default: ``1``
    w_initialization : w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Default: ``False``
    **kwargs : optional

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=False,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 **kwargs):
        #
        version = 2
        #
        m_padding = 'same'
        #
        ###
        # Call the constructor "super.__init__" of the super class:
        ###
        super().__init__(
            version=version,
            in_channels=in_channels, out_channels=out_channels,
            phi_activation=phi_activation,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            w_kernel_size=w_kernel_size, w_padding_mode=w_padding_mode,
            w_groups=w_groups, w_initialization=w_initialization, w_trainable=w_trainable,
            **kwargs
        )

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(INRFv2Layer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass


class INRFv3Layer(INRFFamilyLayer):
    """
    Layer implementing the *version v3* of the
    *Intrinsically Non-Linear Receptive Field* (INRF) neural network layer, that is,

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{\\Omega} \\ast
    \\mathbf{\\sigma} \\big(
    \\,\\, \\mathbf{I} -
    (\\mathbf{M} \\! \\ast \\!\\mathbf{I} \\!-\\! \\mathbf{b})[\\mathbf{p}]
    \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits completely from :py:class:`.INRFFamilyLayer`, sharing all methods and attributes with it. \
    :py:class:`.INRFv3Layer` can in fact be seen as its mere alias of :py:class:`.INRFFamilyLayer` with \
    ``version`` = ``3`` and comprising uniquely the constructor arguments relevant for the former: therefore, for \
    a detailed description of argument, attribute, and method description, refer to :py:class:`.INRFFamilyLayer`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    m_padding_mode : str, optional
        Default: ``'zeros'``
    m_groups : int, optional
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``'zeros'``
    m_trainable : bool, optional
        Default: ``True``
    b_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        Default: ``0.0``
    b_trainable : bool, optional
        Default: ``True``
    in_size : tuple[int], optional
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Default: ``False``
    lambda_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
    lambda_trainable : bool, optional
        Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    w_padding_mode : str, optional
        Default: ``'zeros'``
    w_groups : int, optional
        Default: ``1``
    w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Default: ``False``
    **kwargs : optional

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=False,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 **kwargs):
        #
        version = 3
        #
        m_padding = 'same'
        #
        ###
        # Call the constructor "super.__init__" of the super class:
        ###
        super().__init__(
            version=version,
            in_channels=in_channels, out_channels=out_channels,
            phi_activation=phi_activation,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            w_kernel_size=w_kernel_size, w_padding_mode=w_padding_mode,
            w_groups=w_groups, w_initialization=w_initialization, w_trainable=w_trainable,
            **kwargs
        )

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(INRFv3Layer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass

class INRFv4(INRFFamilyLayer):
    """
    Layer implementing the *version v4* of the
    *Intrinsically Non-Linear Receptive Field* (INRF) neural network layer, that is,

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\bigg(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    - \\mathbf{\\lambda}
    \\Big(
    \\mathbf{\\Omega} \\ast
    \\mathbf{\\sigma} \\big(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I} \\!-\\! \\mathbf{b}) -
    (\\mathbf{M} \\! \\ast \\!\\mathbf{I} \\!-\\! \\mathbf{b})[\\mathbf{p}]
    \\big)
    \\Big) \\!(\\mathbf{p})
    \\bigg)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits completely from :py:class:`.INRFFamilyLayer`, sharing all methods and attributes with it. \
    :py:class:`.IBNNLiteLayer` can in fact be seen as its mere alias of :py:class:`.INRFFamilyLayer` with \
    ``version`` = ``4`` and comprising uniquely the constructor arguments relevant for the former: therefore, for \
    a detailed description of argument, attribute, and method description, refer to :py:class:`.INRFFamilyLayer`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    m_padding : str, optional
        Default: ``‘same’``
    m_padding_mode : str, optional
        Default: ``'zeros'``
    m_groups : int, optional
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``'zeros'``
    m_trainable : bool, optional
        Default: ``True``
    b_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        Default: ``0.0``
    b_trainable : bool, optional
        Default: ``True``
    in_size : tuple[int], optional
        Default: ``None``
    sigma_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Default: ``'tanh'``
    sigma_x_compress : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``10.0``
    sigma_y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``1.0``
    sigma_x_offset, sigma_y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Default: ``0.0``
    sigma_x_compress_trainable, sigma_y_stretch_trainable, \
        sigma_x_offset_trainable, sigma_y_offset_trainable : bool, optional
        Default: ``False``
    lambda_type : str, optional
        Default: ``'scalar_per_channel'``
    initial_lambda : int or float or torch.Tensor
    lambda_trainable : bool, optional
        Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
    w_padding_mode : str, optional
        Default: ``'zeros'``
    w_groups : int, optional
        Default: ``1``
    w_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Default: ``{'initialization_type': 'ones', 'normalization': 'group'}``
    w_trainable : bool, optional
        Default: ``False``
    **kwargs : optional

        - **calculation_mode** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Default: ``'interpolated'``

        - **memory_saving_version** : bool, optional

            Default: see :py:func:`.conv2d_crossdiff`

        - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

            If ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; ``True``, non-uniform sampling. \
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

            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **interpolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. \
            Default: see :py:func:`.generate_builder_1_order_interpolation`

        - **extrapolation_type** : `str <https://docs.python.org/3/library/stdtypes.html#str>`_, optional

            Value among ``'closest'``. Default: see :py:func:`.generate_builder_1_order_interpolation`
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None,
                 sigma_activation='tanh',
                 sigma_x_compress=10.0, sigma_y_stretch=1.0, sigma_x_offset=0.0, sigma_y_offset=0.0,
                 sigma_x_compress_trainable=False, sigma_y_stretch_trainable=False,
                 sigma_x_offset_trainable=False, sigma_y_offset_trainable=False,
                 lambda_type='scalar_per_channel', initial_lambda=None, lambda_trainable=False,
                 w_kernel_size=None, w_padding_mode='zeros', w_groups=1,
                 w_initialization={'initialization_type': 'ones', 'normalization': 'group'},
                 w_trainable=False,
                 **kwargs):
        #
        version = 4
        #
        ###
        # Call the constructor "super.__init__" of the super class:
        ###
        super().__init__(
            version=version,
            in_channels=in_channels, out_channels=out_channels,
            phi_activation=phi_activation,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_padding_mode=m_padding_mode, m_groups=m_groups,
            m_initialization=m_initialization, m_trainable=m_trainable,
            b_type=b_type, initial_b=initial_b, b_trainable=b_trainable,
            in_size=in_size,
            sigma_activation=sigma_activation,
            sigma_x_compress=sigma_x_compress, sigma_y_stretch=sigma_y_stretch,
            sigma_x_offset=sigma_x_offset, sigma_y_offset=sigma_y_offset,
            sigma_x_compress_trainable=sigma_x_compress_trainable, sigma_y_stretch_trainable=sigma_y_stretch_trainable,
            sigma_x_offset_trainable=sigma_x_offset_trainable, sigma_y_offset_trainable=sigma_y_offset_trainable,
            lambda_type=lambda_type, initial_lambda=initial_lambda, lambda_trainable=lambda_trainable,
            w_kernel_size=w_kernel_size, w_padding_mode=w_padding_mode,
            w_groups=w_groups, w_initialization=w_initialization, w_trainable=w_trainable,
            **kwargs
        )

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(IBNNLiteLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass

class SMLayer(ModifiedRFLayer, nn.Module):
    """
    Layer implementing the Standard Model neural network layer, that is,

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{A} (\\mathbf{I}))(\\mathbf{p})
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
    $$

    which accepts general affine transform $\\mathbf{A} (\\mathbf{I}))$ in the form \
    of a fully-connected plus bias but also convolutions: in that case the layer operation can be written as

    $$\\mathbf{Y}(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{M} \\!\\ast\\! \\mathbf{I})(\\mathbf{p})
    \\!-\\! \\mathbf{b}
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
    $$

    The class inherits  from :py:class:`.ModifiedRFLayer`: for \
    a detailed description of argument, attribute, and method description, refer to it.

    The initialization of the parameters \
    $\\Theta = \\big(\\mathbf{M}, \\mathbf{b} \\big)$ of the network \
    will be performed deterministically according to the value of the respective arguments \
    ``m_initialization`` and ``initial_b``: as far as the \
    initialization of the filter masks for $M$ and $\\Omega$ is concerned refer to the description of the \
    arguments of the constructors below. The randomization/random initialization required prior to the training \
    can be achieved using the method :py:meth:`~ModifiedRFLayer.random_initialization` of the parent class \
    :py:class:`ModifiedRFLayer`.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    phi_activation : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`, \
        see also :py:func:`.f_modified_RF`
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        Size of the kernel convolving the input image $\\mathbf{I}$. If tuple, $(H,W)$; if scalar S, $(S, S)$ \
        (see above for the implications of each version). If the input value(s) is (are) float 0<x<0 \
        the absolute pixel size of the kernel will be inferred from the argument ``in_size`` which, although optional, \
        becomes compulsory in such case
    m_padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``. See :py:func:`.conv2d_adapted` for a detailed explanation.
        Default: ``‘same’``
    m_padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'`` \
        (see above for the implications of each version).
        See :py:func:`.conv2d_adapted` for a detailed explanation. Default: ``'zeros'``
    m_groups : int, optional
        For the meaning of groups see :py:class:`torch.nn.Conv2d` \
        (see above for the implications of each version).
        Default: ``1``
    m_initialization : str or dict[str] or torch.Tensor or int or float, optional
        Initialization mode for the filter mask $\\mathbf{M}$. If string, the provided value corresponds to the \
        argument ``initialization_type`` of the method :py:func:`.filter_initialization`; if dictionary, \
        the provided keys correspond to ``initialization_type`` and the corresponding keyword arguments \
        of the method :py:func:`.filter_initialization`; if a scalar (float or int) all elements of the filter \
        will be set to the provided value; and if tensor it will need to exactly match the size of requested filter
        size ``m_kernel_size``. \
        Default: ``'zeros'``
    m_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary, \
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    b_type : str, optional
        Values among ``'scalar_per_channel'``, ``'scalar'``.
        It defines whether the same bias, although independent per channel, is shared for the whole image extent
        (``'scalar_per_channel'``), or otherwise single scalar value is considered all pixels and all channels
        (``'scalar'``) \
        (see above for the implications of each version). Default: ``'scalar_per_channel'``
    initial_b : int or float or torch.Tensor, optional
        If ``b_type`` is ``'scalar_per_channel'`` and the provided value is a scalar, that value will correspond \
        to all the dimensions of the vector.
        Default: ``0.0``
    b_trainable : bool, optional
        Indicates whether the corresponding element of ``theta`` performs gradient calculation or, on the contrary,
        remains with its initial value irrespective of successive optimization steps. \
        Default: ``True``
    in_size : tuple[int], optional
        2D tuple (2D at least; trailing dimensions do not matter for the involved processing regarding only \
        spatial extent) indicating the size of the input image to the layer. This argument is not necessary if \
        the kernel sizes for *m* and *w* are provided as abslute int values; however it becomes necessary if \
        any of them is provided as relative (e.g. floats $0 \\leq x \\leq 1$) to the input size. \
        Default: ``None``
    """

    def __init__(self,
                 in_channels, out_channels,
                 phi_activation,
                 m_kernel_size, m_padding='same', m_padding_mode='zeros', m_groups=1,
                 m_initialization='zeros', m_trainable=True,
                 b_type='scalar_per_channel', initial_b=0.0, b_trainable=True,
                 in_size=None):
        #
        ###
        # Call the constructor "__init__" of the super class 'nn.Module' (the abstract class is not even defined):
        ###
        nn.Module.__init__(self)

        ###
        # Call the constructor "__init__" of the super class 'ModifiedRFLayer' taking care of the kwargs
        ###
        ModifiedRFLayer.__init__(self)

        ###
        # Initial checks regarding compatibility of channel number vs groups vs image dimensions
        ###

        # And fix 'w_padding' since no other value is allowed
        dict_kernel_size = {'m': m_kernel_size}
        dict_padding = {'m': m_padding}
        dict_padding_mode = {'m': None if m_padding=='fc' else m_padding_mode}
        dict_groups = {'m': m_groups}
        dict_kernel_out_in_channels = {'m': (out_channels, in_channels)}
        #
        # Check/adapt/resolve the kernel size when m,b/w is convolutional
        for key in dict_kernel_size:  # 'm' will be now checked by the function 'm_and_b_dimensioning'
            if dict_padding[key] in ['fc']:
                pass
            elif dict_padding[key] in ['same', 'valid']:
                # In this case the kernel size must be provided
                if dict_kernel_size[key] is None:
                    raise Exception(
                        (f"No kernel size for '{key}', or 'None', is provided! (Reminder: " +
                         f"full-image, uniform kernels are noted as '[]' or '()'!)")
                    )
                pass
                #
                # Reformat to help checks (not resolving yet for relative filters)
                dict_kernel_size[key] = nnl.kernel_size_check_and_reformat_into_tuple(dict_kernel_size[key])
                if len(dict_kernel_size[key]) == 0 and key == 'm':
                    raise Exception(f"Full-image, uniform kernels requested for '{key}: only valid for 'w'!")
                pass
                #
                # And resolve the (spatial) kernel size if necessary
                dict_kernel_size[key] = nnl.resolve_kernel_size_for_im_size(
                    dict_kernel_size[key], im_size=in_size, make_odd=False
                )
            else:
                raise Exception(
                    (f"Valid values for '{key}_padding' are " + ("'fc, " if key == 'm' else "") +
                     f"'same', and 'valid'; '{dict_padding[key]}' provided!"))
            pass
            #
        pass

        # Check the validity of the groups
        # Check the validity of the groups
        for key in ['m']:
            if (dict_kernel_out_in_channels[key][0] % dict_groups[key] != 0) or \
                    (dict_kernel_out_in_channels[key][1] % dict_groups[key] != 0):
                raise Exception((f"Both 'in_channels'={dict_kernel_out_in_channels[key][1]} and " +
                                 f"'out_channels'={dict_kernel_out_in_channels[key][0]} of '{key}' " +
                                 f"must be divisible by 'm_groups'={dict_groups[key]}!"))
            pass
        pass

        # Create templates (only size is important) for M and b according to the function 'm_and_b_dimensioning'!!!
        # The function 'm_and_b_dimensioning' additionally checks the compatibility of the different arguments
        template_m, template_b = m_and_b_dimensioning(
            in_channels=in_channels, out_channels=out_channels,
            m_kernel_size=m_kernel_size, m_padding=m_padding, m_groups=m_groups,
            b_type=b_type, in_size=in_size
        )

        #######################################
        # STORE PARAMETERS THAT WILL BE RELEVANT LATER ON AS ATTRIBUTES
        #######################################

        # self._phi_activation_f = nnl.ndim_activation_function_from_1dim_activation_functions(phi_activation)
        self._phi_activation = phi_activation

        self._m_kernel_size = dict_kernel_size['m']
        self._m_groups = dict_groups['m']
        self._m_padding = dict_padding['m']
        self._m_padding_mode = dict_padding_mode['m']

        self._m_initialization = m_initialization if isinstance(m_initialization, str) else 'custom'

        self._b_type = {None if m_padding=='fc' else b_type}

        #######################################
        # Define an initial "_theta0" (kept as record) and transfer it to the dynamically optimized "_theta"
        ###
        # "_theta" is initialized deterministically as "_theta0" according to some desired rules. Later, and before
        # training, random disturbances can be introduced with the appropriate method of the class
        #######################################

        # Initial and "dynamic" parameter set
        self._theta0 = {}

        # Deterministic initialization of "_theta0" according to the provided arguments

        # For 'm' and 'w': We have "dict_groups", "dict_kernel_out_in_channels", and "dict_kernel_size" from before,
        # we define also...
        dict_initialization = {'m': m_initialization}

        for key in ['m']:
            #
            #####
            # Set the correct size for m/w and create an empty matrix for it
            #####
            #
            tensor_dims = template_m.size() if key == 'm' else \
                (dict_kernel_out_in_channels[key][0],
                 int(dict_kernel_out_in_channels[key][1] / dict_groups[key])) + dict_kernel_size[key]
            self._theta0[key] = torch.zeros(tensor_dims)
            #
            #####
            # Initialize its content
            #####
            #
            if isinstance(dict_initialization[key], str):
                self._theta0[key] = nnl.filter_initialization(
                    tensor_dims, initialization_type=dict_initialization[key],
                    im_size=in_size, groups=dict_groups[key],
                    dtype=torch.float32
                )
            elif isinstance(dict_initialization[key], dict):
                self._theta0[key] = nnl.filter_initialization(
                    tensor_dims,
                    im_size=in_size, groups=dict_groups[key],
                    **dict_initialization[key], dtype=torch.float32
                )
            elif isinstance(dict_initialization[key], (int, float)):
                self._theta0[key] = torch.empty(tensor_dims)
                self._theta0[key][:] = dict_initialization[key]
            elif isinstance(dict_initialization[key], torch.Tensor):
                if dict_initialization[key].size() != tensor_dims:
                    raise Exception((
                            f"The size of the provided tensor for '{key}', {dict_initialization[key].size()}, " +
                            f"is not compatible with the requested size {tensor_dims}")
                    )
                self._theta0[key] = torch.empty(tensor_dims)
                self._theta0[key][:] = dict_initialization[key]
            pass
        pass

        # For 'b':
        dict_type = {'b': b_type}
        dict_initial_value = {'b': initial_b}
        for key in ['b']:
            #
            self._theta0[key] = torch.empty(template_b.size())
            #
            if isinstance(dict_initial_value[key], int) or isinstance(dict_initial_value[key], float):
                self._theta0[key][:] = dict_initial_value[key]
            elif isinstance(dict_initial_value[key], torch.Tensor):
                self._theta0[key][:] = dict_initial_value[key][:]
            else:
                raise Exception((f"The selected initialization value for '{key}' is not an accepted type " +
                                 f"float, int, or torch.Tensor," +
                                 f"but the following: {type(dict_initial_value[key])}"))
            pass
            #
        pass

        #######################################
        # CREATE THE (HIDDEN) ATTRIBUTE "_theta_trainable",
        # WHICH IS A DICTIONARY OF BOOLS MIRRORING THE FIELDS OF "_theta"
        #######################################

        self._theta_trainable = {}
        for key in self._theta0:
            self._theta_trainable[key] = False
        pass
        self._theta_trainable['m'] = m_trainable
        self._theta_trainable['b'] = b_trainable

        #######################################
        # FINALLY WE CREATE THE DYNAMIC "_theta" with their ".requires_grad" according to "_theta_trainable"
        #######################################

        self._theta = nn.ParameterDict()
        for key in self._theta0:
            self._theta[key] = nn.Parameter(self._theta0[key].clone(), requires_grad=self._theta_trainable[key])
        pass

        self._update_underlying_f()

    @staticmethod
    def constructor_default_values(query_args=None, only_not_none=False):
        """
        Returns a dictionary with the arguments accepted by the constructor of the class \
        and their default values. When no default value is included in the constructor, the value is set to ``None``. \
        When the argument query_args` is provided, only the values of the arguments of interest are returned. \
        When `only_not_none` is set to ``True``, only the arguments with default values (i.e. different from ``None``) \
        are returned. \
        'kwargs' is not included.

        Parameters
        ----------
        query_args : str or list[str] or tuple[str], optional
            Name of the arguments of interest: only those would be returned. Default: ``None``
        only_not_none : bool, optional
            If ``True``, only the arguments with default values (i.e. different from ``None``) are returned. \
            Default: ``False``

        Returns
        -------
        dict
        """

        return ModifiedRFLayer._constructor_default_values_given_class(SMLayer,
                                                                       query_args=query_args,
                                                                       only_not_none=only_not_none)

    pass

    def _update_underlying_f(self):
        def f_INRF_version_i(im_f, theta_f):
            return f_modified_RF(
                im_f, theta_f, u=None, v=None,
                phi_activation=self._phi_activation,
                m_padding=self._m_padding, m_padding_mode=self._m_padding_mode, m_groups=self._m_groups,
            )

        self._f_INRF_version_i = f_INRF_version_i

    pass

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            The output of the layer.
        """

        ################
        # Check whether operations should be pushed into GPU
        ################
        computation_device = 'cpu'
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # If here, 'cuda' is available.
            # Check if 'theta' is defined so, and raise error if any key in 'theta' disagrees with the rest
            for key in self._theta:
                # print(f"'Device of 'theta['{key}']' = {self._theta[key].device.type}")
                if self._theta[key].device.type == 'cpu' and computation_device == 'cuda':
                    raise Exception(
                        f"Certain keys of 'theta' (e.g. '{key}') are 'cpu' while others 'cuda': inconsistent!")
                else:
                    computation_device = self._theta[key].device.type
                pass
            pass
        pass
        # print(f"'Device after checks = {computation_device}")
        ################
        if x.device.type != computation_device:
            x = x.to(computation_device)
        pass

        return self._f_INRF_version_i(
            x, self._theta
        )

    pass
