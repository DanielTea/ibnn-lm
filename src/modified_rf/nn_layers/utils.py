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

import warnings

import numpy as np
import scipy as sp
from skimage import filters

from math import floor, ceil

import torch
import torch.nn.functional as F

import gc
import modified_rf.memory_handling as memo



######################################################
######################################################
######################################################
#
# THIS FILE CONTAINS AUXILIARY FUNCTIONS AND STRUCTURES FOR nn_layers.core.py
#
######################################################
######################################################
######################################################


######################################################
######################################################
#
# 2D KERNEL GENERATOR FOR STANDARD FILTERS
#
######################################################
######################################################


def mono_image_kernel_generator(size, kernel_type, dtype=torch.float32, **kwargs):
    """
    Generation of a 2D kernel mask, of 1 channel only,  based on the (implicitly) implemented filters of \
    `scikit-image <https://scikit-image.org/docs/stable/api/skimage.filters.html>`_.

    Parameters
    ----------
    size : int or tuple[int]
        1D or 2D size. If the argument is an integer $s$ it is equivalent to the tuple $(s,s)$
    kernel_type : str
        Valid values are: ``'gaussian'``, ``'difference_of_gaussians'``, ``'gabor'``.
    dtype : torch.dtype
        Default: ``torch.float32``
    kwargs
        These keyword arguments relate to the additional parameters required by the corresponding filter of \
        `scikit-image <https://scikit-image.org/docs/stable/api/skimage.filters.html>`_:

        - for ``'gaussian'`` the argument ``sigma`` is additionally required;
        - for ``'difference_of_gaussians'`` the arguments ``low_sigma``, ``high_sigma`` \
             are additionally required;
        - for ``'gabor'`` the arguments ``sigma``, ``frequency``, ``theta`` are additionally required.
    Returns
    -------
    torch.Tensor
        2D tensor with the resulting filter kernel
    """

    # Make size always tuple
    if isinstance(size, int):
        size = (size, size)
    pass
    if len(size) != 2:
        raise Exception(f"Invalid input size!")
    pass

    # Generate a Dirac's delta 2D image with the delta in the middle and the desired kernel size
    delta = sp.signal.unit_impulse(size, idx='mid')

    # Filter it using the corresponding 'scikit-image' function
    kernel = None
    if kernel_type == 'gaussian':
        kernel = filters.gaussian(image=delta, sigma=kwargs['sigma'], mode='constant', cval=0.0)
    elif kernel_type == 'difference_of_gaussians':
        kernel = filters.difference_of_gaussians(
            image=delta, low_sigma=kwargs['low_sigma'], high_sigma=kwargs['high_sigma'],
            mode='constant', cval=0.0
        )
    elif kernel_type == 'gabor':
        kernel, _ = filters.gabor(
            image=delta, frequency=kwargs['frequency'], theta=kwargs['theta'],
            mode='constant', cval=0.0
        )
    else:
        raise Exception((
                f"Only 'gaussian', 'difference_of_gaussians', and 'gabor' as admitted kernel types: " +
                f"{kernel_type} found!"
        ))
    pass
    #
    kernel = torch.Tensor(kernel)
    #
    return kernel


def filter_initialization(kernel_size, initialization_type, im_size=None, groups=None,
                          dtype=torch.float32, **kwargs):
    """
    For a given kernel size, it initializes a filter kernel according to the provided initialization type. \
    The kernel size can be either a 2D or a 4D tensor, wherein the two first dimensions represent the number of output \
    and input channels (remember the channel-related kernel convention of :py:func:`torch.nn.Conv2d`, dependent on \
    the ``groups`` parameter), and the two remaining dimensions, if present, represent the spatial dimensions $(H, W)$ \
    of the required filter.

    Parameters
    ----------
    kernel_size : tuple[int]
        Tuple of length 4 or 2: in both cases the two first dimensions represent \
        the number of output and input channels; the former (4D) case, standard, the two remaining dimensions \
        represent the spatial dimensions of the filter $(H, W)$; in the latter case, where those spatial \
        dimensions are missing, the filter will be understood as a full-image, uniform filter represented as a scalar
        for each combination of input-output channels. In said case the initialization \
        requires the size ``im_size`` of the input image
    initialization_type : str
        Value among ``'zeros'``, ``'ones'``, ``'delta'``, ``'eye'``, ``'gaussian'``, \
        ``'difference_of_gaussians'``, and ``'gabor'``. \
        The respective string types presume different keyword arguments, described below; and \
        in the case of full-image, uniform filter initialization (``kernel_size`` of length 2) only \
        the two first options are valid
    im_size : tuple[int], optional
        Size of the input image. Required only if ``kernel_size`` is of length 2 \
        (full-image, uniform filter), or if the keywords of certain ``initialization_type`` are \
        expressed as relative (e.g. ``rel_sigma`` in the case of ``gaussian``). \
        Only the -2 and -1 dimensions will be used. \
        Default: ``None``
    groups : int
        Number of groups. Required only if ``initialization_type``  if  one of ``'ones'``, ``'delta'``, or \
        ``'gaussian'`` and their ``normalization`` keyword (see below) indicates ``'full'`` normalization.
        Default: ``None``
    dtype : torch.dtype
        Default: ``torch.float32``
    kwargs :
        - For ``'zeros'`` no additional keywords are used.
        - For ``'delta'`` and ``'ones'`` the parameter ``normalization``, of type \
          `bool <https://docs.python.org/3/library/stdtypes.html#str>`_, can be optionally introduced to \
          perform: ``None`` (no normalization, e.g. simply ones in all positions); ``'individual'`` channel \
          normalization (i.e. spatial only, that is, the elements of the transformation between an individual \
          channel $c_i$ of the input and an individual channel $c_o$ of the output sum $1$); \
          ``'group'`` normalization (i.e. the transformations, accumulated, transforming into a certain output channel \
          its corresponding group of input channels accumulate, together, $1$); and \
          ``'full'`` normalization, considering the joint set of input channels and set of output channels. \
          If no value provided, ``'individual'`` normalization is considered as default
        - For ``initialization_type`` in ``'gaussian'`` the parameter ``normalization`` is, as above, accepted, \
          as well as the keyword ``sigma`` described in :py:func:`.mono_image_kernel_generator` or, \
          alternatively, the keyword ``rel_sigma``, of type \
          `float <https://docs.python.org/3/library/stdtypes.html#float>`_ , which refers to the same information \
          but relative to the size of the input image and must be expressed with a scalar between 0 and 1.
        - For ``initialization_type`` in ``'difference_of_gaussians'``, and ``'gabor'`` \
          see the keyword options described in :py:func:`.mono_image_kernel_generator`. Again, \
          relative sigmas (which contain the prefix ``'rel_'``) are resolved with respect to the input image size and \
          and must be expressed with a scalar between 0 and 1.

    Returns
    -------
    torch.Tensor
        Tensor with the resulting filter kernel
    """

    ########
    # Check the validity of the requested kernel size and define (empty) the kernel to return and,
    # if 'kernel_size' of length 2, it is a full-image, uniform filter: 'im_size' is necessary
    ########

    kernel = None
    if len(kernel_size) == 2 or len(kernel_size) == 4:
        c_out = kernel_size[0]
        c_in_per_group = kernel_size[1]
        kernel = torch.empty(kernel_size, dtype=dtype)
        if len(kernel_size) == 2:
            if im_size is None:
                raise Exception((
                        f"Parameter 'im_size' must be provided for full-image, uniform filter initialization: " +
                        f"instead, 'im_size' is {im_size}"
                ))
            pass
        pass
    else:
        raise Exception(f"Parameter 'size' must be 2D or 4D: instead, the provided size is {kernel_size}")
    pass

    ########
    # Check valid options for 'initialization_type' depending on full-im or not
    ########

    if initialization_type not in ['zeros', 'ones', 'delta', 'eye', 'gaussian', 'difference_of_gaussians', 'gabor']:
        raise Exception((
                f"Invalid initialization type: {initialization_type}, of type {type(initialization_type)}; " +
                f"only 'zeros', 'ones', 'delta', 'gaussian', 'difference_of_gaussians', and 'gabor' are accepted."
        ))
    elif len(kernel_size) == 2 and initialization_type not in ['zeros', 'ones']:
        raise Exception((
                f"Invalid initialization type for full-image, uniform filter: {initialization_type} requested; " +
                f"only 'zeros' and 'ones' are accepted."
        ))
    pass

    ########
    # Resolve relative sigma keywords if there are any in 'kwargs'
    ########

    list_rel_keys = [key for key in kwargs if (key.startswith('rel_') and 'sigma' in key)]

    if len(list_rel_keys) > 0:
        #
        if im_size is None:
            raise Exception((
                    f"Parameter 'im_size' must be provided for relative sigma initialization: " +
                    f"instead, 'im_size' is {im_size}"
            ))
        pass
        #
        for key in list_rel_keys:
            rel_key_value = kwargs.pop(key)
            # It must be float and between 0 and 1
            if not isinstance(rel_key_value, float) or rel_key_value <= 0 or rel_key_value >= 1:
                raise Exception((
                        f"Relative keyword parameter '{key}' must be a float between 0 and 1: " +
                        f"instead, '{key}' is {rel_key_value}"
                ))
            pass
            #
            new_key = key[4:]
            kwargs[new_key] = rel_key_value * 0.5 * (im_size[-2] + im_size[-1])
        pass
    pass

    ########
    # Check if the kwarg argument 'groups' is provided
    ########

    normalization = kwargs.pop('normalization', 'individual')
    if normalization not in [None, 'individual', 'group', 'full']:
        raise Exception((
                f"Invalid group normalization type: {normalization}, of type {type(normalization)}; " +
                f"only 'individual', 'group', 'full', and None are accepted."
        ))
    pass

    # Once the checks have been done, we proceed with the initialization:
    if not isinstance(initialization_type, str):
        raise Exception((
                f"Invalid initialization type: {initialization_type}, of type {type(initialization_type)}; " +
                f"only is accepted."
        ))
    else:
        if initialization_type == 'zeros':
            kernel[:] = 0.0
        elif initialization_type in ['ones', 'delta', 'eye', 'gaussian']:
            if initialization_type == 'ones':
                kernel[:] = 1.0
                if normalization is not None:  # At least spatial normalization
                    if len(kernel_size) == 2:
                        numel = im_size[-2] * im_size[-1]
                    elif len(kernel_size) == 4:
                        numel = kernel_size[-2] * kernel_size[-1]
                    else:
                        raise Exception(f"Unexpected kernel size: {kernel_size}")
                    pass
                    kernel[:] = 1.0 / numel
                pass
            elif initialization_type == 'delta':
                pixel_pos_spike = [int(floor(kernel_size[ind] - 1) / 2) for ind in [2, 3]]
                kernel[:] = 0.0
                kernel[:, :, pixel_pos_spike[-2], pixel_pos_spike[-1]] = 1.0
            elif initialization_type == 'eye':
                kernel[:] = 0.0
                for ind in range(min(kernel_size[-2], kernel_size[-1])):
                    kernel[:, :, ind, ind] = 1.0
                pass
            else:  # Which means 'gaussian'
                spatial_kernel = mono_image_kernel_generator(
                    (kernel.size(2), kernel.size(3)),
                    initialization_type, dtype=dtype, **kwargs
                )
                for out_ch in range(kernel.size(0)):
                    for in_ch in range(kernel.size(1)):
                        kernel[out_ch, in_ch, :] = spatial_kernel[:]
                    pass
                pass
            pass
            #
            # Now, each 1-input->1-output transform is normalized. We add additional normalization factor if requested
            if normalization == 'group':
                # Divided by the number of input channels per output, so each channel has the average energy \
                # of its input channels
                kernel[:] = kernel[:] / (kernel.size(1))
            elif normalization == 'full':
                # Divided by the ratio between output and input channels. For that we need the number of groups
                if groups is None:
                    raise Exception((
                            f"Parameter 'groups' must be provided for 'full' normalization: " +
                            f"instead, 'groups' is {groups}"
                    ))
                pass
                kernel[:] = kernel[:] / (kernel.size(0) / (kernel.size(1) * groups))
            pass
        elif initialization_type in ['difference_of_gaussians', 'gabor']:
            spatial_kernel = mono_image_kernel_generator((kernel.size(2), kernel.size(3)),
                                                         initialization_type, dtype=dtype, **kwargs)
            for out_ch in range(kernel.size(0)):
                for in_ch in range(kernel.size(1)):
                    kernel[out_ch, in_ch, :] = spatial_kernel[:]
                pass
            pass
        else:
            raise Exception((
                    f"Invalid initialization type: {initialization_type}, of type {type(initialization_type)}; " +
                    f"only 'zeros', 'ones', 'delta', 'gaussian', 'difference_of_gaussians', and 'gabor' are accepted."
            ))
        pass
    pass

    return kernel


pass

######################################################
######################################################
#
# KERNEL SIZE CHECK AND RESOLUTION
#
######################################################
######################################################

def kernel_size_check_and_reformat_into_tuple(kernel_size, make_odd=False):
    """
    It checks the validity of the kernel size `kernel_size` size and \
    adapts it to tuple in any case.
    Kernel size can be either an int x>0, in which case it means absolute pixels, or a float 0<x<1, \
    in which case it refers to a proportion of the input size.

    Parameters
    ----------
    kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked
    make_odd : bool, optional
        If ``True`` and if `kernel_size` tuple of ints or int, it forces (each dimension) to the nearest integer \
        greater or equal to the provided value. For floats the argument `make_odd` has no efect. \
        Default: ``False``

    Returns
    -------
    tuple[int] or tuple[float]

    """
    adapted_kernel_size = None

    # Make tuple (always) out of the provided values
    if isinstance(kernel_size, (tuple, list)):
        adapted_kernel_size = tuple(kernel_size)
    elif isinstance(kernel_size, float) or isinstance(kernel_size, int):
        if isinstance(kernel_size, int) and kernel_size == 0:
            # THIS CASE IS EQUIVALENT TO THE CASE OF AN EMPTY TUPLE; WE MAKE THEM COINCIDENT HERE
            adapted_kernel_size = ()
        else:
            adapted_kernel_size = (kernel_size, kernel_size)
        pass
    else:
        raise Exception(f"The provided 'kernel_size' is not tuple, float, or int, as required. Instead, " +
                        f"the following object of type {type(kernel_size)} has been provided:\n" +
                        f"{kernel_size}")
    pass

    # Check the values of the tuple
    if not isinstance(adapted_kernel_size, tuple):
        raise Exception(f"Unexpected problem: the function should have made 'adapted_kernel_size' always tuple, " +
                        f"and however it is not tuple but type {type(adapted_kernel_size)}!")
    elif len(adapted_kernel_size) == 0:  # Empty tuple: pass it through
        pass
    elif len(adapted_kernel_size) == 2:
        if all([isinstance(elem, int) for elem in adapted_kernel_size]) and \
                all([elem > 0 for elem in adapted_kernel_size]):
            # Good, all are positive
            pass
            # Make them odd if requested
            if make_odd:
                adapted_kernel_size = tuple([
                    2 * (elem // 2) + 1 for elem in adapted_kernel_size
                ])
            pass
        elif all([isinstance(elem, float) for elem in adapted_kernel_size]) and \
                all([0.0 < elem <= 1.0 for elem in adapted_kernel_size]):
            pass
        else:
            raise Exception(f"The tuple provided as 'kernel_size' must be either floats 0<x<1 or integers x>0, " +
                            f"{adapted_kernel_size} found.")
        pass
    pass
    #
    return adapted_kernel_size


pass


def resolve_kernel_size_for_im_size(kernel_size, im_size=None, make_odd=False):
    """
    When the provided 'kernel_size' is composed of floats this function resolves the indicated proportions, which \
    are to be understood relative to the input image size, using the provided input image size `im_size`.
    adapts it to tuple in any case.
    The kernel size can be either an int x>0, in which case it means absolute pixels, or a float 0<x<1, \
    in which case it refers to a proportion of the input size.

    Parameters
    ----------
    kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked.
    im_size : tuple[int] or list[int], optional
        Default: ``None``
    make_odd : bool, optional
        If ``True`` and if `kernel_size` tuple of ints or int, it forces (each dimension) to the nearest integer \
        greater or equal to the provided value. For floats the argument `make_odd` has no efect. \
        Default: ``False``

    Returns
    -------
    tuple[int]
    """

    # Get the kernel size checked and in tuple mode
    adapted_kernel_size = kernel_size_check_and_reformat_into_tuple(kernel_size, make_odd=make_odd)

    # If empty tuple: pass it through
    if len(adapted_kernel_size) == 0:
        pass
    else:
        adapted_kernel_size = list(adapted_kernel_size)
        #
        if isinstance(adapted_kernel_size[-1], float):
            if im_size is None or (not (isinstance(im_size, (tuple, list)))) or len(im_size) != 2:
                raise Exception(f"Float-based kernel size requires 'im_size' of length 2; " +
                                f"got {im_size} of type {type(im_size)} instead.")
            # If float: elements in 'adapted_kernel_size', already checked, as relative and resolve using the 'im_size'
            for ind in [-2, -1]:
                if adapted_kernel_size[ind] == 1.0:
                    adapted_kernel_size[ind] = im_size[ind]
                else:
                    # Make them odd if requested
                    if make_odd:
                        adapted_kernel_size[ind] = 2 * round((adapted_kernel_size[ind] * im_size[ind] - 1) / 2) + 1
                    else:
                        adapted_kernel_size[ind] = round(adapted_kernel_size[ind] * im_size[ind])
                    pass
                pass
            pass
        elif isinstance(adapted_kernel_size[-1], int):
            # Make them odd if requested
            if make_odd:
                adapted_kernel_size = tuple([
                    2 * (elem // 2) + 1 for elem in adapted_kernel_size
                ])
            pass
        else:
            raise Exception(f"Unexpected problem: the function should have made 'adapted_kernel_size' always tuple, " +
                            f"and however it is not tuple but type {type(adapted_kernel_size)}!")
        pass
    pass

    return tuple(adapted_kernel_size)


pass

######################################################
######################################################
#
# ACTIVATION FUNCTION UTILITIES
#
######################################################
######################################################


######################################################
# Dictionary linking name ids for common activation functions (most from torch.nn.functional) and the function
######################################################

# 'ranges' is currently not used for anything

_activation_function_dict = {
    'identity': {
        'function': lambda x: x,
        # 'ranges': [
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'relu': {
        'function': F.relu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'leaky_relu': {
        'function': F.leaky_relu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'elu': {
        'function': F.elu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'selu': {
        'function': F.selu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'celu': {
        'function': F.celu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'prelu': {
        'function': lambda x: F.prelu(x, weight=torch.Tensor([0.01])),
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'rrelu': {
        'function': F.rrelu,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'atan': {
        'function': lambda x: torch.atan(x),
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'norm_atan': {
        'function': lambda x: torch.atan(np.pi / 2.0 * x) * 2.0 / np.pi,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'tanh': {
        'function': F.tanh,
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'hardtanh': {
        'function': lambda x: F.hardtanh(x, min_val=-1.0, max_val=1.0),
        # 'ranges': [
        #     {'upper_lim': 0.0, 'linear': True},
        #     {'upper_lim': np.inf, 'linear': True}
        # ]
    },
    'softsign': {
        'function': F.softsign,
        'ranges': [
            {'upper_lim': 0.0, 'linear': True},
            {'upper_lim': np.inf, 'linear': True}
        ]
    },
    'sigmoid': {
        'function': F.sigmoid,
        'ranges': [
            {'upper_lim': 0.0, 'linear': True},
            {'upper_lim': np.inf, 'linear': True}
        ]
    },
    'silu': {
        'function': F.silu,
        'ranges': [
            {'upper_lim': 0.0, 'linear': True},
            {'upper_lim': np.inf, 'linear': True}
        ]
    },
}

activation_function_dict = _activation_function_dict


######################################################
# Function obtaining workable/callable activation functions from str ids:
# resolving 1D activation functions from str or from callable
######################################################

def _resolve_activation_function(activation_function):
    #
    # First, define the callable function from the provided argument
    activation_f = None
    if isinstance(activation_function, str):
        try:
            activation_f = _activation_function_dict[activation_function]['function']
        except KeyError:
            raise Exception((
                    f"Provided textual activation function '{activation_function}' is not valid. " +
                    f"Accepted values: {list(_activation_function_dict.keys())}."
            ))
        pass
    elif callable(activation_function):
        activation_f = activation_function
    else:
        raise Exception(
            f"The provided 'activation_function' selector is not str or callable:  type {type(activation_function)}"
        )
    pass
    #
    return activation_f


######################################################
# Function obtaining workable/callable activation functions from str ids:
# obtaining component-wise multi-dimensional activation function from list of 1D functions (str or callables)
######################################################

def ndim_activation_function_from_1dim_activation_functions(
        activation_function,
        x_compress=1.0, y_stretch=1.0, x_offset=0.0, y_offset=0.0):
    """
    Function generating an activation function acting separately on each channel c-th of its input tensor
    according to each c-th 1D activation function of the $C_{in}$ functions provided in ``activation_function``.
    The resulting activation function operates component-wise, that is,
    $$
    \\bar{\\sigma}(\\mathbf{u})=
    \\Big(\\sigma_1(u_1),\\ldots,\\sigma_c(u_c),\\ldots,\\sigma_{C_{in}}(u_{C_{in}})\\Big) \\, \\, .
    $$

    Regarding the activation function modifiers ``x_compress`` and ``y_stretch``: in an example wherein both \
    ``activation_function`` and the modifiers are single values and not vectors, \
    and having ``x_compress`` and ``y_stretch`` as, respectively, $\\alpha_x$ and $\\beta_y$ and having \
    ``x_offset`` and ``y_offset`` as, respectively, $\\gamma_x$ and $\\gamma_y$ (and ), their effect on the \
    resulting activation function derived from ``activation_function`` $\\sigma(u)$ would be
    $$
    \\hat{\\sigma}(u) = \\beta_y \\; \\sigma \\!\\Big( \\! \\alpha_x (u-\\gamma_x) \\!\\Big) + \\gamma_y
    $$
    It can be seen that $\\hat{\\sigma}^\\prime(\\gamma_x) =  \\alpha_x \\, \\beta_y \\, \\sigma^\\prime(0)$.

    Moreover, if ``activation_function`` is a list of activation functions wherein each one corresponds to a channel, \
    so must be the activation function modifiers ``x_compress``, ``y_stretch``, ``x_offset``, and ``y_offset`` \
    so each activation function dimension is affected by the corresponding modifier. In such case, \
    each one of ``x_compress``, ``y_stretch``, ``x_offset``, and ``y_offset`` would be either a list or a tuple \
    or a :py:class:`torch.Tensor` with the same length of ``activation_function``.

    Parameters
    ----------
    activation_function : str or list[str] or tuple[str] or ~collections.abc.Callable or \
    list[~collections.abc.Callable] or tuple[~collections.abc.Callable]
        Iterable with as many 1D activation functions as channels
        Valid string options are: \
        ``'identity'``, ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'selu'``, ``'celu'``, ``'prelu'``, \
        ``'rrelu'``, ``'tanh'``, ``'hardtanh'``, ``'softsign'``, ``'sigmoid'``, and ``'silu'``.
        String description of the desired $\\Phi(\\cdot)$ activation function or :py:obj:`~collections.abc.Callable` \
        (i.e. function or :py:mod:`torch.nn.functional`)
    x_compress, y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifiers to the basic activation function. Default: ``1.0``
    x_offset, y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Offset modifiers to the basic activation function. Default: ``0.0``

    Returns
    -------
    activation_f : ~collections.abc.Callable
        Function
    """

    # Just to ease the dimension checks, since the checks are identical for all modifiers
    list_of_activation_function_modifier_names = ['x_compress', 'y_stretch', 'x_offset', 'y_offset']
    list_of_activation_function_modifiers = [x_compress, y_stretch, x_offset, y_offset]

    activation_f = None
    if isinstance(activation_function, str) or callable(activation_function):
        #
        #############
        # COMPATIBILITY AND TYPE CHECKS
        #############
        for i, modifier in enumerate(list_of_activation_function_modifiers):
            if not (isinstance(modifier, float) or isinstance(modifier, int) or \
                    (isinstance(modifier, torch.Tensor) and (modifier.numel() == 1))):
                raise Exception((f"The provided 'activation_function' {activation_function} suggested a scalar " +
                                 f" or 1D Tensor modifier '{list_of_activation_function_modifier_names[i]}' but " +
                                 f" instead the function found: {list_of_activation_function_modifiers[i]}"))
            pass
        pass
        #############
        # APPLY THE MODIFIERS
        #############
        standard_activation_f = _resolve_activation_function(activation_function)
        activation_f = lambda x: y_stretch * standard_activation_f(x_compress * (x - x_offset)) + y_offset
        #
    elif isinstance(activation_function, list) or isinstance(activation_function, tuple):
        #
        #############
        # COMPATIBILITY AND TYPE CHECKS
        #############
        # Check if types and the dimensions of '' and of the modifiers are compatible
        num_activation_functions = len(activation_function)
        for i, activation_function_i in enumerate(activation_function):
            if not (isinstance(activation_function_i, str) or callable(activation_function_i)):
                raise Exception((f"{i}-th element of 'activation_function' of type {type(activation_function_i)}: " +
                                 f"str or callable was instead expected for each component."))
            pass
        pass
        for i, modifier in enumerate(list_of_activation_function_modifiers):
            if not ((isinstance(modifier, list) and (len(modifier) == num_activation_functions)) or \
                    (isinstance(modifier, tuple) and (len(modifier) == num_activation_functions)) or \
                    (isinstance(modifier, torch.Tensor) and (modifier.numel() == num_activation_functions))):
                raise Exception((f"The provided '{list_of_activation_function_modifier_names[i]}' must contain " +
                                 f"as many elements as 'activation_function', that is, {num_activation_functions}; "
                                 f"however {modifier} has been provided."))
            pass
        pass
        #############
        # APPLY THE MODIFIERS
        #############
        # Resolve the given list of 1dim activation functions applying the modifiers
        list_standard_activation_functions = []
        for activation_function_i in activation_function:
            list_standard_activation_functions.append(_resolve_activation_function(activation_function_i))
        pass

        # Define the function, acting differently for each input channel (dim=-3)
        def ndim_activation_f(x):
            # Are the channel dimensions compatible?
            if len(list_standard_activation_functions) != x.size(-3):
                raise Exception((f"The function is {len(list_standard_activation_functions)}-dim; however the input " +
                                 f"to the function has dimension {x.size()}!"))
            pass
            # If they are compatible, we process each dimension
            # We move the channel dimension to the first position... (and we will return it back by the end)
            x = x.movedim(source=-3, destination=0)
            y = torch.empty(x.size())
            for i in range(y.size(0)):
                y[i] = y_stretch[i] * list_standard_activation_functions[i](
                    x_compress[i] * (x[i] - x_offset[i])
                ) + y_offset[i]
            pass
            # We move the channel dimension back to position -3
            return y.movedim(source=0, destination=-3)

        pass
        activation_f = ndim_activation_f
    else:
        raise Exception((f"The type of the provided 'activation_function', {type(activation_function)}, is " +
                         f"not valid: str, list[str], tuple[str] or callable are instead expected."))
    pass
    #
    return activation_f


######################################################
######################################################
#
# INTERPOLATION UTILITIES
#
######################################################
######################################################


######################################################
# DICTIONARIES OF NORMALIZED INTERPOLATING KERNELS
######################################################

# Basic first-order interpolating functions (defined for all x in R)
_b_raised_cosine = 1.0
_interpolation_kernel_dict = {
    'linear': lambda x, delta_sampling_points: torch.where(
        torch.abs(x) < delta_sampling_points,
        (1 - torch.abs(x) / delta_sampling_points),
        0.0
    ),
    'closest': lambda x, delta_sampling_points: torch.where(
        torch.abs(x) < 0.5 * delta_sampling_points,
        1.0,
        0.0
    ),
    'raised_cosine': lambda x, delta_sampling_points: \
        0.5 * (1 + torch.cos(
            np.pi / _b_raised_cosine * (
                    torch.clamp(
                        torch.abs(x) / delta_sampling_points,
                        min=0.5 * (1.0 - _b_raised_cosine),
                        max=0.5 * (1.0 + _b_raised_cosine)
                    )
                    - 0.5 * (1.0 - _b_raised_cosine)
            )
        )
               ),
    'sinc': lambda x, delta_sampling_points: torch.sinc(x / delta_sampling_points)
}

# Basic first-order extrapolating functions (defined really x>0 in R)
# WELL, 'linear' would not exactly be 1st-order because we need the derivative and the value of the function
_extrapolation_right_kernel_dict = {
    'closest': lambda x, delta_sampling_points: 1.0,
}


######################################################
# Generator of inter+extrapolating function for a given position and delta between samples
######################################################

def _generate_interpolation_kernel(sampling_point, position,
                                   delta_sampling_points, delta_sampling_points_right=None,
                                   interpolation_type='linear', extrapolation_type='closest'
                                   ):
    """
    Generator of an inter+extrapolating function for a given position and delta between samples.
    It accepts spacing between adjacent samples different on the left and on the right: if 'delta_sampling_points_right'
    is not provided or 'None' the value provided for 'delta_sampling_points' applies to both sides.

    Parameters
    ----------
    sampling_point : float or int
         A 1-element `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ is also accepted
    position : str
        Value among ``'first'``, ``'middle'``, ``'last'``
    delta_sampling_points : float or int
        Separation to the next sampling point to the left, of to both left and right if ``delta_sampling_points_right``
        not provided.
    delta_sampling_points_right : float or int, optional
        Default: ``None``, i.e. same ``delta_sampling_points`` on both sides
    interpolation_type : str, optional
        Value among ``'closest'``, ``'linear'``, ``'raised_cosine'``, ``'sinc'``.
        Default: ``'linear'``
    extrapolation_type : str, optional
        Only accepted value is ``'closest'``. Linear extrapolation is already non-first-order,
        since the relationship between value and slope is needed.
        Default: ``'closest'``

    Returns
    -------
    ~collections.abc.Callable
    """

    assert (extrapolation_type == 'closest'), \
        f"Only 'extrapolation_type' == 'closest' considered, '{extrapolation_type}' found!"

    delta_left = delta_sampling_points
    delta_right = delta_sampling_points if delta_sampling_points_right is None else delta_sampling_points_right

    if position == 'first':
        if extrapolation_type == 'closest':
            fun = lambda x: torch.where(
                x < sampling_point,
                _extrapolation_right_kernel_dict['closest'](-(x - sampling_point), delta_left),
                _interpolation_kernel_dict[interpolation_type]((x - sampling_point), delta_right)
            )
        else:
            raise Exception(f"Invalid option {extrapolation_type} for 'extrapolation_type'!")
    elif position == 'middle':
        fun = lambda x: torch.where(
            x < sampling_point,
            _interpolation_kernel_dict[interpolation_type]((x - sampling_point), delta_left),
            _interpolation_kernel_dict[interpolation_type]((x - sampling_point), delta_right)
        )
    elif position == 'last':
        if extrapolation_type == 'closest':
            fun = lambda x: torch.where(
                x < sampling_point,
                _interpolation_kernel_dict[interpolation_type]((x - sampling_point), delta_left),
                _extrapolation_right_kernel_dict['closest']((x - sampling_point), delta_right)
            )
        else:
            raise Exception(f"Invalid option {extrapolation_type} for 'extrapolation_type'!")
    else:
        raise Exception(
            f"Argument 'position' given, '{position}', different from the valid 'first', 'middle' and 'last' options.")
    pass

    return fun


######################################################
# BUILDER OF SAMPLING POINTS AND CORRESPONDING INTER+EXTRAPOLATING FUNCTIONS
######################################################


def generate_builder_1_order_interpolation(
    num_sampling_points=101, nonuniform_sampling=True,
    start_range=-6.0, end_range=6.0,
    interpolation_type='linear', extrapolation_type='closest',
    activation_function=None, x_compress=1.0, y_stretch=1.0, x_offset=0.0, y_offset=0.0,
):
    """
    Auxiliary function that generates sampling points for first-order interpolating and their corresponding
    intra+extrapolation kernels for later usage in first-order function interpolation:

    - The non-uniform sampling is based on calculation of the $2{\\mathrm{nd}}$ derivative of the provided function,
    so more samples are generated where the $1{\\mathrm{st}}$ derivative changes the fastest.
    **In this case, dependent on the specific** ``activation_function'' **and on the modifiers** ``x_compress'',
    ``y_stretch'', ``x_offset'', **and** ``y_offset'', **said values would need to be also provided.**` In addition,
    **this method can potentially provide less sampling points than the requested** ``num_sampling_points'' if the
    performed analysis suggests that these samples are simply not necessary (e.g. for *ReLU* or for *hardtanh*).

    - The default sampling point extraction is uniform: it extracts uniformly spaced samples among the
    (default or specifically provided) values ``start_range`` and ``end_range`` (to be more precise, two samples
    are left to provide, one to the left and one to the right, far from the ends of the provided range to provide
    a linear-like extension of the activation function).

    Note that several of the most common activation functions, e.g. the sigmoid, have their biggest divergence from \
    a linear piece not around zero but at a certain distance from it: it seems therefore advisable to stick to the \
    uniform spacing, since the non-uniform version devotes most samples to the neighborhood of zero.

    Note also that the default values for ``start_range`` and ``end_range``, respectively ``-6.0`` and ``6.0``, \
    have been set with the sigmoid/tanh in mind, \
    since such functions is one of the most common activations and remains almost constant outside that range.

    Parameters
    ----------
    num_sampling_points : int, optional
        Number of equally spaced samples for the interpolation. An odd number would ensure that the value at zero
        is exactly the value of the activation function therein. Default: ``11``
    nonuniform_sampling : bool, optional
        ``False``, uniform sampling among ``start_range`` and ``end_range``; \
        ``True``, non-uniform sampling based on the tangent function. In this case, the specific ``activation_function''
        and its modifiers, if desired, would need to be provided. Default: ``True``
    start_range : float or int, optional
        Start of the \
        Default: ``-6.0``
    end_range : float or int, optional
        Default: ``6.0``
    interpolation_type : str, optional
        Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. Default: ``'linear'``
    extrapolation_type : str, optional
        Value among ``'closest'``. Default: ``'closest'``
    activation_function : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable], optional
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`. \
        Only used, and compulsory in such case, if ``nonuniform_sampling`` is ``True``.
        Default: ``None``
    x_compress, y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifiers to the basic activation function. Only used if ``nonuniform_sampling`` is ``True``. \
        Default: ``1.0``
    x_offset, y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Offset modifiers to the basic activation function. Only used if ``nonuniform_sampling`` is ``True``. \
        Default: ``0.0``

    Returns
    -------
    sampling_points : torch.Tensor
        1D tensor with the vector of points $a_p \\in \\mathbb{R}$, $p \\in {1,\\ldots,P}$, used as samples for the interpolation
    kernel_list : list[~collections.abc.Callable]
        List of callables, where the evaluation of each function gives the corresponding weight for each point in `sampling_points`
    """

    ### Computation device
    computation_device = 'cpu'
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for modifier in [x_compress, y_stretch, x_offset, y_offset]:
            if torch.is_tensor(modifier):
                computation_device = modifier.device.type
                break
            pass
        pass
    pass

    ### If input arguments are None, set to defaults
    start_range = -6.0 if start_range is None else start_range
    end_range = 6.0 if end_range is None else end_range
    num_sampling_points = 101 if num_sampling_points is None else num_sampling_points
    interpolation_type = 'linear' if interpolation_type is None else interpolation_type
    extrapolation_type = 'closest' if extrapolation_type is None else extrapolation_type
    # print(f"start_range, end_range = {start_range}, {end_range}\nnum_sampling_points = {num_sampling_points}")

    # We keep 2 points, aside, to provide basis for extrapolation to -Inf and +Inf
    num_electable_sampling_points = num_sampling_points - 2
    len_range = end_range - start_range

    ### First: get the sampling points using the method indicated by the flag "nonuniform_sampling"`
    if not nonuniform_sampling:
        mid_point_range = 0.5 * (end_range + start_range)
        mid_sampling_points = torch.linspace(start=start_range, end=end_range, steps=num_electable_sampling_points,
                                             device=computation_device)
    else:  # IF NON-UNIFORM SAMPLING
        # Parameters of the calculation: fixed, not left as parameters
        h = 1e-3
        rel_level_constant_1st_der = h  # "Relative" level of the 2nd der for considering that 1st der is constant
        # Resolve the activation function
        resolved_activation_function = ndim_activation_function_from_1dim_activation_functions(
            activation_function,
            x_compress=x_compress, y_stretch=y_stretch, x_offset=x_offset, y_offset=y_offset
        )
        # Setting the range where the 2nd order derivative is calculated
        # Calculation of the 2nd derivative

        vector_x_pre_post = torch.arange(start=start_range - h, end=end_range + h, step=h, device=computation_device)
        act_f_vector_x_pre_post = resolved_activation_function(vector_x_pre_post)
        #
        vector_x = vector_x_pre_post[1:-1]
        first_order_der_act_f_vector_x = 1 / h * (act_f_vector_x_pre_post[2:] - act_f_vector_x_pre_post[1:-1])
        second_order_der_act_f_vector_x = 1 / (h ** 2) * (
                act_f_vector_x_pre_post[2:] - 2 * act_f_vector_x_pre_post[1:-1] + act_f_vector_x_pre_post[0:-2]
        )
        # Procedure:
        # - we have the 2nd derivative at each point, at each and every point, of "vector_x"
        # - the cumulative of its abs has, always, the first weight at its [0], and 1. at its [-1]; in order to "detect" the change in the first
        #   one we would need to add a zero in the first position and then evaluate if the change from one index to the next one causes
        #   a change in the "bucket" corresponding to the F_uniform_samples.
        pdf_from_second_order_der_act_f_vector_x = torch.abs(second_order_der_act_f_vector_x)
        # Apply once the sqrt as a "lower temperature" for the sampling
        pdf_from_second_order_der_act_f_vector_x = torch.sqrt(pdf_from_second_order_der_act_f_vector_x)

        cum_f = torch.cumsum(pdf_from_second_order_der_act_f_vector_x, dim=-1)
        cum_f = cum_f / cum_f[-1]
        F_uniform_samples = torch.linspace(rel_level_constant_1st_der, 1.0 - rel_level_constant_1st_der,
                                           num_electable_sampling_points, device=computation_device)
        mid_sampling_points = torch.zeros((len(F_uniform_samples),))

        for ind in range(len(F_uniform_samples)):
            occurrence_bools = (cum_f[0:-1] < F_uniform_samples[ind]) & (cum_f[1:] >= F_uniform_samples[ind])
            occurrence_indices = torch.nonzero(occurrence_bools).flatten()
            if occurrence_indices is not None and len(occurrence_indices) > 0:
                mid_sampling_points[ind] = vector_x[occurrence_indices[0]]
            else:
                raise Exception(
                    f"Could not find a point in the provided range for the {ind}-th sampling point." +
                    f"check the provided activation function and its modifiers."
                )
            pass
        pass
        mid_sampling_points = torch.unique(mid_sampling_points)
    pass
    #
    # Add to far-away points at both ends
    sampling_points = torch.zeros(num_sampling_points, device=computation_device)
    sampling_points[0] = start_range - 1e3 * len_range
    sampling_points[1:-1] = mid_sampling_points
    sampling_points[-1] = end_range + 1e3 * len_range

    ### Second: get the kernel/interpolating function corresponding to each sampling point a_p
    # for the given "interpolation_type" and "extrapolation_type"
    kernel_list = []
    for ind in range(num_sampling_points):
        position = 'middle'
        if ind == 0:
            position = 'first'
            delta_sampling_point_left = sampling_points[ind + 1] - sampling_points[ind]
            delta_sampling_point_right = None
        elif ind == num_sampling_points - 1:
            position = 'last'
            delta_sampling_point_left = sampling_points[ind] - sampling_points[ind - 1]
            delta_sampling_point_right = None
        else:
            position = 'middle'
            delta_sampling_point_left = sampling_points[ind] - sampling_points[ind - 1]
            delta_sampling_point_right = sampling_points[ind + 1] - sampling_points[ind]
        pass
        kernel_list.append(
            _generate_interpolation_kernel(
                sampling_point=sampling_points[ind],
                position=position,
                delta_sampling_points=delta_sampling_point_left, delta_sampling_points_right=delta_sampling_point_right,
                interpolation_type=interpolation_type, extrapolation_type=extrapolation_type
            )
        )
    pass

    return sampling_points, kernel_list


######################################################
# BUILDER OF SAMPLING POINTS AND CORRESPONDING INTER+EXTRAPOLATING FUNCTIONS
# IT SEEMS THAT THE ACCURACY OF THE APPROXIMATION DEPENDS ON THE INPUT VALUES THAT SIGMA WILL COVER, AND
# OF THE MAXIMUM SLOPE OF THE SIGMA FUNCTION (including p=x_compress): THIS FUNCTION TRIES TO ADAPT THE
# NUMBER OF SAMPLING POINTS TO THE SPECIFIC ACTIVATION FUNCTION AND INPUT VALUES
######################################################


def generate_builder_1_order_interpolation_for_input(
    input_info,
    num_sampling_points=11, interpolation_type='linear', **kwargs
):
    """
    Auxiliary function that generates sampling points for first-order interpolating and their corresponding
    intra+extrapolation kernels for later usage in first-order function interpolation:

    - The non-uniform sampling is based on calculation of the $2{\\mathrm{nd}}$ derivative of the provided function,
    so more samples are generated where the $1{\\mathrm{st}}$ derivative changes the fastest.
    **In this case, dependent on the specific** ``activation_function'' **and on the modifiers** ``x_compress'',
    ``y_stretch'', ``x_offset'', **and** ``y_offset'', **said values would need to be also provided.**` In addition,
    **this method can potentially provide less sampling points than the requested** ``num_sampling_points'' if the
    performed analysis suggests that these samples are simply not necessary (e.g. for *ReLU* or for *hardtanh*).

    - The default sampling point extraction is uniform: it extracts uniformly spaced samples among the
    (default or specifically provided) values ``start_range`` and ``end_range`` (to be more precise, two samples
    are left to provide, one to the left and one to the right, far from the ends of the provided range to provide
    a linear-like extension of the activation function).

    Note that several of the most common activation functions, e.g. the sigmoid, have their biggest divergence from \
    a linear piece not around zero but at a certain distance from it: it seems therefore advisable to stick to the \
    uniform spacing, since the non-uniform version devotes most samples to the neighborhood of zero.

    Note also that the default values for ``start_range`` and ``end_range``, respectively ``-6.0`` and ``6.0``, \
    have been set with the sigmoid/tanh in mind, \
    since such functions is one of the most common activations and remains almost constant outside that range.

    Parameters
    ----------
    input_info : tuple[int or float, int or float] or torch.Tensor
        Information regarding the range of input values relevant to the activation function. \
        The information can be provided in two alternative manners: *(i)* either as a tuple \
        with the minimum and maximum values of the input, or *(ii)* with the full tensor that will be processed \
        (so the minimum and maximum are in fact extracted from it)
    num_sampling_points : int, optional
        Number of equally spaced samples for the interpolation. An odd number would ensure that the value at zero
        is exactly the value of the activation function therein. Default: ``11``
    interpolation_type : str, optional
        Value among ``'linear'``, ``'closest'``, ``'raised_cosine'``, ``'sinc'``. Default: ``'linear'``
    kwargs : dict
        In fact, for now, no extra arguments are considered. The dictionary is kept for future compatibility and \
        to avoid errors when the function is called with extra arguments. Default: ``{}``

    Returns
    -------
    sampling_points : torch.Tensor
        1D tensor with the vector of points $a_p \\in \\mathbb{R}$, $p \\in {1,\\ldots,P}$, used as samples for the interpolation
    kernel_list : list[~collections.abc.Callable]
        List of callables, where the evaluation of each function gives the corresponding weight for each point in `sampling_points`
    """

    ### Extract the range of relevant input values
    min_input = 0.0
    max_input = 0.0
    if isinstance(input_info, tuple):
        if len(input_info) != 2 or not all([isinstance(x, (int, float)) for x in input_info]):
            raise Exception(f"Expected a tuple with 2 float/int elements, but found {len(input_info)} elements " +
                            f"of type {[type(x) for x in input_info]}")
        else:
            min_input, max_input = input_info
        pass
    elif torch.is_tensor(input_info):
        if input_info.dim() == 0:
            raise Exception("The provided tensor is a scalar, not a tensor with data.")
        min_input = torch.min(input_info.detach()).item()
        max_input = torch.max(input_info.detach()).item()
    else:
        raise Exception(f"Expected a tuple or a tensor, but found {type(input_info)}")
    pass

    ### Generate the sampling points
    sampling_points = torch.linspace(start=min_input, end=max_input, steps=num_sampling_points)

    ### Second: get the kernel/interpolating function corresponding to each sampling point
    kernel_list = []
    for ind in range(num_sampling_points):
        position = 'middle'
        if ind == 0:
            position = 'first'
            delta_sampling_point_left = sampling_points[ind + 1] - sampling_points[ind]
            delta_sampling_point_right = None
        elif ind == num_sampling_points - 1:
            position = 'last'
            delta_sampling_point_left = sampling_points[ind] - sampling_points[ind - 1]
            delta_sampling_point_right = None
        else:
            position = 'middle'
            delta_sampling_point_left = sampling_points[ind] - sampling_points[ind - 1]
            delta_sampling_point_right = sampling_points[ind + 1] - sampling_points[ind]
        pass
        kernel_list.append(
            _generate_interpolation_kernel(
                sampling_point=sampling_points[ind],
                position=position,
                delta_sampling_points=delta_sampling_point_left, delta_sampling_points_right=delta_sampling_point_right,
                interpolation_type=interpolation_type
            )
        )
    pass

    return sampling_points, kernel_list



######################################################
# Generation of a function approximating, using first-order interpolation of activation function
######################################################

def generate_1_order_interpolation_activation_function_for_given_u_i(
        u_i,
        activation_function, x_compress=1.0, y_stretch=1.0, x_offset=0.0, y_offset=0.0,
        num_sampling_points=None, nonuniform_sampling=None,
        start_range_std_activation=-6.0, end_range_std_activation=+6.0,
        interpolation_type=None, extrapolation_type=None,
):
    """
    Generation of a function approximating, using first-order interpolation of activation functions, the function
    $$ \\sigma(\\cdot - u_i): \\mathbb{R} \\to \\mathbb{R}$$
    for the given $u_i$ and activation function $\\sigma$.

    Parameters
    ----------
    u_i : float or int
        The value $u_i$ for which the interpolation is to be done.
    activation_function : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`
    x_compress, y_stretch : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Scale modifiers to the basic activation function. Only used if ``nonuniform_sampling`` is ``True``. \
        Default: ``1.0``
    x_offset, y_offset : int or float or list[int or float] or tuple[int or float] or torch.Tensor, optional
        Offset modifiers to the basic activation function. Only used if ``nonuniform_sampling`` is ``True``. \
        Default: ``0.0``
    num_sampling_points : int
        Number of equally spaced samples for the interpolation. An odd number would ensure that the value at zero
        is exactly the value of the activation function therein.
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    nonuniform_sampling : bool, optional
        ``False``, uniform sampling among ``start_range`` and ``end_range``; ``True``, non-uniform sampling
        based on the tangent function. Default: see :py:func:`.generate_builder_1_order_interpolation`
    start_range_std_activation, end_range_std_activation : float or int
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    interpolation_type : {'closest', 'linear', 'raised_cosine', 'sinc'}
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    extrapolation_type : {'closest'}
        Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    interpolated_activation_function : `callable <https://docs.python.org/3/glossary.html#term-callable>`_
        Interpolated approximation for the input `activation_function`.
    """

    # First, define the callable function from the provided argument
    activation_f = ndim_activation_function_from_1dim_activation_functions(
        activation_function, x_compress=x_compress, y_stretch=y_stretch, x_offset=x_offset, y_offset=y_offset
    )

    # Second: generate the builder of the 1st-order interpolation using the auxiliary function returning:
    # - the sampling points
    # - the kernel function for each sampling point
    # and calculate the image, through the chosen function, of the sampling points

    start_range = start_range_std_activation/x_compress + x_offset
    end_range = end_range_std_activation/x_compress + x_offset

    sampling_points, kernel_list = generate_builder_1_order_interpolation(
        num_sampling_points=num_sampling_points, nonuniform_sampling=nonuniform_sampling,
        start_range=start_range, end_range=end_range,
        interpolation_type=interpolation_type, extrapolation_type=extrapolation_type,
        activation_function=activation_function,
        x_compress=x_compress, y_stretch=y_stretch, x_offset=x_offset, y_offset=y_offset
    )

    # Third: interpolating function from the sampling points (constant outside)
    # WARNING: OUTSIDE THE GIVEN LIMITS, IN THE CURRENT CONFIGURATION, THE OUTPUT IS CONSTANT!
    def interpolated_activation_function(u_j):
        #
        # Generate the interpolating factors, based on the kernel list
        k_s_u_i = torch.tensor([kernel(u_i) for ind, kernel in enumerate(kernel_list)])
        #
        activation_f_sampling_points = activation_f(sampling_points)
        #
        # sigma(u_j - a(s)) for all possible s, and getting that to first dimension
        sigma_u_j_minus_a_s = torch.zeros((len(sampling_points),)+u_j.size(), device=u_j.device)
        for s in range(len(sampling_points)):
            sigma_u_j_minus_a_s[s] = activation_f(u_j - sampling_points[s])
        pass
        # "Replicate" "k_s_ui", multiply, and sum
        k_s_u_i = k_s_u_i.to(u_j.device)
        k_s_ui_replicated = k_s_u_i.view((len(sampling_points),)+(1,)*u_j.ndim).repeat((1,)+u_j.size())
        interpolated_f_u_j = (k_s_ui_replicated * sigma_u_j_minus_a_s).sum(0)
        #
        return interpolated_f_u_j

    return interpolated_activation_function

######################################################
# Generation of a function approximating, using first-order interpolation of activation function
######################################################

def generate_1_order_interpolation_activation_function(
        activation_function, num_sampling_points=None, nonuniform_sampling=None,
        start_range=None, end_range=None,
        interpolation_type=None, extrapolation_type=None
):
    """
    Generation of a function approximating, using first-order interpolation of activation function \
    $\\sigma: \\mathbb{R} \\to \\mathbb{R}$, that is, a linear combination of conveniently scaled and \
    displaced versions of an interpolant function $\\Pi: \\mathbb{R} \\to \\mathbb{R}$ so
    $$\\sigma(x) \\approx \\sum_{k} \\sigma (k\Delta) \, \\Pi \Big(\\frac{x-k\Delta}{\Delta}\Big) $$ .
    Different interpolant functions are considered, with the default being a triangular interpolant, resulting in
    the traditional linear interpolation.
    The returned interpolation function is (internally) based on a finite number of equally spaced sampling points
    between the given start and end values: the extrapolation outside those points is performed, by default, using
    linear extrapolation.

    Parameters
    ----------
    activation_function : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`
    num_sampling_points : int
        Number of equally spaced samples for the interpolation. An odd number would ensure that the value at zero
        is exactly the value of the activation function therein.\
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    nonuniform_sampling : bool, optional
        ``False``, uniform sampling among ``start_range`` and ``end_range``; ``True``, non-uniform sampling
        based on the tangent function. Default: see :py:func:`.generate_builder_1_order_interpolation`
    start_range, end_range : float or int
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    interpolation_type : {'closest', 'linear', 'raised_cosine', 'sinc'}
        Default: see :py:func:`.generate_builder_1_order_interpolation`
    extrapolation_type : {'closest'}
        Default: see :py:func:`.generate_builder_1_order_interpolation`

    Returns
    -------
    interpolated_activation_function : `callable <https://docs.python.org/3/glossary.html#term-callable>`_
        Interpolated approximation for the input `activation_function`.
    """

    # First, define the callable function from the provided argument
    activation_f = _resolve_activation_function(activation_function)

    # Second: generate the builder of the 1st-order interpolation using the auxiliary function returning:
    # - the sampling points
    # - the kernel function for each sampling point
    # and calculate the image, through the chosen function, of the sampling points
    sampling_points, kernel_list = generate_builder_1_order_interpolation(
        num_sampling_points=num_sampling_points, nonuniform_sampling=nonuniform_sampling,
        start_range=start_range, end_range=end_range,
        interpolation_type=interpolation_type, extrapolation_type=extrapolation_type,
        activation_function=activation_f
    )
    activation_f_sampling_points = activation_f(sampling_points)

    # Third: interpolating function from the sampling points (constant outside)
    # WARNING: OUTSIDE THE GIVEN LIMITS, IN THE CURRENT CONFIGURATION, THE OUTPUT IS CONSTANT!
    def interpolated_activation_function(x):
        size_required_repetition_of_x = tuple(sampling_points.size()) + tuple(np.ones(x.ndim, dtype=int))
        expanded_kernelized_x = x.unsqueeze(0).repeat(size_required_repetition_of_x)

        size_required_review_of_sampling_points = tuple(sampling_points.size()) + tuple(np.ones(x.ndim, dtype=int))
        size_required_repetition_of_sampling_points = tuple(np.ones(sampling_points.ndim, dtype=int)) + tuple(x.size())

        expanded_activation_f_sampling_points = activation_f_sampling_points.view(
            size_required_review_of_sampling_points
        ).repeat(
            size_required_repetition_of_sampling_points
        )

        for ind in range(len(kernel_list)):
            expanded_kernelized_x[ind, :] = kernel_list[ind](expanded_kernelized_x[ind, :])

        interpolated_f_x = (
                expanded_kernelized_x *
                expanded_activation_f_sampling_points
        ).sum(0)

        return interpolated_f_x

    return interpolated_activation_function


######################################################
######################################################
#
# CONVOLUTION FUNCTIONS
#
######################################################
######################################################


######################################################
# MODIFIED torch.nn.functional.conv2d TO INCLUDE THE OPTION padding_mode
######################################################

def cast_scalar_like_to_image(scalar_like, size_target_tensor):
    """
    It transforms a tensor ``scalar_like``, representing either a scalar or a vector, to the size \
    ``size_target_tensor`` of the tensor it will be applied to in an element-wise manner. Therefore: \
    the possibility of casting given the size of ``scalar_like`` and ``size_target_tensor`` is analyzed;
    and ``scalar_like`` will be replicated/reformatted so the common operations (+, -, \\*, /) would
    perform a univocal casting with the resulted casted ``scalar_like`` and target tensor.

    Parameters
    ----------
    scalar_like : torch.Tensor
    size_target_tensor : torch.Tensor.size or tuple

    Returns
    -------
    torch.Tensor
    """

    # Now we cast the bias if necessary and add it separately
    casted_scalar_like = None

    if scalar_like.numel() == 1:
        casted_scalar_like = scalar_like
    elif scalar_like.numel() == size_target_tensor[-3]:
        if scalar_like.ndim == 1:  # It means it is simply a vector with "C_out" elements
            casted_scalar_like = scalar_like.unsqueeze(-1).unsqueeze(-1)
        elif (scalar_like.ndim == 3) and (scalar_like.size(-1) == 1) and (scalar_like.size(-2) == 1) and \
                (scalar_like.size(-3) == size_target_tensor[-3]):
            casted_scalar_like = scalar_like
        else:
            raise Exception(
                (f"It is not clear how to cast 'bias', of size {scalar_like.size()}, " +
                 f"to 'C_output' = {size_target_tensor[-3]}!")
            )
        pass
        casted_scalar_like = casted_scalar_like.repeat((1, size_target_tensor[-2], size_target_tensor[-1]))
    elif (scalar_like.size(-1) == size_target_tensor[-1]) and (scalar_like.size(-2) == size_target_tensor[-2]) and \
            (scalar_like.size(-3) == size_target_tensor[-3]) and (scalar_like.ndim == 3):
        casted_scalar_like = scalar_like
    else:
        raise Exception(
            (f"It is not clear how to cast 'bias']', of size {scalar_like.size()}, to the output size " +
             f"({size_target_tensor[-3]}, {size_target_tensor[-2]}, {size_target_tensor[-1]})!")
        )
    pass

    return casted_scalar_like

######################################################
# Adaptation of the 2D convolution to include padding_mode and multi-channel bias
######################################################

def conv2d_adapted(input, weight, bias=None, stride=1,
                   padding='same', padding_mode='zeros', groups=1):
    """
    Identical to `torch.nn.functional.conv2d <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>`_
    but for the following aspects:

    **Additions/modifications with respect to** \
    `torch.nn.functional.conv2d <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>`_:

    - This function includes the ``padding_mode`` option denied by the former (and included in class \
      `torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_. \
      The inclusion of various ``padding_mode`` options has been included using a padding function.

    - This function accepts, not only a scalar ``bias``, but also multi-channel bias corresponding to the output \
      of the function.

    **Restrictions with respect to** \
    `torch.nn.functional.conv2d <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>`_:

    - ``padding`` accepts only ``'valid'`` and ``'same'``, with the latter as default.

    - The argument ``dilation`` is not included (always 1, no *convolution à trous*).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape $(\\mathrm{minibatch}, \\mathrm{in\\_channels}, i H, i W)$ \
        or $(\\mathrm{in\\_channels}, i H, i W)$
    weight : torch.Tensor
        Filters of shape $(\\mathrm{out_channels}, \\frac{\\mathrm{in\\_channels}}{\\mathrm{in_channels}}, k H, k W)$
    bias : torch.Tensor, optional
        Bias tensor of shape $1$ or $(\\mathrm{out\\_channels})$ (one per output channel). Default: ``None``
    stride : int or tuple[int], optional
        The stride of the convolving kernel. Can be a single number or a tuple $(s_H, s_W)$. Default: ``1``
    padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``, indicating the implicit paddings on both sides of the input. Unlike in
        :py:func:`torch.nn.functional.conv2d` the only accepted values are ``‘same’` and ``‘valid’``:
        padding ``'valid'`` is the same as no padding; padding ``'same'`` pads the input so the output has the same shape
        as the input, but this mode does not support any stride values other than 1. Default: ``‘same’``
    padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``

    Returns
    -------
    torch.Tensor
    """

    ################
    # Check whether operations should be pushed into GPU: based on availability and based on 'weight'
    ################
    computation_device = 'cuda' \
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and \
           (weight.device.type == 'cuda' or input.device.type == 'cuda') \
        else 'cpu'
    computation_device = 'mps' if torch.backends.mps.is_available() else computation_device
    if input is not None and computation_device != input.device.type:
        input = input.to(computation_device)
        # print(f"Device 'input' = {input.device.type}")
    if weight is not None and computation_device != weight.device.type:
        weight = weight.to(computation_device)
        # print(f"Device 'weight' = {weight.device.type}")
    ################

    ################
    # Checks on the input arguments
    ################
    assert padding in ['same', 'valid'], f"Only valid options for ``padding`` are 'same' and 'valid', {padding} found!"
    assert (isinstance(stride, int)) or (isinstance(stride, tuple) and (len(stride) == 2)), \
        f"Invalid 'stride' provided: only 'int' or 2-element 'tuple' accepted, {stride} found!"
    stride_tuple = (stride, stride) if isinstance(stride, int) else stride
    for i in range(len(stride_tuple)):
        assert stride_tuple[i] > 0, f"Stride values must be positive int, {stride} given!"
        assert (padding == 'valid') or (stride_tuple[i] == 1), \
            f"Padding='same' is not supported for strided convolutions (stride {stride} found)!"
    assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular'], \
        f"Only valid options for ``padding_mode`` are 'same' and 'valid', {padding_mode} found!"
    ################

    # Padding input if necessary and as necessary
    if padding == 'same':
        H, W = weight.size(-2), weight.size(-1)
        upper_pad, bottom_pad, left_pad, right_pad = floor((H - 1) / 2.), ceil((H - 1) / 2.), floor((W - 1) / 2.), ceil(
            (W - 1) / 2.)
        if padding_mode == 'zeros':
            padded_input = F.pad(input, pad=(left_pad, right_pad, upper_pad, bottom_pad), mode='constant', value=0)
        else:  # 'reflect', 'replicate' or 'circular'
            padded_input = F.pad(input, pad=(left_pad, right_pad, upper_pad, bottom_pad), mode=padding_mode)
        pass
    else:  # 'valid'
        padded_input = input
    pass

    # Convolve and return the padded input with the option 'valid'!!! AND NO BIAS!
    convolved_result = F.conv2d(
        input=padded_input, weight=weight, bias=None, stride=stride, padding='valid', dilation=1, groups=groups)

    # Now we cast the bias if necessary and add it separately
    # casted_bias = None
    # if bias.numel() == 1:
    #     casted_bias = bias
    # elif bias.numel() == convolved_result.size(-3):
    #     if bias.ndim == 1:
    #         casted_bias = bias.unsqueeze(-1).unsqueeze(-1)
    #     elif (bias.ndim == 3) and (bias.size(-1) == 1) and (bias.size(-2) == 1):
    #         casted_bias = bias
    #     else:
    #         raise Exception(
    #             (f"It is not clear how to cast 'bias', of size {bias.size()}, " +
    #              f"to 'C_output' = {convolved_result.size(-3)}!")
    #         )
    #     pass
    #     casted_bias = casted_bias.repeat((1, convolved_result.size(-2), convolved_result.size(-1)))
    # else:
    #     raise Exception(
    #         (f"It is not clear how to cast 'bias']', of size {bias.size()}, to the output size " +
    #          f"({convolved_result.size(-3)}, {convolved_result.size(-2)}, {convolved_result.size(-1)})!")
    #     )
    # pass

    final_convolved_result = convolved_result if bias is None else \
        convolved_result + cast_scalar_like_to_image(bias, convolved_result.size())

    return final_convolved_result


######################################################
# Creation of a FC layer that considers image-like formatting and groups
######################################################


def fc_2d(input, weight, bias, groups=1):
    """
    Similar to the function `torch.nn.functional.linear <https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html>`_
    but taking into account that the `input` is image-like, that is, with shape $(B, C_{in}, H_{in}, W_{in})$, \
    and that the `weight` is formatted accordingly to provide an output with shape $(B, C_{out}, H_{out}, W_{out})$ \
    given by the structure of the `weight` tensor and the grouping given by `groups`. At this respect, \
    the rules regarding channels and groups are identical to those of \
    `torch.nn.functional.conv2d <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>`_.

    Regarding the details of the sizes of the tensors:

    - The spatial size $(H_{out}, W_{out})$ of the output is not directly encoded within `weight`, or not there alone. \

      - The matrix `weight` is a 4D tensor, where \
        $(C_{out}, \\frac{C_{out}}{\\textrm{groups}}, W_{w}, H_{w})$, and wherein each channel $c$ of the output \
        image results from the application of the matrix `weight`[c] \
        (of size $(\\frac{C_{out}}{\\textrm{groups}}, W_{w}, H_{w})$) to the corresponding group of \
        $\\frac{C_{out}}{\\textrm{groups}}$ channels of the input image. Therefore, the size of `weight` is \
        $(C_{out}, \\frac{C_{in}}{\\textrm{groups}}, H_{out} \\times W_{out}, H_{in} \\times W_{in})$ so \
        the channel $c$ of the output performs a linear operation between the \
        $\\frac{C_{in}}{\\textrm{groups}} \\times H_{in} \\times W_{in}$ values of the corresponding \
        channels of the input image and the corresponding $H_{out} \\times W_{out}$ values of the corresponding \
        channel of the output.

      - The above operation, however, does not keep track of the aspect ratio of the output, since the only \
        information kept in `weight` is the product $H_{out} \\times W_{out}$ but not their specific value. \
        The vector/offset `bias` will contain that information, and will be sized as \
        $(C_{out}, H_{out}, W_{out})$.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape $(B, C_{in}, H_{in}, W_{in})$ \
        or $(C_{in}, H_{in}, W_{in})$
    weight : torch.Tensor
        4D-tensor of size $(C_{out}, \\frac{C_{in}}{\\textrm{groups}}, H_{out} \\times W_{out}, H_{in} \\times W_{in})$
    bias : torch.Tensor
        Bias 3D-tensor of shape $(C_{out}, H_{out}, W_{out})$
    groups : int, optional
        Split input into groups. Both $C_{in}$ and $C_{out}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``

    Returns
    -------
    torch.Tensor
    """

    ################
    # Check whether operations should be pushed into GPU: based on availability and based on 'weight'
    ################
    computation_device = 'cuda' \
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and \
           (weight.device.type == 'cuda' or input.device.type == 'cuda') \
        else 'cpu'
    computation_device = 'mps' if torch.backends.mps.is_available() else computation_device
    if input is not None and computation_device != input.device.type:
        input = input.to(computation_device)
        # print(f"Device 'input' = {input.device.type}")
    if weight is not None and computation_device != weight.device.type:
        weight = weight.to(computation_device)
        # print(f"Device 'weight' = {weight.device.type}")
    ################

    ################
    # Checks on the input arguments
    ################
    # Types
    dict_compulsory_args = {'input': input, 'weight': weight, 'bias': bias}
    for key_compulsory_arg, value_compulsory_arg in dict_compulsory_args.items():
        if not isinstance(value_compulsory_arg, torch.Tensor):
            raise ValueError((f"The compulsory argument '{key_compulsory_arg}' must be a Tensor; " +
                              f"found {type(value_compulsory_arg)} instead!"))
        pass
    pass
    assert isinstance(groups, int) and groups > 0, \
        f"The argument 'groups' must be a positive integer, {groups} found!"
    #
    # Dimensions and groups (wrt. dimensions)
    assert bias.ndim == 3, f"The 'bias' must be a 3D tensor, {bias.ndim}D found!"
    assert weight.ndim == 4, f"The 'weight' must be a 4D tensor, {weight.ndim}D found!"
    assert input.ndim in [3, 4], \
        f"The 'input' must be a 3D or 4D tensor, {input.ndim}D found!"
    C_in, H_in, W_in = input.size(-3), input.size(-2), input.size(-1)
    C_out_from_bias, H_out_from_bias, W_out_from_bias = bias.size(-3), bias.size(-2), bias.size(-1)
    C_out_from_weight = weight.size(-4)
    assert C_out_from_bias == C_out_from_weight, \
        (f"The number of output channels inferred from 'bias' ({C_out_from_bias}) does not match " +
         f"the one inferred from 'weight' ({C_out_from_weight})!")
    H_in_x_W_in = H_in * W_in
    H_out_from_bias_x_W_out_from_bias = H_out_from_bias * W_out_from_bias
    assert weight.size(-1) == H_in_x_W_in, \
        (f"The spatial size of the weight kernel {weight.size(-1)} is different from " +
         f"the product {H_in_x_W_in} of the input spatial dimensions ({H_in}, {W_in})!")
    assert weight.size(-2) == H_out_from_bias_x_W_out_from_bias, \
        (f"The spatial size of the weight kernel {weight.size(-2)} is different from " +
         f"the product {H_out_from_bias_x_W_out_from_bias} of the output spatial dimensions " +
         f"({H_out_from_bias}, {W_out_from_bias}) (taken from the bias)!")
    ################

    ################
    # Cast the input image to 4D if 3D, and indicate if so performed to revert it later
    ################
    # Transform to 4D in any case
    casted_images = False
    if input.ndim == 3:
        casted_images = True
        input = input.unsqueeze(0)
    pass
    B, C_out, H_out, W_out = input.size(-4), C_out_from_bias, H_out_from_bias, W_out_from_bias
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    ################

    ################
    # Operation
    # F.linear takes as input (N, *, in_features) and weight (out_features, in_features)
    ################
    output = torch.empty((B, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    # We process group by group
    for g in range(groups):
        input_g = input[:, g*C_in_per_group:(g+1)*C_in_per_group, :, :]
        input_g_reshaped = input_g.view(B, C_in_per_group * H_in * W_in)
        for c_o in range(C_out_per_group):
            abs_c_o = g * C_out_per_group + c_o
            weight_abs_c_o = weight[abs_c_o, :, :, :]
            bias_abs_c_o = bias[abs_c_o, :, :]
            # To reshape 'weight_abs_c_o' to (H_out * W_out, C_in_per_group * H_in * W_in) keeping the \
            # "channel contiguity" we have to move the dimensions around
            weight_abs_c_o_reshaped = weight_abs_c_o.transpose(-3, -2).reshape(H_out * W_out, C_in_per_group * H_in * W_in)
            # We reshape the bias too (to flat)
            bias_abs_c_o_reshaped = bias_abs_c_o.reshape(H_out * W_out)
            output_g_c_o_reshaped = F.linear(
                input=input_g_reshaped, weight=weight_abs_c_o_reshaped, bias=bias_abs_c_o_reshaped
            )
            output_g_c_o = output_g_c_o_reshaped.reshape(B, H_out, W_out)
            # Assign to output
            output[:, abs_c_o, :, :] = output_g_c_o
        pass

    ################
    # Un-cast the output, if the input was casted
    ################

    if casted_images:
        output = output[0]
    pass

    return output


######################################################
# Cross-difference convolution performed by the ibnn_internal/INRF
######################################################


def conv2d_crossdiff(input, weight=None, reference=None, activation_function=None,
                     stride=1, padding='same', padding_mode='zeros', groups=1,
                     calculation_mode='interpolated', memory_saving_version=True,
                     **kwargs
                     ):
    """
    Calculation of the expression
    $$
    \\Big(
    \\Omega \\ast \\sigma(\\mathbf{v}-\\mathbf{u}[\\mathbf{p}])
    \\Big)
    (\\mathbf{p}) \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
    $$
    wherein both images $\\mathbf{u}$ and $\\mathbf{v}$ have the same size, and
    wherein $\\mathbf{u}$ represents the ``input`` and
    the image $\\mathbf{v}$ represents its ``reference``, since the calculation
    compares each pixel value of the former with the values of the neighbouring pixels of the latter.

    When no ``reference`` is provided the function considers $\\mathbf{v}=\\mathbf{u}$
    (i.e. $u$ is compared to itself), which represents the implicit function basis for the :py:class:`.IBNNInternalLayer`.

    The images $\\mathbf{u}$ and $\\mathbf{v}$ are regarded as composed of multiple channels $C_{in}$, that is,
    $$
    \\mathbf{u}, \\mathbf{v}: \\mathcal{P}_u\\!\\subset\\!\\mathbb{N}^{2} \\to \\mathbb{R}^{C_{in}} \\, \\, ,
    $$
    and therefore ``input`` and ``reference`` are
    `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_
    of size $(B,C_{in},H_{in},W_{in})$. The size of the output $(B,C_{out},H_{out},W_{out})$,
    of the filter $(C_{out},\\frac{C_{in}}{\\mathrm{groups}},H_{\\Omega},W_{\\Omega})$, and the relationship between
    channels of input images, filter weights, ``groups``, and output correspond to the same relationships allowed by
    :py:func:`.conv2d_adapted`, basis function for the convolution operations performed herein.

    **--- IMPORTANT NOTE: ----------------------------------------------------------------------**

    **Filter kernels** $\\Omega$ **without spatial extent**, that is, 2D tensors \
    $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, \
    are considered to **represent a uniform value for the whole extent of the image. This behavior occurs** \
    **in this function** :py:func:`.conv2d_crossdiff` **only; the function** :py:func:`.conv2d_adapted` \
    **would fail for an input filter of such size.** \
    **In the case of these full-image uniform kernels, and only in that case,** :py:func:`.conv2d_crossdiff` \
    **is insensitive to the value of the following arguments, irrelevant for its operation:**

    - the options ``padding`` and ``padding_mode``, irrespective of the values provided, do not apply: the result \
      of the operation will always be of the same size as the input images, and with no value padding applied; and

    - the algorithm for the calculation of the convolution is simplified with respect to the case of kernels with \
      defined spatial extent.

    **-----------------------------------------------------------------------------------------------------------**

    The activation function acts separably on each of the dimensions of the input, that is,
    $$
    \\bar{\\sigma}(\\mathbf{u}) = \\Big(\\ldots, \\sigma_c(\\mathbf{u}), \\ldots \\Big) \\,\\, ,
    \\mathrm{where } \\,\\, \\sigma_c(u_c) \\,\\,\\, \\forall c \\in \\{1,\\ldots, C_{in} \\}
    \\, \\, .
    $$
    All $\\sigma_c(\\cdot)$ can be either a 1D function $\\sigma_c(\\cdot)=\\sigma(\\cdot)$,
    indicated with one single 1D ``activation_function``
    acting equally on each component; or each $\\sigma_c(\\cdot)$ can be defined as having a different behavior,
    indicated with a separable $C_{in}$-dimensional ``activation_function``
    defined e.g. using :py:func:`.ndim_activation_function_from_1dim_activation_functions`.

    Since this function is based on the function :py:func:`.conv2d_adapted`, and like the latter, it has two
    restrictions with respect to the usual options of
    `torch.nn.functional.conv2d <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>`_,
    to wit:

    - ``padding`` accepts only 'valid' and 'same', with the latter as default, and
    - the argument ``dilation`` is not included (always 1, no *convolution à trois* accepted).

    Parameters
    ----------
    input : torch.Tensor
        Tensor of size $(B,C_{in},H_{in},W_{in})$ or $(C_{in},H_{in},W_{in})$
    weight : torch.Tensor
        Description of the filter kernel $\\Omega$. Two possibilities:

        - *(1)* a 4D tensor, whose size is $(C_{out},\\frac{C_{in}}{\\mathrm{groups}},H_{\\Omega},W_{\\Omega})$, \
        corresponding to standard, displacing, convolution-like kernels; or

        - *(2)* a 2D tensor, whose size is \
        $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$, missing the spatial dimensions, which will understood \
        as a uniform $\\Omega$ constant for the whole extent of the image
    reference : torch.Tensor, optional
        Tensor of same size of ``input``
    activation_function : str or ~collections.abc.Callable or list[str] or list[~collections.abc.Callable]
        Same specifications of :py:func:`.ndim_activation_function_from_1dim_activation_functions`
    stride : int or tuple[int], optional
        The stride of the convolving kernel. Can be a single number or a tuple $(s_H, s_W)$. Default: ``1``
    padding : str, optional
        Value among  ``‘same’`` and ``‘valid’``, indicating the implicit paddings on both sides of the input. Unlike in
        :py:func:`torch.nn.functional.conv2d` the only accepted values are ``‘same’` and ``‘valid’``:
        padding ``'valid'`` is the same as no padding; padding ``'same'`` pads the input so the output has the same shape
        as the input, but this mode does not support any stride values other than 1. Default: ``‘same’``
    padding_mode : str, optional
        Value among ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'zeros'``
    groups : int, optional
        Split input into groups. Both $\\mathrm{in_channels}$ and $\\mathrm{out_channels}$ should be divisible by the
        number of groups; identical to that in :py:func:`torch.nn.functional.conv2d`.
        Default: ``1``
    calculation_mode : str, optional
        Value among ``'interpolated'``, ``'n4'``. Use of interpolated approximations of the  activation functions, \
        based on the utilities provided by :py:func:`.generate_builder_1_order_interpolation`, keeping the complexity \
        of the underlying calculations in the order of $S \\times N^2$ (for $N$ the side of the input image(s) \
        and $S$ the number of interpolation points, and disregarding the contribution of the convolution kernel), \
        or using the exact calculation of complexity in the order of $N^4$. \
        Default: ``'interpolated'``
    memory_saving_version : bool, optional
        For the option ``calculation_mode`` = ``'interpolated'`` two computation versions are implemented: one, \
        corresponding to ``memory_saving_version`` ``True``, prioritizes memory occupancy over speed; \
        the other, corresponding ``False``, prioritizes speed. As a guidance, although values depends on the inputs, \
        the difference between both versions in both aspects lies around 30%. Default: ``True``
    **kwargs : optional
        These keyword refer to the following arguments of two functions used internally by
        this function.

        - The arguments admitted by the internally used function \
        :py:func:`.ndim_activation_function_from_1dim_activation_functions` (see therein for detailed information) are:

            - **x_compress**, **y_stretch**, **x_offset**, **y_offset** : torch.Tensor, int, float, list[int/float], \
            or tuple[int/float], optional

                Default: see :py:func:`.ndim_activation_function_from_1dim_activation_functions`

        - The arguments admitted by the internally used function \
        :py:func:`.generate_builder_1_order_interpolation` (see therein for detailed information) are:

            - **nonuniform_sampling** : `bool <https://docs.python.org/3/library/stdtypes.html#bool>`_, optional

                ``False``, uniform sampling among ``start_range_std_activation`` and ``end_range_std_activation``; \
                ``True``, non-uniform sampling. \
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
    """

    ##########
    # Initial checks
    ##########

    ### Input tensors
    if reference is None:
        reference = input
    else:
        if input.size() != reference.size():
            raise Exception(f"Sizes of 'input' {input.size()} and 'reference' {reference.size()} not coincident!")
        pass
    pass

    flag_full_im_weight = False
    if (weight is None) or not isinstance(weight, torch.Tensor):
        raise Exception(f"No valid 'weight' has been provided: {weight}!")
    elif weight.ndim == 4:  # Standard convolution-like kernel
        flag_full_im_weight = False
    elif weight.ndim == 2:  # Full-image uniform kernel: flag and modify size
        flag_full_im_weight = True
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        # In such case:
    pass
    padding = 'valid' if flag_full_im_weight else padding  # Valid is in such case the full image

    ################
    # Check whether operations should be pushed into GPU: based on availability and based on 'weight'
    ################
    computation_device = 'cuda' \
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and \
           (weight.device.type == 'cuda' or input.device.type == 'cuda') \
        else 'cpu'
    computation_device = 'mps' if torch.backends.mps.is_available() else computation_device
    # print(f"'computation_device' in 'conv2d_crossdiff': {computation_device}")

    if input is not None and computation_device != input.device.type:
        input = input.to(computation_device)
        # print(f"Device 'input' = {input.device.type}")
    if reference is not None and computation_device != reference.device.type:
        reference = reference.to(computation_device)
        # print(f"Device 'reference' = {reference.device.type}")
    if weight is not None and computation_device != weight.device.type:
        weight = weight.to(computation_device)
        # print(f"Device 'weight' = {weight.device.type}")
    ################

    ################
    flag_single_image = True
    if input.ndim == 4:
        flag_single_image = False
    elif input.ndim != 3:
        raise Exception(f"'input' (and/or 'reference') can be either 3D or 4D: however 'input' of {input.ndim}D!")
    pass
    if flag_single_image:
        input = input.unsqueeze(0)
        reference = reference.unsqueeze(0)
    pass

    ### Check groups
    assert isinstance(groups, int), f"Invalid 'groups' provided: only 'int' accepted, {groups} found!"
    assert groups > 0, f"Number of 'groups' must be positive int, {groups} given!"
    assert (input.size(-3) % groups == 0) and (weight.size(-4) % groups == 0), \
        f"Number of 'groups' must divide 'C_in' and 'C_out', {groups} given!"
    assert (input.size(-3) / groups == weight.size(-3)), \
        f"The number of input channels of 'weights' must be 'Cin'/'groups', i.e. {input.size(-3)}/{groups}; " + \
        f"{weight.size(-3)} found instead (for 'weight' of size {weight.size()}!"

    ### Stride and padding:
    # Since this options are restricted to "conv2d_adapted" SOME of the  corresponding checks are performed therein.
    # WE NEED TO CREATE STRIDE_TUPLE THOUGHT!
    assert padding in ['same', 'valid'], f"Only valid options for ``padding`` are 'same' and 'valid', {padding} found!"
    assert (isinstance(stride, int)) or (isinstance(stride, tuple) and (len(stride) == 2)), \
        f"Invalid 'stride' provided: only 'int' or 2-element 'tuple' accepted, {stride} found!"
    stride_tuple = (stride, stride) if isinstance(stride, int) else stride
    for i in range(len(stride_tuple)):
        assert stride_tuple[i] > 0, f"Stride values must be positive int, {stride} given!"
        assert (padding == 'valid') or (stride_tuple[i] == 1), \
            f"Padding='same' is not supported for strided convolutions (stride {stride} found)!"
    assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular'], \
        f"Only valid options for ``padding_mode`` are 'same' and 'valid', {padding_mode} found!"

    ### Interpolation options
    if calculation_mode is None:
        calculation_mode = 'interpolated'
    pass
    if memory_saving_version is None:
        memory_saving_version = True
    pass
    assert calculation_mode in ['interpolated', 'n4'], \
        f"Invalid 'calculation_mode': valid options 'interpolated' and 'n4', {calculation_mode} found!"

    ##########
    # Activation function
    ##########

    # Extract sigma function modifiers were specified in 'theta' and, if so, pass them as additional keyword params
    # (what is left in **kwargs will go later to the function 'generate_builder_1_order_interpolation' in the cases of \
    # interpolated calculations

    modifiers_kwargs = {}
    keys_potential_modifiers = ['x_compress', 'y_stretch', 'x_offset', 'y_offset']
    for key in keys_potential_modifiers:
        value = kwargs.pop(key, None)
        modifiers_kwargs[key] = value if value is not None else 0.0 if 'offset' in key else 1.0
    pass

    activation_f = ndim_activation_function_from_1dim_activation_functions(activation_function,
                                                                           **modifiers_kwargs)

    ##########
    # Common calculations for all the different modes
    # We will need to cut/stride the resulting differences to accommodate to "input" the operations performed
    # by "conv2d" on the "reference"-"input(p)"
    ##########

    # For that we need to have into account, first, the offset and padding that the "conv2d" methods apply,
    # so we can correspondingly calculate the pixels where the interpolating factors need calculation
    #
    pos_1st_sample = [0, 0]
    conv_output_size = list(input.size()[-2:])
    # print(f"\nPadding {padding} and size of kernel {weight.size()}")
    for i in range(2):
        # print(f"\tDim: {i}: padding {padding}")
        pos_1st_sample[i] = 0 if (padding == 'same') else int((weight.size(-2 + i) - 1) / 2.0)
        conv_output_size[i] = conv_output_size[i] if padding == 'same' else \
            floor((conv_output_size[i] - (weight.size(-2 + i) - 1) - 1) / stride_tuple[i]) + 1

    # Channel and group information
    #
    c_in = input.size(-3)
    c_out = weight.size(-4)
    #
    c_in_per_group = input.size(-3) // groups
    c_out_per_group = weight.size(-4) // groups

    ##########
    # Operation for the different modes
    ##########

    if calculation_mode == 'n4':
        ################################################################
        # For each 'u[p]' we need to create an image 'v-u[p]'
        ################################################################

        # Replicate the reference v so every pixel has the full image
        expanded_reference = reference.unsqueeze(-4).unsqueeze(-4).repeat(
            tuple([1 for i in range(len(reference.size()[0:-3]))]) + (reference.size(-2), reference.size(-1)) +
            tuple((1, 1, 1))
        )
        # Subtract the image u for each and every pixel and apply the activation
        expanded_input = input.unsqueeze(-4).unsqueeze(-4).repeat(
            tuple([1 for i in range(len(input.size()[0:-3]))]) + (input.size(-2), input.size(-1)) + tuple((1, 1, 1))
        ).transpose(-2, -5).transpose(-1, -4)
        #
        expanded_activated_diff = activation_f(expanded_reference - expanded_input)

        if not flag_full_im_weight:  # i.e. for 'flag_full_im_weight'=False
            #
            ###
            # Convolve the activated differences with the filter
            ###
            # In order to use the utility "conv2d" (or "conv2d_adapted") we need to collapse/uncollapse the first dimensions
            # getting a tensor of the type (...*B*H*W, Cin, H, W) (since that is what "conv2d" likes)
            collapsed_expanded_activated_diff = expanded_activated_diff.view(
                tuple([np.array(list(expanded_activated_diff.size()[0:-3])).prod()]) +
                tuple(expanded_activated_diff.size()[-3:])
            )
            #
            collapsed_filtered_expanded_activated_diff = conv2d_adapted(
                collapsed_expanded_activated_diff, weight, groups=groups,
                stride=stride, padding=padding, padding_mode=padding_mode
            )
            #
            # We uncollapse the filtered images, going back to the (..., B, H, W, Cin, H, W) structure
            filtered_expanded_activated_diff = collapsed_filtered_expanded_activated_diff.view(
                tuple(expanded_activated_diff.size()[0:-3]) +
                tuple(collapsed_filtered_expanded_activated_diff.size()[-3:])
            )

            # WARNING: WE NEED TO TAKE INTO ACCOUNT THE DIFFERENCE IN SIZE THAT THE CONVOLUTIONAL OPERATOR
            # HAS CAUSED TO THE "v"/"reference" COPIES OF THE IMAGE, WHICH SHOULD REFLECT IN THE SAME SIZE DIFFERENCE
            # FOR "u"/"input". TO ACCOUNT FOR THE OFFSET PROPERTY WE NEED TO TAKE INTO ACCOUNT WHETHER THERE WAS PADDING
            # ('valid' vs 'same') AND WHETHER THE FILTER WAS ODD OR EVEN.

            filtered_expanded_activated_diff = \
                filtered_expanded_activated_diff.movedim((-5, -4), (0, 1)) \
                    [pos_1st_sample[0]:pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]:stride_tuple[0], \
                pos_1st_sample[1]:pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]:stride_tuple[1] \
                ] \
                    .movedim((0, 1), (-4, -3))

            # Extract the required values using reordering of the channels and extracting the diagonal
            output = filtered_expanded_activated_diff.diagonal(dim1=-4, dim2=-2).diagonal(dim1=-3, dim2=-2)
            #
        else:  # If 'flag_full_im_weight'=True
            #
            # We will need to go accumulating the results of scalar multiplications
            # We are going to obtain a result of size (B, H, W, Cout)
            filtered_expanded_activated_diff = torch.zeros(
                expanded_activated_diff.size()[0:-3] + tuple([c_out]),
                device=computation_device,
                dtype=expanded_activated_diff.dtype
            )
            #
            for c_out_i in range(c_out):
                corresponding_group = c_out_i // c_out_per_group
                list_c_in_group = list(
                    range(corresponding_group * c_in_per_group, (corresponding_group + 1) * c_in_per_group)
                )
                #
                filtered_expanded_activated_diff[:, :, :, c_out_i] = \
                    torch.mul(
                        expanded_activated_diff[:, :, :, list_c_in_group], weight[c_out_i]
                    ).sum(dim=list(range(-3, 0)))
                #
            pass
            #
            # Remaining: set the number of channels in dim=-3 (instead of -1) and "gruyerize" the result
            ch_in_pos_minus_3 = filtered_expanded_activated_diff.movedim(-1, -3)
            output = ch_in_pos_minus_3[
                     :, :,
                     pos_1st_sample[0]:(pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]):stride_tuple[0],
                     pos_1st_sample[1]:(pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]):stride_tuple[1]
                     ]
            #
        pass

    elif calculation_mode == 'interpolated':

        ############################################################################################################
        # THERE ARE TWO SUB-METHODS FOR INTERPOLATED:
        # - INTERPOLATED + (EXTRA)MEMORY-SAVING VERSION, slightly slower but slightly less memory-consuming
        # - INTERPOLATED + NO (EXTRA)MEMORY-SAVING VERSION, slightly faster but slightly more memory-consuming
        ############################################################################################################
        # PROCEDURE, WHICH IS COMMON TO BOTH (SOME PARTS OF THE COMMON PROCEDURE ARE KEPT OUTSIDE THE BRANCHING SO THEY APPEAR
        # ONLY ONCE, SOME PARTS OF THE COMMON PROCEDURE ARE VERY SPECIFIC WITHIN EACH BRANCH):
        # IN ORDER TO ACCEPT MULTIPLE CHANNELS AND EVEN GROUPS IN THE CONVOLUTIONS
        # WE WILL PROCESS EACH COMBINATION INPUT CHANNEL->OUTPUT CHANNEL SEPARATELY. THAT IS:
        # - WE WILL CALCULATE SEPARATELY 1-ch->1-ch CONVOLUTIONS USING "OUR" INTERPOLATION METHOD: THE NUMBER
        #   OF 1-ch->1-ch CONVOLUTIONS DEPENDS NOT ONLY ON THE NUMBER OF IN/OUT CHANNELS BUT ALSO
        #   ON THE NUMBER OF "groups", SINCE THOSE SEPARATED 1-ch->1-ch CONVOLUTIONS ARE THE "UNWRAPPED" COMPONENT
        #   CONVOLUTIONS OF THE MULTI-GROUP CONVOLUTION;
        #   (BY THE WAY: THOSE 1-ch->1-ch CONVOLUTIONS WILL BE CALCULATED USING AN UNWRAPPED VERSION OF THE FILTER,
        #   WHERE ALL THE 1-ch->1-ch CONVOLUTIONS HAVING THE SAME INPUT CHANNEL AS "ORIGIN" APPEAR CONTIGUOUS SO
        #   CALCULATIONS ARE EASED.)
        # - WE WILL THEN COMBINE THE RESULTS OF THESE 1-ch->1-ch CONVOLUTIONS INTO THE FINAL RESULT.
        #   THE FINAL STRUCTURE GIVEN BY THE HIERARCHICAL STRUCTURE OF THE FILTER
        #   AND THE VALUE OF GROUPS WILL BE ENFORCED, BY THE END OF THE ALGORITHM, MANUALLY.
        ############################################################################################################
        # Example: for a filter W of (C_out, C_in, H_w, W_w) WITH "groups"=1,
        # where C_in is the number of channels of the input,
        # we would need to:
        # - reshape the filter W into (C_out*C_in, 1, H_w, W_w)
        # - call the convolution with parameter "group"=C_in, so the input channels are addressed separately;
        #   doing so we obtain the INDIVIDUAL CONVOLUTIONS FOR COMPOSING THE RESULT;
        # - do this for all the samples for interpolation, obtaining C_out*C_in interpolated output channels
        # - combine these C_out*C_in channels into THE CORRECT EXIT.
        # HOWEVER, FOR OTHER "group" NUMBERS, INPUTS HAVE TO BE SPLIT FOR THEIR CORRESPONDING OUTPUTS
        # (AND FILTER MASKS) AND THEN RECOMBINE THEM IN THE RIGHT ORDER (AND SUCH ORDER IS NOT STRAIGHTFORWARD!)
        ############################################################################################################

        # Parameters for the function 'generate_builder_1_order_interpolation' (used in fact only in the cases of \
        # interpolated calculations)

        range_kwargs = {}
        #
        range_kwargs['activation_function'] = activation_function
        #
        detached_modified_kwargs = {}
        for key in modifiers_kwargs:
            detached_modified_kwargs[key] = modifiers_kwargs[key].detach().clone()
        pass
        #
        if 'num_sampling_points' in list(kwargs.keys()):
            range_kwargs['num_sampling_points'] = kwargs.pop('num_sampling_points', None)
        if 'nonuniform_sampling' in list(kwargs.keys()):
            range_kwargs['nonuniform_sampling'] = kwargs.pop('nonuniform_sampling', None)
        if 'interpolation_type' in list(kwargs.keys()):
            range_kwargs['interpolation_type'] = kwargs.pop('interpolation_type', None)
        if 'extrapolation_type' in list(kwargs.keys()):
            range_kwargs['extrapolation_type'] = kwargs.pop('extrapolation_type', None)
        pass

        for key in ['start_range', 'end_range']:
            extracted_value = kwargs.pop(key + '_std_activation', None)
            if extracted_value is not None:
                range_kwargs[key] = extracted_value / detached_modified_kwargs['x_compress'] \
                                    + detached_modified_kwargs['x_offset']
                # To make sure that the value is a scalar (int or float)
                if isinstance(range_kwargs[key], torch.Tensor):
                    range_kwargs[key] = range_kwargs[key].item()
                pass
            pass
        pass

        ##################
        # Load the sampling points and the corresponding interpolating kernels for interpolating the activation function
        ##################
        # a_s, kernel_list = generate_builder_1_order_interpolation(**range_kwargs, **detached_modified_kwargs)
        a_s, kernel_list = generate_builder_1_order_interpolation_for_input(input.detach(), **range_kwargs)
        # Force the computation device presumed for all the calculations within the function (see above)of the 
        if a_s is not None and computation_device != a_s.device.type:
            a_s = a_s.to(computation_device)
        pass

        ##################
        # Unwrap the filter for 1-ch->1-ch convolutions with contiguous input channels
        ##################

        weight_121_contig_out = weight.view(
            (weight.size(0) * weight.size(1), 1, (weight.size(2)), (weight.size(3)))
        )

        ind2collapsed_weight = np.array(
            [g * c_out_per_group * c_in_per_group + c_i * c_out_per_group + c_o
             for g in range(groups)
             for c_o in range(c_out_per_group)
             for c_i in range(c_in_per_group)
             ]
        )
        ind2contiguous_input_ch = ind2collapsed_weight.argsort()

        # Reformatted weights for expanded convolutions with contiguous output channels
        # Reformatted weights for expanded convolutions with contiguous input channels
        weight_121_contig_in = weight_121_contig_out[ind2contiguous_input_ch, :]

        ##################
        # We "gruyerize" the input image 'input' so it corresponds to the stride and padding of the 'reference'
        # (it will be necessary to make the interpolating factor that we will calculate compatible with the latter)
        ##################

        gruyerized_input = input[
                           :, :,
                           pos_1st_sample[0]:(pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]):stride_tuple[
                               0],
                           pos_1st_sample[1]:(pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]):stride_tuple[1]
                           ]

        if memory_saving_version:
            ############################################################################################################
            # Branch INTERPOLATED + EXTRA-MEMORY-SAVING
            ############################################################################################################
            # PROCEDURE:
            # - we perform all the 1-ch->1-ch convolutions corresponding to the same input channel c_i by
            #   CALCULATING ALL THE CONVS FOR ALL THE SAMPLES sample a_s IN A LOOP, AND ACCUMULATING THE INTERPOLATED
            #   RESULTS, OBTAINING THE COMPLETE 1-ch->1-ch CONVOLUTIONS for c_i BEFORE PASSING TO THE NEXT INPUT CHANNEL
            #   (instead of calculating, for each sample a_s, the convolutions for all the input channels, which is
            #   what happens in the method without memory saving.)
            # We will obtain a tensor (B,C_out*C_in,H,W) due to the expansion of the filter and the use of "groups".
            # CAREFUL (1)! The "conv2d" takes 4 input dims, we have 5: we need to collapse the S into B
            # CAREFUL (2)! The fact that "groups" make that not all inputs are linked to all outputs makes some "funny"
            #              reordering necessary: for that reason we create a pair of kind-of "translating structures"
            #              called "ind2collapsed_weight" and "ind2contiguous_input_ch" for getting the resulting 1D
            #              indices of the collapsed "weight" correspond to contiguous repetitions of the same input ch.
            ############################################################################################################

            ##################
            # We calculate "simultaneously" all the operations corresponding to the same input channel
            # (Remember: one input channel is involved in 'c_out_per_group' = 'c_out' / 'groups' outputs
            ##################
            # WARNING! Only valid for 1D activation functions (acting thus identically on all input channels)

            # We will be accumulating the contribution of each sample s and each input channel to the output here,
            # which will eventually form the result of the 1-ch->1-ch CONVOLUTIONS
            individual_convolution_result = torch.zeros(
                (gruyerized_input.size(0), len(ind2contiguous_input_ch),
                 gruyerized_input.size(-2), gruyerized_input.size(-1)),
                device=computation_device
            )

            # if flag_full_im_weight:  # Only necessary for 'flag_full_im_weight'=True; conv2d manages strides
            #     gruyerized_reference = reference[:, :, \
            #                            pos_1st_sample[0]:(pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]):
            #                            stride_tuple[0], \
            #                            pos_1st_sample[1]:(pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]):
            #                            stride_tuple[1] \
            #                            ]
            # pass

            for c_i in range(input.size(-3)):  # Each individual input channel
                #
                for s in range(len(kernel_list)):
                    #
                    if not flag_full_im_weight:  # i.e. for 'flag_full_im_weight'=False
                        #
                        # NOTE: - "reference[:, (c_i):(c_i + 1), :, :]" has a size of [B, 1, H, W]
                        #       - "weight_121_contig_in[(c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :, :]"
                        #         has a size of [c_out_per_group, 1, H, W]
                        #       - the result of this convolution is thus has a size of [B, c_out_per_group, H, W]
                        #
                        omega_conv2d_sigma_c_i_a_s = conv2d_adapted(
                            activation_f(reference[:, (c_i):(c_i + 1), :, :] - a_s[s]),
                            weight_121_contig_in[(c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :, :],
                            groups=1,
                            stride=stride, padding=padding, padding_mode=padding_mode
                        )
                        #
                    else:  # i.e. for 'flag_full_im_weight'=True
                        #
                        # NOTE: - "reference[:, (c_i):(c_i + 1), :, :]" has a size of [B, 1, H, W]
                        #         and we need [B, c_out_per_group, H, W] -> replicate!
                        #       - remember that in this case
                        #         "weight_121_contig_in[(c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :, :]"
                        #         has a size of [c_out_per_group, 1, 1, 1] -> we need to unsqueeze and replicate!
                        # BUT: GOOD NEWS!!! THE DEFAULT CASING OPERATIONS OF PYTORCH (for torch.mul) DO EXACTLY
                        # WHAT WE NEED, SO WE ONLY NEED  TO MULTIPLY THE TWO TENSORS!

                        # This 'omega_conv2d_sigma_c_i_a_s' should occupy [B, c_out_per_group, 1, 1] (unlike \
                        # the 'omega_conv2d_sigma_c_i_a_s' in the non-full-im case, which was [B, c_out_per_group, H, W]
                        omega_conv2d_sigma_c_i_a_s = torch.mul(
                            activation_f(reference[:, (c_i):(c_i + 1), :, :] - a_s[s]).sum(dim=(-2, -1), keepdims=True),
                            weight_121_contig_in[
                                (c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :, :
                            ].movedim(-4, -3)
                        )
                        # What is yet to be done is: multiply to the interpolant factor, which is full image (since \
                        # it is a full image, point-wise function of the input image/gruyerized input image).

                    pass
                    #
                    # We calculate the contribution of sample s and input channel c_i to the output:
                    interpolant_factor_c_i_a_s = kernel_list[s](gruyerized_input[:, (c_i):(c_i + 1), :, :])
                    repeated_interpolant_factor_c_i_a_s = interpolant_factor_c_i_a_s.repeat((1, c_out_per_group, 1, 1))
                    #
                    # And we accumulate the contribution to the result of 1-ch->1-ch CONVOLUTIONS
                    individual_convolution_result[:, (c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :] = \
                        individual_convolution_result[:, (c_i) * c_out_per_group:(c_i + 1) * c_out_per_group, :, :] + \
                        repeated_interpolant_factor_c_i_a_s * omega_conv2d_sigma_c_i_a_s
                pass

            pass

        else:  # a.k.a. not memory_saving_version
            ############################################################################################################
            # INTERPOLATED + NO EXTRA-MEMORY-SAVING
            ############################################################################################################
            # - We calculate, in one go, all the 1-ch->1-ch convolutions for all the B batch images and for all the
            #   interpolating points a_s, which will be a tensor (S,B,C_out*C_in,H,W) due to the expansion of the filter
            # - From there, we fuse together all the a_s, 1-ch->1-ch convolutions corresponding to the same batch image
            #   to obtain the same final 1-ch->1-ch convolution result of each bath image,
            #   which will be a tensor (B,C_out*C_in,H,W) due to the expansion of the filter because of "groups".
            ############################################################################################################

            # Calculate the displaced/substracted, activated versions for all the sampling points.
            # For that, for the input (B,C_in,H,W), we create an intermediate dimension, so (S,B,C_in,H,W), S=num a_s
            expanded_a_s = a_s.unsqueeze(0).repeat(
                tuple(reference.size()) +
                tuple([1])
            ).movedim(-1, 0)

            expanded_reference = reference.unsqueeze(0).repeat(
                tuple(a_s.size()) +
                tuple([1 for i in range(reference.ndim)])
            )

            # Regarding the activation: it can act differently on each dimension!
            sigma_reference_minus_a_s = activation_f(expanded_reference - expanded_a_s)

            # We apply the filter to obtain, FOR EACH SAMPLE a_s, the INDIVIDUAL CONVOLUTIONS FOR COMPOSING THE RESULT.
            # That is, we will obtain (S,B,C_out*C_in,H,W) due to the expansion of the filter and the use of "groups".

            #######
            # Re-shape (S,B)->(S*B,C_in,H,W)..., calculate convolution (S*B,C_out*C_in/groups,H,W), and
            # back to (S*B)->(S,B)
            #######

            # Re-shape:
            s_x_b_sigma_reference_minus_a_s = sigma_reference_minus_a_s.view(
                tuple([np.array(list(sigma_reference_minus_a_s.size()[0:-3])).prod()]) +
                tuple(sigma_reference_minus_a_s.size()[-3:])
            )

            # Conv (or whatever you do if the filter is a full-im, constant filter):
            #
            if not flag_full_im_weight:  # If 'flag_full_im_weight'=False
                #
                per_a_s_and_per_b_image_individual_convolution_result = conv2d_adapted(
                    s_x_b_sigma_reference_minus_a_s, weight_121_contig_in, groups=c_in,
                    stride=stride, padding=padding, padding_mode=padding_mode
                )
                #
            else:  # If 'flag_full_im_weight'=True
                ###
                # NOTE: - "s_x_b_sigma_reference_minus_a_s" has a size of [S*B, C_in, H, W]
                #       - "weight_121" has, in this case of 'flag_full_im_weight'=True,
                #         a size of [C_out*C_in/groups, 1, 1, 1], and has been reordered
                #         to have the input channels contiguous
                #       - so one way to get the result is to:
                #           - repeat as many times as C_in/groups each input channel
                #           - directly multiply with the filter (dim-shift required)
                ###

                # Remember that each 's_x_b_sigma_reference_minus_a_s'[bs,:,:,:] has C_in channels, wherein 'bs' \
                # represents a certain batch image 'b' subtracted by the specific interpolating sample 's'. \
                # In order to calculate the full-image convolution for constant filter (in 'weight_121')
                # we need to: (1) sum the whole image per-channel, and (2) apply the filter using multiplication.
                # 'per_a_s_and_per_b_image_individual_convolution_result' -> [B*S, C_out*C_in_per_group, 1, 1]
                per_a_s_and_per_b_image_individual_convolution_result = torch.mul(
                    s_x_b_sigma_reference_minus_a_s.sum(dim=(-2, -1), keepdims=True).repeat_interleave(
                        repeats=c_out_per_group, dim=-3
                    ),
                    weight_121_contig_in.movedim(-4, -3)
                )

            pass  # End of the 'flag_full_im_weight' condition

            # ...  and shape-back (S*B)->(S,B) the images to filter
            per_a_s_and_per_b_image_individual_convolution_result = \
                per_a_s_and_per_b_image_individual_convolution_result.view(
                    tuple(sigma_reference_minus_a_s.size()[0:-3]) +
                    tuple(per_a_s_and_per_b_image_individual_convolution_result.size()[-3:])
                )

            # So far, the first (C_out/groups)*(C_in/groups) output channels
            # correspond to the first (C_in/groups) input channels, etc.
            # So, for the interpolation, we need to do this same expansion. This expansion will be done
            # after calculating the values for interpolation.

            per_a_s_interpolant_factors_before_expansion = torch.empty(
                tuple([a_s.numel()]) + tuple(input.size()[0:-2]) + tuple(conv_output_size),
                device=computation_device
            )
            # Original command used .view; depending on the later operations, and due to contiguity problems and not
            # to the fact that the size is not correct, we use .reshape.
            # input_into_conv_output_size = input.view(
            #     tuple([np.array(list(input.size()[0:-2])).prod()]) + tuple(input.size()[-2:])
            # )[:,pos_1st_sample[0]:(pos_1st_sample[0]+conv_output_size[0]*stride_tuple[0]):stride_tuple[0], \
            #     pos_1st_sample[1]:(pos_1st_sample[1]+conv_output_size[1]*stride_tuple[1]):stride_tuple[1] \
            # ]
            # input_into_conv_output_size = input.reshape(
            #     tuple([np.array(list(input.size()[0:-2])).prod()]) + tuple(input.size()[-2:])
            # )[:, pos_1st_sample[0]:(pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]):stride_tuple[0], \
            #                               pos_1st_sample[1]:(
            #                                       pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]):
            #                               stride_tuple[1] \
            #                               ]
            # input_into_conv_output_size = input_into_conv_output_size.view(
            #     tuple(input.size()[0:-2]) + tuple(input_into_conv_output_size.size()[-2:])
            # )
            # input_into_conv_output_size = input[:, :, \
            #     pos_1st_sample[0]:(pos_1st_sample[0] + conv_output_size[0] * stride_tuple[0]):stride_tuple[0], \
            #     pos_1st_sample[1]:(pos_1st_sample[1] + conv_output_size[1] * stride_tuple[1]):stride_tuple[1] \
            # ]

            for s in range(len(kernel_list)):
                per_a_s_interpolant_factors_before_expansion[s, :] = kernel_list[s](gruyerized_input)
            pass

            # We need to expand to match "per_a_s_and_per_b_image_individual_convolution_result", which corresponds to the following
            # aspects:  (   S sampling points,
            #               B batch images,
            #               (C_out/groups) corresponding to input channel 0, (C_out/groups) to input channel 1, ...
            #               H, W
            #           )

            # per_a_s_interpolant_factors_after_expansion = per_a_s_interpolant_factors_before_expansion.unsqueeze(
            #     -3).repeat(
            #     tuple([1 for i in range(len(per_a_s_interpolant_factors_before_expansion.size()[0:-2]))]) +
            #     tuple([c_out_per_group]) + (1, 1)
            # )
            #
            # per_a_s_interpolant_factors_after_expansion = per_a_s_interpolant_factors_after_expansion.view(
            #     tuple(per_a_s_interpolant_factors_after_expansion.size()[0:-4]) +
            #     tuple([per_a_s_interpolant_factors_after_expansion.size(
            #         -4) * per_a_s_interpolant_factors_after_expansion.size(-3)]) +
            #     tuple(per_a_s_interpolant_factors_after_expansion.size()[-2:])
            # )

            per_a_s_interpolant_factors_after_expansion = \
                per_a_s_interpolant_factors_before_expansion.repeat_interleave(repeats=c_out_per_group, dim=-3)

            # Final calculation of the individual convolutions to be combined into groups
            individual_convolution_result = (
                    per_a_s_interpolant_factors_after_expansion * per_a_s_and_per_b_image_individual_convolution_result
            ).sum(0)

        pass  # End of the 'memory_saving_version' condition

        # Final combination of channels according to groups (COMMON TO BOTH MEMORY-SAVING VERSIONS)
        output = individual_convolution_result.movedim(-3, 0)[ind2collapsed_weight, :].movedim(
            0, -3).view(
            tuple(individual_convolution_result.size()[0:-3]) +
            tuple(weight.size()[-4:-2]) +
            tuple(individual_convolution_result.size()[-2:])
        ).sum(-3)


    else:
        raise Exception(
            f"Valid values for 'calculation_mode' are 'interpolated' and 'n4', {calculation_mode} found!"
        )
    pass

    if flag_single_image:
        output = output[0]
    pass

    return output


######################################################
### Base function for ibnn_internal and INRF
######################################################

def f_modified_RF(im, theta, u=None, v=None,
                  phi_activation='relu', sigma_activation='tanh',
                  m_padding='same', m_padding_mode='zeros', m_groups=1,
                  w_padding_mode='zeros', w_groups=1,
                  calculation_mode='interpolated', memory_saving_version=True,
                  **kwargs):
    """
    Calculation of the expression

    $$\\mathbf{y}(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{A} (\\mathbf{I}))(\\mathbf{p})
    - \\mathbf{\\lambda}
    \\big( \\mathbf{\\Omega} \\ast \\mathbf{\\sigma} ( \\mathbf{V}-\\mathbf{U}[\\mathbf{p}] ) \\big) (\\mathbf{p})
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
    $$

    wherein $\\mathbf{I}$ represents an input image, $\\mathbf{A} (\\mathbf{I})$ is an affine transform of the input \
    image ruled by the linear matrix $\\mathbf{M}$ and the bias $\\mathbf{b}$, \
    and wherein both images $\\mathbf{U}$ and $\\mathbf{V}$, \
    having the same size and with the latter regarded as a *reference* against which $\\mathbf{U}$ is contrasted, \
    are combined to provide a content-based threshold/bias.

    This function accepts affine transform $\\mathbf{A} (\\mathbf{I}))$ in the form of a convolution or in the form \
    of a fully-connected plus bias. The above expression, whenever the affine transform $\\mathbf{A} (\\mathbf{I}))$ \
    is a convolution, can be written as

    $$\\mathbf{y}(\\mathbf{p}) = \\Phi
    \\Big(
    (\\mathbf{M} \\ast \\mathbf{I})(\\mathbf{p})
    - \\mathbf{b}
    - \\mathbf{\\lambda}
    \\big( \\mathbf{\\Omega} \\ast \\mathbf{\\sigma} ( \\mathbf{V}-\\mathbf{U}[\\mathbf{p}] ) \\big) (\\mathbf{p})
    \\Big)
    \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
    $$

    The function has a **different behavior for different configurations of its arguments**:

    - It has **different behavior depending on the inputs left empty**:

        - When no reference ``v`` is provided the function considers $\\mathbf{V}=\\mathbf{U}$ \
          (i.e. $\\mathbf{U}$ is compared to itself), that is,

          $$
          \\mathbf{y}(\\mathbf{p}) = \\Phi
          \\Big(
          (\\mathbf{M} \\ast \\mathbf{I})(\\mathbf{p})
          - \\mathbf{b}
          - \\mathbf{\\lambda}
          \\big( \\mathbf{\\Omega} \\ast \\mathbf{\\sigma} ( \\mathbf{U}-\\mathbf{U}[\\mathbf{p}] ) \\big) (\\mathbf{p})
          \\Big)
          \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, ,
          $$

          which in fact represents the operation forming the basis of the *Implicitly-Biased Neural Network* (ibnn_internal) and thus \
          of the corresponding class :py:class:`.IBNNInternalLayer`.

        - When neither ``u`` nor ``v`` are provided the function uniquely calculates

          $$\\mathbf{y}(\\mathbf{p}) = \\Phi \\Big(
          (\\mathbf{M} \\ast \\mathbf{I})(\\mathbf{p})
          - \\mathbf{b}
          \\Big) \\,\\, , \\, \\forall \\mathbf{p} \\in \\mathcal{P}_u \\, .
          $$

          which corresponds to the standard model.

    - It works **interpreting** $\\mathbf{A} (\\mathbf{I})$ **as affine or convolutional depending on the value** \
      **of the argument** `m_padding`:

      - For the values of `m_padding` ``‘same’`` and ``‘valid’`` the transform $\\mathbf{A} (\\mathbf{I})$ is \
        interpreted as convolutional ($\\mathbf{M} \\ast \\mathbf{I} - \\mathbf{b}$). In such case, the tensors \
        are defined according to the definitions in :py:func:`.conv2d_adapted`: see its documentation for \
        details about the allowed formats for $\\mathbf{b}$. In this case, \

        - the filter $\\mathbf{M}$, common for the whole image, has a sizes \
          $(C_U,\\frac{C_I}{\\textrm{groups}_{M}},H_M,W_M)$;

        - the bias $\\mathbf{b}$ is common for the whole image but does not need common for all channels, \
          that is, it would be a scalar or a vector with $C_U$ elements.

      - For the value of `m_padding` ``‘fc’`` the transform $\\mathbf{A} (\\mathbf{I})$ is \
        interpreted as pure affine transform, without spatial limits, but with channel-based restrictions regarding \
        groups, as stated in :py:func:`.fc_2d`: see its documentation for \
        details about the allowed formats for $\\mathbf{b}$.


    The following assumptions have been made:

    - the input image $\\mathbf{I}$ has $C_I$ color channels and the input $\\mathbf{U}$/$\\mathbf{V}$ \
      and the output have $C_U$ color channels; both $\\mathbf{I}$ and $\\mathbf{U}$/$\\mathbf{V}$, \
      and consequently the resulting output or the function, are

        - either 4D, representing batches with the same number $B$ of inputs, that is, they have \
        $(B, C_I, H_I, W_I)$ and $(B, C_U, H_U, W_U)$ as their respective sizes; or

        - 3D, representing one single entry (pair), that is, they have \
        $(C_I, H_I, W_I)$ and $(C_U, H_U, W_U)$ as their respective sizes;

    - the factor $\\mathbf{\\lambda}$ is common for the whole image but \
      does not need common for all channels, that is, it is are scalars or a vector with $C_U$ elements;

    - the multi-dimensional activation function $\\bar{\\sigma}: \\mathbb{R}^{C_u} \\to \\mathbb{R}^{C_u}$ is composed
      of component-wise individual 1D activation functions which can be different from each other, that is,
      $\\bar{\\sigma}(\\mathbf{u}) = \\big( \\ldots,\\sigma_{c}(u_c),\\ldots \\big)$;

    - the filter $\\mathbf{\\Omega}$, common for the whole image, has size \
      $(C_U,\\frac{C_U}{\\textrm{groups}_{U}},H_{\\Omega},W_{\\Omega})$, or **size** \
      $(C_{out},\\frac{C_{in}}{\\mathrm{groups}})$ **if it is set as lacking spatial extent, and representing ** \
      **in this case a uniform value for the whole extent of the image**; this latter convention for no-spatial-extent \
      does not apply to convolutional $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_I, H_I, W_I)$ \
        and having the same number of batch elements of ``u``, or 3D with one single entry and size $(C_U, H_U, W_U)$
    theta : dict[torch.Tensor]
        Dictionary containing the following elements (**description therein refers to the convolutional case,** \
        **see above for the fully-connected case**):
            - ``theta['m']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the 2D filter for the image I (``im``);
            - ``theta['b']``: `torch.Tensor <https://pytorch.org/docs/stable/tensors.html#torch.Tensor>`_ \
            representing the bias of the filter m, which is considered \
            either a scalar or a vector with the same number of channels as u;
        Depending on the input (at least ``u`` different from ``None``) it must contain also:
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
    u, v : torch.Tensor, optional
        2D tensor representing the outputs of a 2D lattice of hidden units, either 4D with size $(B, C_U, H_U, W_U)$ \
        or 3D with one single entry and size $(C_U, H_U, W_U)$
        See the definition of ``u``. Default: ``None`` (that is, ``v`` = ``u``)
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

    ################
    # Check whether operations should be pushed into GPU: based on availability and based on 'theta'
    ################
    computation_device = 'cpu'
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # If here, 'cuda' is available.
        # Check if 'theta' is defined so, and raise error if any key in 'theta' disagrees with the rest
        for key in theta:
            # print(f"'Device of 'theta['{key}']' = {theta[key].device.type}")
            if theta[key].device.type == 'cpu' and computation_device == 'cuda':
                raise Exception(f"Certain keys of 'theta' (e.g. '{key}') are 'cpu' while others 'cuda': inconsistent!")
            else:
                computation_device = theta[key].device.type
            pass
        pass
    pass
    if torch.backends.mps.is_available():
        for key in theta:
            # print(f"'Device of 'theta['{key}']' = {theta[key].device.type}")
            if theta[key].device.type == 'cpu' and computation_device == 'mps':
                raise Exception(f"Certain keys of 'theta' (e.g. '{key}') are 'cpu' while others 'mps': inconsistent!")
            else:
                computation_device = theta[key].device.type
            pass
        pass
    pass

        # print(f"'Device after checks = {computation_device}")
    ################

    # First: if 'v' is None, provide 'u'.
    if u is None:
        if v is not None:
            raise Exception(f"'u' is None but a value is provided for 'v': such behavior is not allowed!")
        pass
    elif v is None:
        v = u
    pass

    #############
    # Move variables to device if necessary
    #############
    if im is not None and computation_device != im.device.type:
        im = im.to(computation_device)
        # print(f"Device im = {im.device.type}")
    if u is not None and computation_device != u.device.type:
        u = u.to(computation_device)
        # print(f"Device u = {u.device.type}")
    if v is not None and computation_device != v.device.type:
        v = v.to(computation_device)
        # print(f"Device v = {v.device.type}")
    #############

    #############
    # Check and uniformize the sizes/num. of dimensions of 'u', 'v', and 'im'
    #############

    ### Check sizes of the input images
    if u is not None:
        ### Checks for 'u', 'v', and 'im'
        # Sizes and num of dimensions of 'u', 'v' and 'im'
        if (u.ndim != im.ndim) or (u.ndim != v.ndim):
            raise Exception((f"Tensors 'u', 'v', and 'im' must have the same batch/individual entry format; " +
                             f"however respective {u.ndim}D, {v.ndim}D, and {im.ndim}D found!"))
        pass
        if u.size() != v.size():
            raise Exception(f"Sizes of 'u' and 'v', respectively {u.size()} and {v.size()}, not corresponding!")
        pass
    pass

    # Transform to 4D in any case
    casted_images = False
    if im.ndim == 3:
        casted_images = True
        im = im.unsqueeze(0)
        if u is not None:
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)
        pass
    elif im.ndim == 4:
        pass
    else:
        raise Exception(f"Tensors 'u' and 'im' must be 3D or 4D, they are {u.ndim} instead!")
    pass

    # Check the number of images "in a batch"
    if u is not None:
        if u.size(0) != im.size(0):
            raise Exception((f"Respective batch sizes {u.size(0)} for 'u' " +
                             f"and {im.size(0)} for 'im' not coincident!"))
        pass
    pass

    #############
    # Get image dimensions
    #############

    ### Input image sizes: input images and relationship between theta, inputs, and outputs
    B_im, C_im, H_im, W_im = im.size(-4), im.size(-3), im.size(-2), im.size(-1)

    ### Output image sizes as far as we can infer from e.g. theta
    B_out = B_im
    C_out = theta['m'].size(0)

    ### The H and W sizes of the output image will be inferred from the input image and kernel size in the conv. case \
    ### and from theta['b'] in the 'fc' case; so we initialize them to None

    H_out, W_out = None, None

    #############
    # Parameter checks
    #############

    ### Padding options: REMEMBER THAT 'm_padding'='fc' IS A SPECIAL CASE

    w_padding = 'same'  # Not an option

    if m_padding in ['valid', 'same']:
        if m_padding_mode not in ['zeros', 'reflect', 'replicate' or 'circular']:
            raise Exception(f"Provided value for 'm_padding_mode', {m_padding_mode}, not allowed.")
    elif m_padding in ['fc']:
        if m_padding_mode is not None:
            warnings.warn((f"'m_padding' is 'fc': the provided " +
                           f"'m_padding_mode'={m_padding_mode} is thus ignored and set to 'None'!"))
    else:
        raise Exception(f"Provided '{m_padding}', other than 'valid', 'same', and 'fc', not allowed for 'm_padding'.")
    pass

    if w_padding_mode not in ['zeros', 'reflect', 'replicate' or 'circular']:
        raise Exception(f"Provided value for 'w_padding_mode', {w_padding_mode}, not allowed.")
    pass

    # Check the compatibility of 'm_groups' and 'w_groups' with the number of channels of the images
    #
    if C_im % m_groups != 0:
        raise Exception((f"Number of channels of 'im', {C_im}, is not divisible by 'm_groups', {m_groups}!"))
    pass
    if C_out % m_groups != 0:
        raise Exception((f"Number of channels of output according to 'm', {C_out}, " +
                         f"is not divisible by 'm_groups', {m_groups}!"))
    pass
    #
    if u is not None:
        if C_out != u.size(-3):
            raise Exception((f"Number of channels of 'u', {u.size(-3)}, does not match " +
                             f"the number of output channels according to 'm', {C_out}!"))
        pass
        if C_out % w_groups != 0:
            raise Exception((f"Number of channels of 'u', {C_out}, is not divisible by 'w_groups', {w_groups}!"))
        pass
    pass

    ### Fields of 'theta'

    # Correct types?
    key_list = ['m', 'b'] if u is None else ['m', 'b', 'w', 'lambda']
    for key in key_list:
        if not isinstance(theta[key], torch.Tensor):
            raise Exception(f"Provided theta['{key}'] is not a tensor, but of type {type(theta[key])}!")
        pass
    pass

    # Correct dimensions of the fields in 'theta'. Remember that the case 'fc' is slightly different
    # 'm' and (if present) 'w'. And remember that the functions 'conv2d_adapted'/'fc_2d':

    key_list = ['m'] if u is None else ['m', 'w']
    dict_groups = {'m': m_groups, 'w': w_groups}
    for key in key_list:
        if key == 'm':
            if theta[key].ndim != 4:
                raise Exception((f"Tensor 'theta['{key}']' must have 4 dimensions for 'm_padding'='{m_padding}'; " +
                                 f"{theta[key].ndim} found!"))
            pass
        else:
            if theta[key].ndim != 4 and theta[key].ndim != 2:
                raise Exception(f"Tensor 'theta['{key}']' must have 2 or 4 dimensions, {theta[key].ndim} found!")
            pass
        pass
    pass

    # # if m_padding == 'fc':
    # #     # Check 'b' itself
    # #     if theta['b'].size(-3) != C_out:
    # #         raise Exception(
    # #             (f"Dimensions of 'theta['b']' do not match those expected for 'm_padding'='fc': " +
    # #              f"{theta['b'].size(-3)} channels found, {C_out} expected.")
    # #         )
    # #     # Get the size
    # #     H_out, W_out = theta['b'].size(-2), theta['b'].size(-1)
    # #     # Check 'm' in view of this information
    # #     if H_out * W_out
    # # else:
    # # pass
    #
    # ### Infer the size of the output image from 'theta' and 'im';
    # ### need for indirect calculation in the case of m_padding='fc' since theta['m'] (and theta['b']) indirectly
    # ### encode the number of channels of the output
    #
    # B_output_according_to_m, C_output_according_to_m, H_output_according_to_m, W_output_according_to_m = \
    #     None, None, None, None
    # #
    # if m_padding == 'fc':
    #     #
    #     # "H_im", "W_im" stay the same
    #     H_output_according_to_m, W_output_according_to_m = H_im, W_im
    #     # "B_im" stays the same
    #     B_output_according_to_m = B_im
    #     # "C_output_according_to_m" must be inferred from e.g. "theta['b']"
    #     num_pixels = H_im * W_im
    #     if theta['b'].numel() % num_pixels == 0:
    #         C_output_according_to_m = int(theta['b'].numel() / num_pixels)
    #     else:
    #         raise Exception((f"Number of elements of 'theta['b']', {theta['b'].numel()}, is not multiple of " +
    #                          f"the number of pixels of 'im', {num_pixels} (HxW={H_im}x{W_im})!"))
    #     pass
    #     #
    # else:  # m_padding in ['same', 'valid']
    #     #
    #     B_output_according_to_m = B_im
    #     C_output_according_to_m = theta['m'].size(0)
    #     H_output_according_to_m = H_im if m_padding == 'same' else H_im - theta['m'].size(-2) + 1
    #     W_output_according_to_m = W_im if m_padding == 'same' else W_im - theta['m'].size(-1) + 1
    #     #
    #     if theta['m'].size(1) != int(C_im / m_groups):
    #         raise Exception(
    #             (f"Channel-related dimensions of 'theta['m']' do not match the corresponding groups: " +
    #              f"({theta['m'].size(0)},{theta['m'].size(1)}) found, " +
    #              f"({C_output_according_to_m},{int(C_im / m_groups)}) expected.")
    #         )
    #
    #     pass
    # pass
    #
    # if u is not None:
    #     C_u, H_u, W_u = u.size(-3), u.size(-2), u.size(-1)
    #     # print(f"u.size() = {u.size()}")
    #     # print(f"theta['w'].size() = {theta['w'].size()}")
    #     # print(f"w_groups = {w_groups}")
    #     #
    #     C_output_according_to_w = theta['w'].size(0)
    #     H_output_according_to_w = H_u if w_padding == 'same' else H_u - theta['w'].size(-2) + 1
    #     W_output_according_to_w = W_u if w_padding == 'same' else W_u - theta['w'].size(-1) + 1
    #     #
    #     # Fields of theta['w']
    #     if theta['w'].size(1) != int(C_u / w_groups):
    #         raise Exception(
    #             (f"Channel-related dimensions of 'theta['w']' do not match the corresponding groups: " +
    #              f"({theta['w'].size(0)},{theta['w'].size(1)}) found, " +
    #              f"({C_output_according_to_w},{int(C_u / w_groups)}) expected.")
    #         )
    #     pass
    #     #
    #     if C_output_according_to_w != C_output_according_to_m:
    #         raise Exception(
    #             (f"The number of output channels of 'theta['w']' do not match the output according to 'theta['w']': " +
    #              f"{C_output_according_to_w} found, {C_output_according_to_m} expected.")
    #         )
    #     pass
    #     #
    #     if (H_output_according_to_w != H_output_according_to_m) or (W_output_according_to_w != W_output_according_to_m):
    #         raise Exception(
    #             (f"The output (spatial) size judging from 'theta['w']' do not match those according to 'theta['m']': " +
    #              f"({H_output_according_to_w},{W_output_according_to_w}) found from 'w', " +
    #              f"({H_output_according_to_m},{W_output_according_to_m}) expected from 'm'!")
    #         )
    #     pass
    # pass
    # #
    # C_output = C_output_according_to_m
    # H_output = H_output_according_to_m
    # W_output = W_output_according_to_m
    #
    # # ### Check if valid 'calculation_mode':
    # # if calculation_mode not in ['interpolated', 'n4']:
    # #     raise Exception(f"Provided value {calculation_mode} for 'calculation_mode' not allowed.")
    # # pass

    #############
    # Check/resolve the given activation functions are valid
    #############
    phi_activation_f = ndim_activation_function_from_1dim_activation_functions(phi_activation)

    #############
    # Calculation of the AFFINE TRANSFORMATION, CONVOLUTIONAL OR FC (m, im, and b)
    #############

    im_affine_with_bias = None

    if m_padding == 'fc':
        # Fully-connected (linear) operation
        im_affine_with_bias = fc_2d(input=im, weight=theta['m'], bias=-theta['b'], groups=m_groups)
    else: # if m_padding in ['same', 'valid'], convolutional operation
        im_affine_with_bias = conv2d_adapted(
            input=im, weight=theta['m'],
            bias=-theta['b'], padding=m_padding, padding_mode=m_padding_mode, groups=m_groups
        )
    pass

    #############
    # Addition of the part ruled by lambda
    #############

    if u is not None:
        #
        # Check of sigma function modifiers were specified in 'theta' and, if so, pass them as additional keyword params
        sigma_modifiers_kwargs = {}
        potential_sigma_modifiers = ['x_compress', 'y_stretch', 'x_offset', 'y_offset']
        for key in potential_sigma_modifiers:
            sigma_key = 'sigma_' + key
            if sigma_key in list(theta.keys()):
                sigma_modifiers_kwargs[key] = theta[sigma_key]
            pass
        pass
        #
        # u_u_conv2d_crossdiff = conv2d_crossdiff(
        #     input=u, reference=v, weight=theta['w'], activation_function=sigma_activation_f,
        #     padding='same', padding_mode=w_padding_mode, groups=w_groups,
        #     calculation_mode=calculation_mode, memory_saving_version=memory_saving_version,
        #     **kwargs
        # )
        u_u_conv2d_crossdiff = conv2d_crossdiff(
            input=u, reference=v, weight=theta['w'], activation_function=sigma_activation,
            padding='same', padding_mode=w_padding_mode, groups=w_groups,
            calculation_mode=calculation_mode, memory_saving_version=memory_saving_version,
            **sigma_modifiers_kwargs, **kwargs
        )
        output_y = phi_activation_f(
            im_affine_with_bias - \
            cast_scalar_like_to_image(theta['lambda'], u_u_conv2d_crossdiff.size()) * u_u_conv2d_crossdiff
        )
        #
    else:
        output_y = phi_activation_f(im_affine_with_bias)
    pass

    if casted_images:
        output_y = output_y[0]
    pass

    return output_y