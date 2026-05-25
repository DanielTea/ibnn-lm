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
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from modified_rf.nn_layers import conv2d_adapted, conv2d_crossdiff, cast_scalar_like_to_image, f_modified_RF
from modified_rf.nn_layers import IBNNLiteLayer, IBNNInternalLayer, IBNNLayer, SMLayer

import gc
import modified_rf.memory_handling as memo


######################################################
### Functions for format conversion
######################################################

def im2vec_format(I_in_im_format, order='col'):
    """
    Formatting an image, the tensor ``I_in_im_format``, of 2D or 3D (depending on whether it is multichannel) \
    in vectorized format following ``order``-major order: respectively, column-major or \
    row-major order.

    Parameters
    ----------
    I_in_im_format : torch.Tensor
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``

    Returns
    -------
    im_in_vec_format : torch.Tensor
    """

    ### Initial checks for the input
    if I_in_im_format.ndim not in [2, 3]:
        raise Exception(f"The function takes a single 2D or 3D input image: size {I_in_im_format.size()} found!")
    pass

    ### Initial checks for the specific arguments for the function
    allowable_orders = ['col', 'row']
    if order not in ['col', 'row']:
        raise Exception(f"Allowable orders: {allowable_orders}. '{order}' found instead!")
    pass

    if order == 'col':
        I_in_vec_format = I_in_im_format.transpose(dim0=-2, dim1=-1).flatten()
    else:
        I_in_vec_format = I_in_im_format.flatten()
    pass
    #
    return I_in_vec_format


def vec2im_format(I_in_vec_format, I_in_im_size, order='col'):
    """
    Formatting the tensor ``I_in_vec_format``, an image in vector format (that is, wherein a column represents \
    a flattened image) of 1D or 2D if it corresponds to B images as an image, or B images, of the size provided
    with ``I_in_im_size``: the latter is a 3D tuple/size indicating $(\\mathrm{C}, \\mathrm{H}, \\mathrm{W})$. \
    The transform will correspond to the provided ``order``-major order presumed for the \
    vector representation.

    Parameters
    ----------
    I_in_vec_format : torch.Tensor
    I_in_im_size : list[int] or list[float] or tuple[int] or list[float] or torch.Size
        Size $(\\mathrm{C}, \\mathrm{H}, \\mathrm{W})$ of the images in image format
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``

    Returns
    -------
    I_in_im_format : torch.Tensor
    """

    ### Initial checks for the input
    if I_in_vec_format.ndim not in [1, 2]:
        raise Exception((f"The function presumes a single 1D or 2D vec-format input image: " +
                         f"size {I_in_vec_format.size()} found!"))
    pass

    ### Initial checks for the input
    if len(I_in_im_size) != 3:
        raise Exception(f"The function presumes a 3D argument 'I_in_im_size': {I_in_im_size} found!")
    pass

    ### Initial checks for the input
    num_elements_per_im = np.prod(list(I_in_im_size))
    if num_elements_per_im != I_in_vec_format.size(-1):
        raise Exception((f"The number of elements {num_elements_per_im} for the presumed im-format of size " +
                         f"{I_in_im_size} is not coincident with the vector length of {I_in_vec_format.size(-1)}"))
    pass

    ### Initial checks for the specific arguments for the function
    allowable_orders = ['col', 'row']
    if order not in ['col', 'row']:
        raise Exception(f"Allowable orders: {allowable_orders}. '{order}' found instead!")
    pass

    if order == 'col':
        I_in_im_format = I_in_vec_format.view((-1, I_in_im_size[-3], I_in_im_size[-1], I_in_im_size[-2]))
        I_in_im_format = I_in_im_format.transpose(dim0=-2, dim1=-1)
    else:
        I_in_im_format = I_in_vec_format.view((-1,) + tuple(I_in_im_size))
    pass
    #
    if I_in_vec_format.ndim == 1:
        I_in_im_format = I_in_im_format.squeeze(dim=0)
    pass
    #
    return I_in_im_format


######################################################
### Decompositions into real singular-values or eigenvalues
######################################################

def real_value_decomposition_vec_format(X, atol=1e-08, rtol=1e-05, computation_device='cuda'):
    """
    Decomposition of the linear transform between images expressed by the 2D matrix/tensor ``X``. It is assumed \
    2D since it is assumed that the underlying images are represented in vector (col- or row-major order) \
    representations.
    This version of the function, named *_vec_format*, returns the eigenvalues of the transform in (the same) \
    vector format (of the given transform).
    The type of the transform used will differ depending on the characteristics \
    of the transform so the results provided are given in real numbers:

    - Eigenvalue decomposition only applies to square matrices (that is, same size for input and output). Additionally,\
      the eigenvalue decomposition is real for real symmetric matrices. Thus, an initial check is performed in the
      function, and when ``X`` is (square and) *approximately* symmetric then the function :py:func:`torch.linalg.eigh`\
      is used. The check is performed using the function :py:func:`torch.allclose`, which uses the \
      arguments ``atol`` and ``rtol`` and which in turn justifies the inclusion of them as arguments of the \
      present function.

    - Singular value decomposition (SVD) applies to general matrices, and allows for a non-negative real-value diagonal\
      decomposition based on two real rotation-reflection matrices, respectively, in the input and output spaces
      (`SVD in Wikipedia <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_). The decomposition is calculated using \
      the function :py:func:`torch.linalg.svd`.

    Both cases will return the same triplet with identical format: (``U``, ``S``, and ``V``), wherein \
    $\\mathrm{X} = \\mathrm{U} \\mathrm{S} \\mathrm{V}^T$, wherein $\\mathrm{U}$ and $\\mathrm{V}$ are \
    orthogonal and $\\mathrm{S}$ is a 1D vector/tensor with the eigenvalues/singular values in \
    decreasing order of magnitude/abs. \
    The differences between both decompositions reside in that \
    *(i)* for the eigenvalue decomposition, $\\mathrm{U} = \\mathrm{V}$ and the length of \
    $\\mathrm{S}$ is the same as the side each of them, whereas for SVD they are not, and \
    *(ii)* when eigenvalue decomposition is used the eigenvalues might not be positive.

    Parameters
    ----------
    X : torch.Tensor
        2D tensor corresponding to a linear transform between images (e.g. the Jacobian matrix of \
        a given image transform) for either col-major or row-major order representations \
        of underlying images
    atol : float, optional
        Absolute tolerance for the symmetry assessment (see :py:func:`torch.allclose`). Default: ``1e-08``
    rtol : float, optional
        Relative tolerance for the symmetry assessment (see :py:func:`torch.allclose`). Default: ``1e-05``
    computation_device : str
        Value among ``'cuda'`` and ``'cpu'``. Default: ``cuda`` (if available)

    Returns
    -------
    U_vec : :py:class:`torch.Tensor`
        Left basis for the decomposition, in vector format: that is, 2D tensor
    S : :py:class:`torch.Tensor`
        1D corresponding to either the eigenvalues or the singular values
    V_vec : :py:class:`torch.Tensor`
        Right basis for the decomposition, in vector format: that is, 2D tensor
    """

    # Initial checks
    if X.ndim != 2:
        raise Exception(f"The function considers an input tensor of 2D: X of size {X.size()} found!")
    pass

    # Computation device and conversion to "dense" tensor if necessary
    device_result = X.device.type
    if computation_device not in ['cpu', 'cuda']:
        raise Exception(f"Allowed 'computation_device' are 'cpu', 'cuda': {computation_device} found!")
    pass
    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and computation_device == 'cuda':
        computation_device = 'cuda'
    else:
        computation_device = 'cpu'
    pass
    X = X.to(computation_device).to_dense()

    # Choose the type of computation
    flag_eigenvalue = True
    if X.size(-1) != X.size(-2):
        flag_eigenvalue = False
    elif not torch.allclose(X, torch.transpose(X, dim0=-1, dim1=-2), rtol=rtol, atol=atol):
        flag_eigenvalue = False
    pass

    if flag_eigenvalue:  # Eigenvalue decoposition
        print('EIGENVALUE decomposition!')
        S, U_vec = torch.linalg.eigh(X)
        #
        # The eigenvalue decomposition is not ordered by magnitude: we order it
        _, sorting_indices = torch.sort(S.abs(), descending=True)
        #
        S = S[sorting_indices]
        gc.collect()
        torch.cuda.empty_cache()
        #
        U_vec = U_vec[:, sorting_indices]
        gc.collect()
        torch.cuda.empty_cache()
        #
        V_vec = U_vec.to(device_result)
    else:  # SVD
        print('SVD decomposition!')
        U_vec, S, Vt_vec = torch.linalg.svd(X)
        #
        V_vec = torch.transpose(Vt_vec, dim0=-1, dim1=-2).to(device_result)
        del Vt_vec
        gc.collect()
        torch.cuda.empty_cache()
    pass
    #
    gc.collect()
    torch.cuda.empty_cache()
    #
    U_vec = U_vec.to(device_result)
    gc.collect()
    torch.cuda.empty_cache()
    #
    S = S.to(device_result)
    gc.collect()
    torch.cuda.empty_cache()

    return U_vec, S, V_vec


def real_value_decomposition_im_format(X, input_im_size, output_im_size=None,
                                       order='col', atol=1e-08, rtol=1e-05, computation_device='cuda'):
    """
    Decomposition of the linear transform between images expressed by the 2D matrix/tensor ``X``. It is assumed \
    2D since it is assumed that the underlying images are represented in vector (col- or row-major order) \
    representations.
    This version of the function, named *_im_format*, returns the eigenvalues of the transform in image format: \
    it uses internally the function :py:func:`.real_value_decomposition_vec_format` (see its documentation for further \
    details) and transforms the result into vector format, according to the provided \
    ``order``-major order, using :py:func:`.vec2im_format`.

    Parameters
    ----------
    X : torch.Tensor
        2D tensor corresponding to a linear transform between images (e.g. the Jacobian matrix of \
        a given image transform) for either col-major or row-major order representations \
        of underlying images
    input_im_size : list[int] or list[float] or tuple[int] or list[float] or torch.Size
        Size $(\\mathrm{C}, \\mathrm{H}, \\mathrm{W})$ of the input images of the linear transform in image format
    output_im_size : list[int] or list[float] or tuple[int] or list[float] or torch.Size, optional
        Size $(\\mathrm{C}, \\mathrm{H}, \\mathrm{W})$ of the output images of the linear transform in image format. \
        If no value (or if ``None``) is provided the same size provided for the input is presumed. \
        Default: ``None``
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``
    atol : float, optional
        Absolute tolerance for the symmetry assessment (see :py:func:`torch.allclose`). Default: ``1e-08``
    rtol : float, optional
        Relative tolerance for the symmetry assessment (see :py:func:`torch.allclose`). Default: ``1e-05``
    computation_device : str
        Value among ``'cuda'`` and ``'cpu'``. Default: ``cuda`` (if available)

    Returns
    -------
    U_im : :py:class:`torch.Tensor`
        Left basis for the decomposition, in im format: that is, a 4D tensor $(N_U, C, H_U, W_U)$
    S : :py:class:`torch.Tensor`
        1D corresponding to either the eigenvalues or the singular values
    V_im : :py:class:`torch.Tensor`
        Right basis for the decomposition, in vector format: that is, a 4D tensor $(N_V, C, H_V, W_V)$
    """

    # Result in *_vec_format:
    U_vec, S, V_vec = real_value_decomposition_vec_format(
        X, atol=atol, rtol=rtol, computation_device=computation_device
    )

    # If 'output_im_size' is None, we use the same size 'input_im_size' for the input
    output_im_size = input_im_size if output_im_size is None else output_im_size

    # transform to *_im_format:
    U_im = vec2im_format(U_vec, I_in_im_size=output_im_size, order=order)
    del U_vec
    gc.collect()
    torch.cuda.empty_cache()
    #
    V_im = vec2im_format(V_vec, I_in_im_size=input_im_size, order=order)
    del V_vec
    gc.collect()
    torch.cuda.empty_cache()

    return U_im, S, V_im


######################################################
### Extraction of full Jacobian matrix for a certain operation
######################################################


def full_jacobian_matrix(f_x, x, order='col'):
    """
    Calculates the full Jacobian matrix for the relationship $\\mathbf{y} = \\mathbf{f}(\\mathbf{x})$, wherein \
    said functional relationship is encoded (as black box) by a :py:class:`torch.nn.Module` (or another \
    `autograd`-friendly relationship).

    This function is relevant since `autograd` (as well as other frameworks) does not calculate the full Jacobian \
    but vector-Jacobian products (VJPs) instead. The full Jacobian matrix is therefore calculated by calculating \
    multiple VJPs and packing the result as a single matrix (tensor).

    **WARNING**: If either the tensor ``f_x`` or ``x`` is (or both are) 1D then the resulting Jacobian matrix \
    is a vector; in that case the returned tensor is a vector, *i.e.* a tensor of dimension 1.

    Parameters
    ----------
    f_x : torch.Tensor
        Description of f_x
    x : torch.Tensor
        Description of x

    Returns
    -------
    torch.Tensor
        Description of the returned Jacobian matrix
    """

    computation_device = x.device.type
    f_x = f_x.to(computation_device)

    list_of_gradients = []
    for i in range(f_x.numel()):
        x.grad = None
        f_x_i = torch.take(f_x, torch.tensor(i, device=computation_device))
        f_x_i.backward(retain_graph=True)
        list_of_gradients.append(x.grad.detach().clone())
    pass

    jacobian_matrix = torch.column_stack(list_of_gradients)

    ### Format the output depending on the chosen 'order' mode
    jacobian_vec_tensor = None
    if order == 'col':
        jacobian_vec_tensor = jacobian_matrix.transpose(dim0=-2, dim1=-1).transpose(dim0=1, dim1=2)
    else:
        jacobian_vec_tensor = jacobian_matrix
    pass
    jacobian_vec_tensor = jacobian_vec_tensor.reshape(
        (np.prod(jacobian_vec_tensor.size()[0:3]), np.prod(jacobian_vec_tensor.size()[3:]))
    )

    return jacobian_vec_tensor


###


################################################################################
### Extraction of full Jacobian matrix for a certain operation and input image
################################################################################


def autograd_jacobian_matrix_black_box_im_format(F, I):
    """
    Calculates the full Jacobian matrix for the relationship $\\mathrm{Z} = \\mathrm{F}(\\mathrm{I})$, wherein \
    said functional relationship is encoded (as black box) by a :py:class:`torch.nn.Module` (or another \
    `autograd`-friendly relationship). The format of the output Jacobian tensor is identical to that of \
    :py:func:`.manual_jacobian_matrix_black_box_im_format`: refer to the latter for further information.

    This function is relevant since `autograd` (as well as other frameworks) does not calculate the full Jacobian \
    but vector-Jacobian products (VJPs) instead. The full Jacobian matrix is therefore calculated by calculating \
    multiple VJPs and packing the result as a single matrix (tensor).

    Parameters
    ----------
    F : :py:class:`torch.nn.Module` or ~collections.abc.Callable
        The (image-to-image) function whose Jacobian is calculated. ``F`` can be a
        ~collections.abc.Callable, if allowing autograd; it can indeed be a :py:class:`.torch.nn.Module` object, \
        since its direct evaluation (using parenthesis) would call its method :py:meth:`.torch.nn.Module.forward`.
    I : torch.Tensor
        Input image at which the Jacobian is calculated.

    Returns
    -------
    jacobian_im_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output

    F_I : :py:class:`torch.Tensor`
        F(I)
    """

    ### Initial checks for the input
    if I.ndim != 3:
        raise Exception(f"The function presumes a single input image I, which corresponds to 3D: {I.size()} found!")
    pass

    # Calculation of F(I) JUST TO KNOW THE DEVICE OF THE FUNCTION/MODEL
    I.requires_grad = False
    F_I = F(I)

    # Computation device from the output of the model/function
    computation_device = F_I.device.type
    I = I.to(computation_device)
    I.requires_grad = True
    F_I = F(I)

    jacobian_im_tensor = torch.empty((F_I.size()) + (I.size()), device=computation_device)

    for ch in range(F_I.size(-3)):
        for row in range(F_I.size(-2)):
            for col in range(F_I.size(-1)):
                # We zero the gradients
                method_for_zeroing_gradients = getattr(F_I, "zero_grad", None)
                if method_for_zeroing_gradients is not None:
                    method_for_zeroing_gradients()
                I.grad = None
                # Calculate the gradient for the specific element
                F_I_ch_row_col = F_I[ch, row, col]
                F_I_ch_row_col.backward(retain_graph=True)
                jacobian_im_tensor[ch, row, col][:] = I.grad.detach().clone()[:]
            pass
        pass
    pass

    # We zero the gradients for the last time, again
    method_for_zeroing_gradients = getattr(F_I, "zero_grad", None)
    if method_for_zeroing_gradients is not None:
        method_for_zeroing_gradients()
    I.grad = None

    return jacobian_im_tensor.transpose(dim0=0, dim1=3).transpose(dim0=1, dim1=4).transpose(dim0=2, dim1=5), F_I


def autograd_jacobian_matrix_black_box_vec_format(F, I, order='col'):
    """
    Calculates the full Jacobian matrix (tensor) $\\mathrm{D}F$ for the function ``F``, that is, $F(\\mathrm{I})$, \
    transforming images into images, at the input image ``I``, **wherein the resulting Jacobian tensor is formatted
    as a 2D matrix corresponding to the input and output images vectorized as column-major order \
    (or, respectively, row-major order) even thought** ``F`` **and** ``I`` \
    **work with/are image (not vectorized) format**.

    This function is based directly on :py:func:`.autograd_jacobian_matrix_black_box_im_format` \
    (it is advisable to read its corresponding documentation for additional information) and simply \
    reorders its returned elements according to the selected vectorized image order ``order``. The order ``'col'``, \
    considered as default, traces column-wise each image channel; the order ``'row'``, instead, row-wise \
    (see `Wiki <https://en.wikipedia.org/w/index.php?title=Row-_and_column-major_order&oldid=1226393111>`_).

    **Warning:** **PyTorch appears to work naturally using raw-major order: this might have an impact \
    on the performance of this function if used in column-major order and depending on the application!**

    Parameters
    ----------
    F : ~collections.abc.Callable
        The (image-to-image) function whose Jacobian is calculated. ``F`` can indeed be a :py:class:`.torch.nn.Module` \
        object, since directly evaluating it using parenthesis would call its :py:meth:`.torch.nn.Module.forward`
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``


    Returns
    -------
    jacobian_vec_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output

    F_I : :py:class:`torch.Tensor`
        F(I)
    """

    ### Initial checks for the input
    if I.ndim != 3:
        raise Exception(f"The function presumes a single input image I, which corresponds to 3D: {I.size()} found!")
    pass

    ### Initial checks for the specific arguments for the function
    allowable_orders = ['col', 'row']
    if order not in ['col', 'row']:
        raise Exception(f"Allowable orders: {allowable_orders}. '{order}' found instead!")
    pass

    ### It calls 'manual_jacobian_matrix_black_box_im_format'
    jacobian_im_tensor, F_I = autograd_jacobian_matrix_black_box_im_format(F=F, I=I)

    with torch.no_grad():
        # The original 'jacobian_im_tensor' is ordered "according to modifications in the input": that is,
        # the tensor (C_in, H_in, W_in, C_out, H_out, W_out) stores, at [ch_in, h_in, w_in], the (infinitesimal)
        # modification of the (complete) output for a(n infinitesimal) variation in such pixel value.
        # We need now, for each final row r of the matrix, [dF_r/dI_0, dF_r/dI_1, ..., dF_r/dI_N]. We will solve this
        # with a final transposition.
        # Format the output depending on the chosen 'order' mode
        transposed_jacobian_vec_tensor = None
        if order == 'col':
            transposed_jacobian_vec_tensor = jacobian_im_tensor.transpose(dim0=-2, dim1=-1).transpose(dim0=1, dim1=2)
        else:
            transposed_jacobian_vec_tensor = jacobian_im_tensor
        pass
        transposed_jacobian_vec_tensor = transposed_jacobian_vec_tensor.reshape(
            (np.prod(transposed_jacobian_vec_tensor.size()[0:3]), np.prod(transposed_jacobian_vec_tensor.size()[3:]))
        )
        #
    pass

    return torch.t(transposed_jacobian_vec_tensor), F_I


def manual_jacobian_matrix_black_box_im_format(F, I, h=1e-3, sparse_result=False):
    """
    Calculates the full Jacobian matrix (tensor) $\\mathrm{D}F$ for the function ``F``, that is, $F(\\mathrm{I})$, \
    transforming images into images, at the input image ``I``. The provided ``F`` is treated purely as a black-box, \
    *i.e.* the Jacobian (matrix) of the operation is purely obtained through numerical methods and without any
    presumed knowledge about the inners of said mapping.  \
    The procedure presumes that the provided ~collections.abc.Callable ``F`` allows for \
    for batch calculation, and that ``I`` is one single image with size $(C_I, H_I, W_I)$.
    Partial derivatives are numerically calculated using the traditional 2-point estimator for the provided step ``h`` \
    as in
    $$
    \\frac{\\mathrm{d}f}{\\mathrm{d}x} \\approx \\frac{f(x+h)-f(x)}{h} \\, .
    $$

    The procedure, which keeps the image structure of input and output, produces an image of the same size \
    $(C_O, H_O, W_O)$ of $F(\\mathrm{I})$ for each and every channel-pixel of the entry, the image at each \
    channel-pixel meaning the (compensated infinitesimal change) caused in the complete output image by an \
    infinitesimal change of the corresponding channel-pixel of the input image. That is: the output of this function \
    is a tensor of size $(C_I, H_I, W_I, C_O, H_O, W_O)$, wherein the 3D image at the position $[ch, r, c]$ \
    is the partial derivative of $F$ with respect to the input element at $[ch, r, c]$. In other words,
    $$
    \\mathrm{Output}[ch_I, r_I, c_I, ch_O, r_O, c_O] =
    \\frac{\\partial F_{[ch_O, r_O, c_O]}(\\mathrm{I})}{\\partial \\mathrm{I}_{[ch_I, r_I, c_I]}} \\, .
    $$

    **Warning:** **PyTorch (e.g. with** :py:meth:`.torch.Tensor.view` **) \
    unwraps tensors first row-wise, then column-wise, and finally channel-wise**, and consequently the transform \
    of the output of this function to vectorized inputs/outputs might not be direct. \
    For a Jacobian formatted presuming vectorized (column-major or row-major) input and output images \
    see :py:func:`.manual_jacobian_matrix_black_box_vec_format`.

    Parameters
    ----------
    F : ~collections.abc.Callable
        The (image-to-image) function whose Jacobian is calculated. ``F`` can indeed be a :py:class:`.torch.nn.Module` \
        object, since directly evaluating it using parenthesis would call its :py:meth:`.torch.nn.Module.forward`
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    h : int
        Step for the 2-point estimator of the partial derivative. Default: ``1e-3``
    sparse_result : bool
        Output as sparse tensor or not. Default: ``False``

    Returns
    -------
    jacobian_im_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor (not matrix due to the tensor structure of input and output)

    F_I : :py:class:`torch.Tensor`
        F(I)
    """

    with torch.no_grad():
        #
        ### Initial checks for the input
        if I.ndim != 3:
            raise Exception(f"The function presumes a single input image I, which corresponds to 3D: {I.size()} found!")
        pass

        computation_device = I.device.type

        # We calculate the output of the input image
        F_I = F(I)

        # We infer the size of the resulting Jacobian tensor from its foreseeable size
        foreseeable_size = I.size() + F_I.size()
        memory_ratio = memo.fraction_desired_tensor_fitting_available_memory(foreseeable_size,
                                                                             computation_device=computation_device)
        if memory_ratio > 0.95:
            print(" ")
            memo.print_memory_progress()
            raise Exception((
                    f"The desired Jacobian tensor/matrix, of foreseeable size {foreseeable_size} "
                    f"and {memo.desired_tensor_memory_size(foreseeable_size, units='MB')} MB, would occupy " +
                    f"x{memory_ratio:.1f} % the available memory for the device {computation_device}"
            ))
        pass

        # Try 'fast', 'semi-fast', and 'slow' computation methods... sequentially, if the previous does not work

        flag_achieved_gradient_computation = False
        jacobian_im_tensor = None

        if not flag_achieved_gradient_computation:
            try:
                ### Try FAST GRADIENT COMPUTATION
                gc.collect()
                torch.cuda.empty_cache()

                ##########
                # FAST computation: one tensor block
                ##########
                ### Multiplication of the input image as many times as required to have the image of images (so far without h)
                expanded_I = I.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                expanded_I = expanded_I.repeat(I.size() + (1, 1, 1))

                ### Now add, to each image, noise h in the corresponding coordinate to obtain probe images
                delta_I = torch.eye(I.numel(), device=computation_device)
                delta_I = delta_I.view(I.size() + I.size())
                probe_I = expanded_I + h * delta_I

                # Delete the already useless variables
                del delta_I, expanded_I
                gc.collect()
                torch.cuda.empty_cache()

                # We calculate the output of the probe images.
                # Remember! We have an image of images (6D), but the function to evaluate addresses max. 4D (B, C, H, W).
                F_probe_I = F(
                    probe_I.view((-1,) + tuple(list(probe_I.size()[-3:])))
                )
                F_probe_I = F_probe_I.view(
                    tuple(list(probe_I.size()[0:-3])) + tuple(list(F_probe_I.size()[-3:]))
                )

                # Delete the already useless variables
                del probe_I
                gc.collect()
                torch.cuda.empty_cache()

                # Subtract the f(I) from the output of the probe images
                expanded_F_I = F_I.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                expanded_F_I = expanded_F_I.repeat(tuple(list(F_probe_I.size()[0:-3])) + (1, 1, 1))

                # Output tensor
                jacobian_im_tensor = 1 / h * (F_probe_I - expanded_F_I)
                flag_achieved_gradient_computation = True

                # Delete the already useless variables
                del F_probe_I, expanded_F_I
                gc.collect()
                torch.cuda.empty_cache()

                ### OK  FAST GRADIENT COMPUTATION
                flag_achieved_gradient_computation = True

            except torch.cuda.OutOfMemoryError as e:
                # print(f"\t--- LACK OF MEMORY IN {computation_device}, 'fast' gradient computation method ---")
                pass
            pass
        pass

        if not flag_achieved_gradient_computation:
            try:
                ### Try 'semi-fast' (= row-wise) gradient computation method
                gc.collect()
                torch.cuda.empty_cache()

                # It is similar but the batch of probe images correspond only to variations of the elements of one row
                # at a time
                #
                jacobian_im_tensor = torch.empty(foreseeable_size, device=computation_device)
                #
                expanded_I_row = I.unsqueeze(0)
                expanded_I_row = expanded_I_row.repeat((I.size(-1),) + (1, 1, 1))
                expanded_F_I_row = F_I.unsqueeze(0)
                expanded_F_I_row = expanded_F_I_row.repeat((I.size(-1),) + (1, 1, 1))
                #
                delta_I_row_vec = torch.zeros((I.size(-1), I.numel()), device=computation_device, dtype=torch.bool)
                delta_I_row_vec.fill_diagonal_(1.0)
                #
                for ch in range(I.size(-3)):
                    for row in range(I.size(-2)):
                        delta_I_row_im = delta_I_row_vec.clone().view((I.size(-1),) + I.size())
                        probe_I_row = expanded_I_row + h * delta_I_row_im
                        #
                        # Delete the already useless variables
                        del delta_I_row_im
                        gc.collect()
                        torch.cuda.empty_cache()
                        #
                        F_probe_I_row = F(probe_I_row)

                        # Delete the already useless variables
                        del probe_I_row
                        gc.collect()
                        torch.cuda.empty_cache()

                        jacobian_im_tensor_row = 1 / h * (F_probe_I_row - expanded_F_I_row)
                        jacobian_im_tensor[ch, row, :] = jacobian_im_tensor_row[:]
                        #
                        # Delete the already useless variables
                        del F_probe_I_row, jacobian_im_tensor_row
                        gc.collect()
                        torch.cuda.empty_cache()
                        #
                        # For the next iteration we roll one extra "row" the current "delta_I_row_vec"
                        delta_I_row_vec = delta_I_row_vec.roll(shifts=(0, I.size(-1)), dims=(0, 1))
                    pass
                    #
                    # Delete the already useless variables
                    gc.collect()
                    torch.cuda.empty_cache()
                pass
                #
                flag_achieved_gradient_computation = True
                #
                # Delete the already useless variables
                del expanded_I_row, expanded_F_I_row, delta_I_row_vec
                gc.collect()
                torch.cuda.empty_cache()

                ### OK  'semi-fast' (= row-wise) gradient computation method
                flag_achieved_gradient_computation = True

            except torch.cuda.OutOfMemoryError as e:
                # print(f"\t--- LACK OF MEMORY IN {computation_device}, 'semi-fast' gradient computation method ---")
                pass
            pass
            #
        pass

        if not flag_achieved_gradient_computation:
            try:
                ### Try 'slow' gradient computation method
                gc.collect()
                torch.cuda.empty_cache()
                #
                ##########
                # SLOW computation: loops
                ##########
                jacobian_im_tensor = torch.empty(foreseeable_size, device=computation_device)
                for ch in range(I.size(0)):
                    for row in range(I.size(1)):
                        for col in range(I.size(2)):
                            probe_I = I
                            probe_I[ch, row, col] = probe_I[ch, row, col] + h
                            F_probe_I = F(I)
                            jacobian_im_tensor[ch, row, col, :] = 1 / h * (F_probe_I - I)[:]
                            del probe_I, F_probe_I
                            torch.cuda.empty_cache()
                        pass
                    pass
                pass

                ### OK  'slow' gradient computation method
                flag_achieved_gradient_computation = True

            except torch.cuda.OutOfMemoryError as e:
                # print(f"\t--- LACK OF MEMORY IN {computation_device}, 'slow' gradient computation method ---")
                pass
            pass
            #
        pass
        #
        if not flag_achieved_gradient_computation:
            raise Exception((
                    f"None of 'fast', 'semi-fast', and 'slow' computation methods could be computed: " +
                    f"all of them caused 'torch.cuda.OutOfMemoryError' exceptions"
            ))
        pass
        #
    pass

    # Return a sparse version of the gradient (since they will be inherently sparse)
    if sparse_result:
        jacobian_im_tensor = jacobian_im_tensor.to_sparse()
        torch.cuda.empty_cache()
    pass

    return jacobian_im_tensor, F_I


def manual_jacobian_matrix_black_box_vec_format(F, I, h=1e-3, order='col', sparse_result=False):
    """
    Calculates the full Jacobian matrix (tensor) $\\mathrm{D}F$ for the function ``F``, that is, $F(\\mathrm{I})$, \
    transforming images into images, at the input image ``I``, **wherein the resulting Jacobian tensor is formatted
    as a 2D matrix corresponding to the input and output images vectorized as column-major order \
    (or, respectively, row-major order) even thought** ``F`` **and** ``I`` \
    **work with/are image (not vectorized) format**.

    This function is based directly on :py:func:`.manual_jacobian_matrix_black_box_im_format` \
    (it is advisable to read its corresponding documentation for additional information) and simply \
    reorders its returned elements according to the selected vectorized image order ``order``. The order ``'col'``, \
    considered as default, traces column-wise each image channel; the order ``'row'``, instead, row-wise \
    (see `Wiki <https://en.wikipedia.org/w/index.php?title=Row-_and_column-major_order&oldid=1226393111>`_).

    **Warning:** **PyTorch appears to work naturally using raw-major order: this might have an impact \
    on the performance of this function if used in column-major order and depending on the application!**

    Parameters
    ----------
    F : ~collections.abc.Callable
        The (image-to-image) function whose Jacobian is calculated. ``F`` can indeed be a :py:class:`.torch.nn.Module` \
        object, since directly evaluating it using parenthesis would call its :py:meth:`.torch.nn.Module.forward`
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    h : int, optional
        Step for the 2-point estimator of the partial derivative. Default: ``1e-3``
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``
    sparse_result : bool
        Output as sparse tensor or not. Default: ``False``


    Returns
    -------
    jacobian_vec_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output

    F_I : :py:class:`torch.Tensor`
        F(I)
    """

    ### Initial checks for the input
    if I.ndim != 3:
        raise Exception(f"The function presumes a single input image I, which corresponds to 3D: {I.size()} found!")
    pass

    ### Initial checks for the specific arguments for the function
    allowable_orders = ['col', 'row']
    if order not in ['col', 'row']:
        raise Exception(f"Allowable orders: {allowable_orders}. '{order}' found instead!")
    pass

    ### It calls 'manual_jacobian_matrix_black_box_im_format'
    jacobian_im_tensor, F_I = manual_jacobian_matrix_black_box_im_format(F=F, I=I, h=1e-3, sparse_result=False)

    with torch.no_grad():
        # The original 'jacobian_im_tensor' is ordered "according to modifications in the input": that is,
        # the tensor (C_in, H_in, W_in, C_out, H_out, W_out) stores, at [ch_in, h_in, w_in], the (infinitesimal)
        # modification of the (complete) output for a(n infinitesimal) variation in such pixel value.
        # We need now, for each final row r of the matrix, [dF_r/dI_0, dF_r/dI_1, ..., dF_r/dI_N]. We will solve this
        # with a final transposition.
        # Format the output depending on the chosen 'order' mode
        transposed_jacobian_vec_tensor = None
        if order == 'col':
            transposed_jacobian_vec_tensor = jacobian_im_tensor.transpose(dim0=-2, dim1=-1).transpose(dim0=1, dim1=2)
        else:
            transposed_jacobian_vec_tensor = jacobian_im_tensor
        pass
        transposed_jacobian_vec_tensor = transposed_jacobian_vec_tensor.reshape(
            (np.prod(transposed_jacobian_vec_tensor.size()[0:3]), np.prod(transposed_jacobian_vec_tensor.size()[3:]))
        )
        #
        transposed_jacobian_vec_tensor = torch.t(transposed_jacobian_vec_tensor)
    pass

    # Return a sparse version of the gradient (since they will be inherently sparse)
    if sparse_result:
        transposed_jacobian_vec_tensor = transposed_jacobian_vec_tensor.to_sparse()
    pass

    return transposed_jacobian_vec_tensor, F_I


def manual_jacobian_matrices_SM_vec_format(smlayer, I,
                                           h=1e-3, order='col',
                                           component_jacobians=False, sparse_result=False, device_result=None):
    """
    This function returns the different Jacobian matrices, evaluated at the respective input values, of the building \
    sub-operations composing the operation of the Standard Model (SM), to wit: $\\Phi$ and $A$, wherein

    $$\\mathrm{SM}(\\mathrm{I}) = \\Phi \\big( A(\\mathrm{I}) \\big) ,$$

    wherein, and including the dependency on \
    $\\Theta = (\\mathrm{M}_\\mathbf{p}, {b}_\\mathbf{p})$,

    $$
    \\Phi(\\mathrm{I})(\\mathbf{p}) = \\phi \\big( \\mathrm{I}(\\mathbf{p}) \\big) ,
    $$

    $$
    A(\\mathrm{I}; \\Theta)(\\mathbf{p}) =
    \\sum_{\\mathbf{r}} \\mathrm{M}_\\mathbf{p}(\\mathbf{r}) \\mathrm{I}(\\mathbf{p}) - {b}_\\mathbf{p} .
    $$

    The function returns three elements, wherein the latter is only calculated when the argument \
    ``component_jacobians``, by default ``False``, is set to ``True``: \
    the complete ``jacobian_vec_tensor`` of the complete transformation \
    (for images in vectorized format following ``order``-major order), the output ``F_I`` \
    of the complete transformation in image/matrix format, and a dictionary ``dict_component_jacobians`` \
    with the respective Jacobian matrices/derivatives of the composing operations \
    composed of the following fields:

    - ``DA``, corresponding to the Jacobian matrix of $A$ evaluated at $\\mathrm{I}$; and

    - ``DPhi``, corresponding to the Jacobian matrix of $\\Phi$ \
      evaluated at $A(\\mathrm{I}) - \\lambda B \\big( A(\\mathrm{I}) \\big)$; and

    - ``jacobian_vec_tensor_multiplied_manually`` (only if *b* is scalar), the Jacobian matrix \
      resulting from the manual multiplication of the component Jacobian matrices \
      according to its theoretical derivation.

    Parameters
    ----------
    smlayer : :py:class:`.SMLayer`
        The SMLayer for the calculation
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    h : int, optional
        Step for the 2-point estimator of the partial derivative. Default: ``1e-3``
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``
    component_jacobians : bool
        Required calculation of the dictionary ``dict_component_jacobians`` or not. Default: ``False``
    sparse_result : bool
        Output as sparse tensor or not. Default: ``False``
    device_result : str or None
        Value among ``None``, ``'cpu'``, ``'cuda'``: ``None`` respects the inherent output of the layer for which \
        the Jacobian matrices are calculated. Default: ``None``

    Returns
    -------
    jacobian_vec_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output
    sm_I : :py:class:`torch.Tensor`
        ``smlayer`` (``I``)
    dict_component_jacobians : dict[:py:class:`torch.Tensor`]
        Dictionary with the tensors ``DA``, ``DPhi`` corresponding to inputs-outputs in ``order``-major order, \
        and ``jacobian_vec_tensor_multiplied_manually`` (only if *b* is scalar)
    """

    # Check if
    if not isinstance(smlayer, SMLayer):
        raise Exception(f"'smlayer' must be of type SMLayer; however, it is of type {type(smlayer)}!")
    pass
    # Obtain computation device
    computation_device = smlayer.theta['m'].device.type
    I = I.to(computation_device)
    if device_result is None:
        device_result = computation_device
    pass

    # In order to perform the calculations of the Jacobians of all partial operations,
    # go backup-ing all matrices in CPU
    backup_device = 'cpu'

    with torch.no_grad():

        # Exit of the full layer
        jacobian_vec_tensor, sm_I = manual_jacobian_matrix_black_box_vec_format(
            smlayer, I, h=h, order=order, sparse_result=sparse_result
        )

        # Backup 'jacobian_vec_tensor' and 'sm_I'
        jacobian_vec_tensor = jacobian_vec_tensor.to(backup_device)
        sm_I = sm_I.to(backup_device)
        gc.collect()
        torch.cuda.empty_cache()

        if component_jacobians:

            # Populate the dictionary of Jacobians by calculating the different suboperations
            dict_component_jacobians = {}
            detached_smlayer_theta = {}
            for key in smlayer.theta:
                detached_smlayer_theta[key] = smlayer.theta[key].detach().clone().to(computation_device)
                detached_smlayer_theta[key].requires_grad = False
            pass

            ##########
            # For DA (corresponding to the Jacobian matrix of A evaluated at I)
            ##########

            def transformation_A(input):
                return conv2d_adapted(
                    input,
                    detached_smlayer_theta['m'], bias=-detached_smlayer_theta['b'],
                    padding=smlayer.m_padding, padding_mode=smlayer.m_padding_mode, groups=smlayer.m_groups
                )

            pass

            DA, A_I = manual_jacobian_matrix_black_box_vec_format(transformation_A, I, h=h,
                                                                  order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DA']' ('A_I' do not delete yet)
            dict_component_jacobians['DA'] = DA.to(backup_device)
            del DA
            gc.collect()
            torch.cuda.empty_cache()

            ##########
            # For DPhi (corresponding to the Jacobian matrix of Phi evaluated at A(I)):
            ##########
            def transformation_Phi(input):
                return smlayer.phi_activation(input)

            pass

            DPhi, Phi_I = manual_jacobian_matrix_black_box_vec_format(transformation_Phi, A_I, h=h,
                                                                      order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DPhi']' and delete 'Phi_I' and 'A_I'
            dict_component_jacobians['DPhi'] = DPhi.to(backup_device)
            del DPhi, Phi_I, A_I
            gc.collect()
            torch.cuda.empty_cache()

            dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = torch.matmul(
                dict_component_jacobians['DPhi'].to(computation_device),
                dict_component_jacobians['DA'].to(computation_device)
            )

            # Get to "sparse" if that was the choice
            if sparse_result and \
                    (not dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].is_sparse):
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                    dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to_sparse()
            pass
            gc.collect()
            torch.cuda.empty_cache()

            # Backup 'dict_component_jacobians['jacobian_vec_tensor_multiplied_manually']
            dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to(backup_device)
            gc.collect()
            torch.cuda.empty_cache()

            # Move to the device of the result
            for key in dict_component_jacobians:
                dict_component_jacobians[key] = dict_component_jacobians[key].to(device_result)
            pass
            torch.cuda.empty_cache()
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            sm_I.to(device_result),
                            dict_component_jacobians)
            #
        else:
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            sm_I.to(device_result))
        pass
        #
    pass

    return output_tuple


def manual_jacobian_matrices_ibnn_lite_vec_format(IBNNLiteLayer, I,
                                               h=1e-3, order='col',
                                               component_jacobians=False, sparse_result=False, device_result=None):
    """
    This function returns the different Jacobian matrices, evaluated at the respective input values, of the building \
    sub-operations composing the operation of ibnn_lite, to wit: $\\Phi$, $A$, and $B$, wherein

    $$\\mathrm{ibnn_lite}(\\mathrm{I}) = \\Phi \\Big( A(\\mathrm{I}) - \\lambda B \\big( A(\\mathrm{I}) \\big) \\Big) ,$$

    wherein, and including the dependency on \
    $\\Theta = (\\mathrm{M}_\\mathbf{p}, {b}_\\mathbf{p}, \\lambda, \\mathrm{\\Omega}_\\mathbf{p})$,

    $$
    \\Phi(\\mathrm{V})(\\mathbf{p}) = \\phi \\big( \\mathrm{V}(\\mathbf{p}) \\big) ,
    $$

    $$
    A(\\mathrm{I}; \\Theta)(\\mathbf{p}) =
    \\sum_{\\mathbf{r}} \\mathrm{M}_\\mathbf{p}(\\mathbf{r}) \\mathrm{I}(\\mathbf{p}) - {b}_\\mathbf{p} ,
    $$

    $$
    B(\\mathrm{U}; \\Theta)(\\mathbf{p}) =
    \\sum_{\\mathbf{r}} \\mathrm{\\Omega}_\\mathbf{p}(\\mathbf{r}) \\,
    \\mathbf{\\sigma} \\big( \\mathrm{U}(\\mathbf{r}) - \\mathrm{U}(\\mathbf{p}) \\big) .
    $$

    The function returns three elements, wherein the latter is only calculated when the argument \
    ``component_jacobians``, by default ``False``, is set to ``True``:
    the complete ``jacobian_vec_tensor`` of the complete transformation \
    (for images in vectorized format following ``order``-major order), the output ``F_I`` \
    of the complete transformation in image/matrix format, and a dictionary ``dict_component_jacobians`` \
    with the respective Jacobian matrices/derivatives of the composing operations \
    composed of the following fields:

    - ``DA``, corresponding to the Jacobian matrix of $A$ evaluated at $\\mathrm{I}$;

    - ``DB``, corresponding to the Jacobian matrix of $B$ evaluated at $A(\\mathrm{I})$; and

    - ``DPhi``, corresponding to the Jacobian matrix of $\\Phi$ \
      evaluated at $A(\\mathrm{I}) - \\lambda B \\big( A(\\mathrm{I}) \\big)$; and

    - ``jacobian_vec_tensor_multiplied_manually`` (only if $\\lambda$ and *b* are scalars), the Jacobian matrix \
      resulting from the manual multiplication of the component Jacobian matrices \
      according to its theoretical derivation.

    Parameters
    ----------
    IBNNLiteLayer : :py:class:`.IBNNLiteLayer`
        The IBNNLiteLayer for the calculation
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    h : int, optional
        Step for the 2-point estimator of the partial derivative. Default: ``1e-3``
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``
    component_jacobians : bool
        Required calculation of the dictionary ``dict_component_jacobians`` or not. Default: ``False``
    sparse_result : bool
        Output as sparse tensor or not. Default: ``False``
    device_result : str or None
        Value among ``None``, ``'cpu'``, ``'cuda'``: ``None`` respects the inherent output of the layer for which \
        the Jacobian matrices are calculated. Default: ``None``

    Returns
    -------
    jacobian_vec_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output
    ibnn_lite_I : :py:class:`torch.Tensor`
        ``IBNNLiteLayer`` (``I``)
    dict_component_jacobians : dict[:py:class:`torch.Tensor`]
        Dictionary with the tensors ``DA``, ``DB``, ``DPhi`` corresponding to inputs-outputs in ``order``-major order, \
        and ``jacobian_vec_tensor_multiplied_manually`` (only if $\\lambda$ and *b* are scalars)
    """

    # Check if
    if not isinstance(IBNNLiteLayer, IBNNLiteLayer):
        raise Exception(f"'IBNNLiteLayer' must be of type IBNNLiteLayer; however, it is of type {type(IBNNLiteLayer)}!")
    pass
    # Obtain computation device
    computation_device = IBNNLiteLayer.theta['m'].device.type
    I = I.to(computation_device)
    if device_result is None:
        device_result = computation_device
    pass

    # In order to perform the calculations of the Jacobians of all partial operations,
    # go backup-ing all matrices in CPU
    backup_device = 'cpu'

    with torch.no_grad():

        # Exit of the full layer
        jacobian_vec_tensor, ibnn_lite_I = manual_jacobian_matrix_black_box_vec_format(
            IBNNLiteLayer, I, h=h, order=order, sparse_result=sparse_result
        )

        # Backup 'jacobian_vec_tensor' and 'ibnn_lite_I'
        jacobian_vec_tensor = jacobian_vec_tensor.to(backup_device)
        ibnn_lite_I = ibnn_lite_I.to(backup_device)
        gc.collect()
        torch.cuda.empty_cache()

        if component_jacobians:

            # Populate the dictionary of Jacobians by calculating the different suboperations
            dict_component_jacobians = {}
            detached_IBNNLiteLayer_theta = {}
            for key in IBNNLiteLayer.theta:
                detached_IBNNLiteLayer_theta[key] = IBNNLiteLayer.theta[key].detach().clone().to(computation_device)
                detached_IBNNLiteLayer_theta[key].requires_grad = False
            pass

            ##########
            # For DA (corresponding to the Jacobian matrix of A evaluated at I)
            ##########

            def transformation_A(input):
                return conv2d_adapted(
                    input,
                    detached_IBNNLiteLayer_theta['m'], bias=-detached_IBNNLiteLayer_theta['b'],
                    padding=IBNNLiteLayer.m_padding, padding_mode=IBNNLiteLayer.m_padding_mode, groups=IBNNLiteLayer.m_groups
                )

            pass

            DA, A_I = manual_jacobian_matrix_black_box_vec_format(transformation_A, I, h=h,
                                                                  order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DA']' ('A_I' do not delete yet)
            dict_component_jacobians['DA'] = DA.to(backup_device)
            del DA
            gc.collect()
            torch.cuda.empty_cache()

            ##########
            # For DB (corresponding to the Jacobian matrix of B evaluated at A_I)
            ##########

            def transformation_B(input):
                return conv2d_crossdiff(
                    input,
                    weight=detached_IBNNLiteLayer_theta['w'], activation_function=IBNNLiteLayer.sigma_activation,
                    padding=IBNNLiteLayer.w_padding, padding_mode=IBNNLiteLayer.w_padding_mode, groups=IBNNLiteLayer.w_groups,
                    calculation_mode=IBNNLiteLayer.calculation_mode,
                    memory_saving_version=IBNNLiteLayer.memory_saving_version,
                    **IBNNLiteLayer._fixed_kwargs
                )

            pass

            DB, B_I = manual_jacobian_matrix_black_box_vec_format(transformation_B, A_I, h=h,
                                                                  order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DB']' and delete 'B_I' ('A_I' not yet)
            dict_component_jacobians['DB'] = DB.to(backup_device)
            del DB, B_I
            gc.collect()
            torch.cuda.empty_cache()

            ##########
            # For DPhi (corresponding to the Jacobian matrix of Phi evaluated at A(I)-lambda*B(A(I)):
            ##########

            def transformation_Phi(input):
                return IBNNLiteLayer.phi_activation(input)

            pass

            A_I_minus_lambda_B = f_modified_RF(
                I, detached_IBNNLiteLayer_theta, u=A_I,
                phi_activation='identity', sigma_activation=IBNNLiteLayer.sigma_activation,
                m_padding=IBNNLiteLayer.m_padding, m_padding_mode=IBNNLiteLayer.m_padding_mode,
                m_groups=IBNNLiteLayer.m_groups,
                w_padding_mode=IBNNLiteLayer.w_padding_mode, w_groups=IBNNLiteLayer.w_groups,
                calculation_mode=IBNNLiteLayer.calculation_mode, memory_saving_version=IBNNLiteLayer.memory_saving_version,
                **IBNNLiteLayer._fixed_kwargs
            )

            # Delete 'A_I'
            del A_I
            gc.collect()
            torch.cuda.empty_cache()

            DPhi, Phi_I = manual_jacobian_matrix_black_box_vec_format(transformation_Phi, A_I_minus_lambda_B, h=h,
                                                                      order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DPhi']' and delete 'Phi_I'
            dict_component_jacobians['DPhi'] = DPhi.to(backup_device)
            del DPhi, Phi_I
            gc.collect()
            torch.cuda.empty_cache()

            if detached_IBNNLiteLayer_theta['lambda'].numel() == 1:
                #
                identity_tensor = torch.eye(dict_component_jacobians['DB'].size(-1), device=computation_device)
                if sparse_result:
                    identity_tensor = identity_tensor.to_sparse()
                pass
                #
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = torch.matmul(
                    torch.matmul(
                        dict_component_jacobians['DPhi'].to(computation_device),
                        identity_tensor - \
                        detached_IBNNLiteLayer_theta['lambda'].item() * \
                        dict_component_jacobians['DB'].to(computation_device)
                    ),
                    dict_component_jacobians['DA'].to(computation_device)
                )

                # Get to "sparse" if that was the choice
                if sparse_result and \
                        (not dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].is_sparse):
                    dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                        dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to_sparse()
                pass
                gc.collect()
                torch.cuda.empty_cache()

                # Backup 'dict_component_jacobians['jacobian_vec_tensor_multiplied_manually']
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                    dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to(backup_device)
                gc.collect()
                torch.cuda.empty_cache()
                #
            pass

            # Move to the device of the result
            for key in dict_component_jacobians:
                dict_component_jacobians[key] = dict_component_jacobians[key].to(device_result)
            pass
            torch.cuda.empty_cache()
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            ibnn_lite_I.to(device_result),
                            dict_component_jacobians)
            #
        else:
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            ibnn_lite_I.to(device_result))
        pass
        #
    pass

    return output_tuple


def manual_jacobian_matrices_IBNN_vec_format(IBNNInternalLayer, I,
                                             h=1e-3, order='col',
                                             component_jacobians=False, sparse_result=False, device_result=None):
    """
    This function returns the different Jacobian matrices, evaluated at the respective input values, of the building \
    sub-operations composing the operation of ibnn_internal, to wit: $\\Phi$, $A$, and $B$, wherein


    In greater detail: the later obtains the fixed point solution \

    $$\\mathrm{ibnn_internal}(\\mathrm{I}) = \\mathrm{U}^* | \\mathrm{U}^* = F_(\\mathbf{I}, \\mathrm{U}^*; \\Theta)$$

    for

    $$
    F(\\mathrm{I}, \\mathrm{U}; \\Theta) = \\Phi \\Big(A(\\mathrm{I}) - \\lambda B \\big(A(\\mathrm{I})\\big)\\Big) .
    $$

    wherein, and including the dependency on \



    $$
    \\Phi(\\mathrm{V})(\\mathbf{p}) = \\phi \\big( \\mathrm{V}(\\mathbf{p}) \\big) ,
    $$

    $$
    A(\\mathrm{I}; \\Theta)(\\mathbf{p}) =
    \\sum_{\\mathbf{r}} \\mathrm{M}_\\mathbf{p}(\\mathbf{r}) \\mathrm{I}(\\mathbf{p}) - {b}_\\mathbf{p} ,
    $$

    $$
    B(\\mathrm{U}; \\Theta)(\\mathbf{p}) =
    \\sum_{\\mathbf{r}} \\mathrm{\\Omega}_\\mathbf{p}(\\mathbf{r}) \\,
    \\mathbf{\\sigma} \\big( \\mathrm{U}(\\mathbf{r}) - \\mathrm{U}(\\mathbf{p}) \\big) .
    $$

    The function returns three elements, wherein the latter is only calculated when the argument \
    ``component_jacobians``, by default ``False``, is set to ``True``: \
    the complete ``jacobian_vec_tensor`` of the complete transformation \
    (for images in vectorized format following ``order``-major order), the output ``F_I`` \
    of the complete transformation in image/matrix format, and a dictionary ``dict_component_jacobians`` \
    with the respective Jacobian matrices/derivatives of the composing operations \
    composed of the following fields:

    - ``DA``, corresponding to the Jacobian matrix of $A$ evaluated at $\\mathrm{I}$;

    - ``DB``, corresponding to the Jacobian matrix of $B$ evaluated at $A(\\mathrm{I})$; and

    - ``DPhi``, corresponding to the Jacobian matrix of $\\Phi$ \
      evaluated at $A(\\mathrm{I}) - \\lambda B \\big( A(\\mathrm{I}) \\big)$; and

    - ``jacobian_vec_tensor_multiplied_manually`` (only if $\\lambda$ and *b* are scalars), the Jacobian matrix \
      resulting from the manual multiplication of the component Jacobian matrices \
      according to its theoretical derivation.

    Parameters
    ----------
    IBNNInternalLayer : :py:class:`.IBNNInternalLayer`
        The IBNNLiteLayer for the calculation
    I : torch.Tensor
        Input image at which the Jacobian is calculated.
    h : int, optional
        Step for the 2-point estimator of the partial derivative. Default: ``1e-3``
    order : str, optional
        Value among  ``‘col’`` and ``‘row’``. Default: ``‘col’``
    component_jacobians : bool
        Required calculation of the dictionary ``dict_component_jacobians`` or not. Default: ``False``
    sparse_result : bool
        Output as sparse tensor or not. Default: ``False``
    device_result : str or None
        Value among ``None``, ``'cpu'``, ``'cuda'``: ``None`` respects the inherent output of the layer for which \
        the Jacobian matrices are calculated. Default: ``None``

    Returns
    -------
    jacobian_vec_tensor : :py:class:`torch.Tensor`
        Returned Jacobian tensor: matrix due to the vector structure of input and output
    U_ast : :py:class:`torch.Tensor`
        ``ibnn_internal`` (``I``)
    dict_component_jacobians : dict[:py:class:`torch.Tensor`]
        Dictionary with the tensors ``DA``, ``DB``, ``DPhi`` corresponding to inputs-outputs in ``order``-major order, \
        and ``jacobian_vec_tensor_multiplied_manually`` (only if $\\lambda$ and *b* are scalars)
    """

    # Check if
    if not isinstance(IBNNInternalLayer, IBNNInternalLayer):
        raise Exception(f"'IBNNInternalLayer' must be of type IBNNInternalLayer; however, it is of type {type(IBNNInternalLayer)}!")
    pass
    # Obtain computation device
    computation_device = IBNNInternalLayer.theta['m'].device.type
    I = I.to(computation_device)
    if device_result is None:
        device_result = computation_device
    pass

    # In order to perform the calculations of the Jacobians of all partial operations,
    # go backup-ing all matrices in CPU
    backup_device = 'cpu'

    with torch.no_grad():

        # Exit of the full layer
        jacobian_vec_tensor, U_ast = manual_jacobian_matrix_black_box_vec_format(
            IBNNInternalLayer, I, h=h, order=order, sparse_result=sparse_result
        )

        # Backup 'jacobian_vec_tensor': 'U_ast' not yet
        jacobian_vec_tensor = jacobian_vec_tensor.to(device_result)
        gc.collect()
        torch.cuda.empty_cache()

        if component_jacobians:

            # Populate the dictionary of Jacobians by calculating the different suboperations
            dict_component_jacobians = {}
            detached_ibnnlayer_theta = {}
            for key in IBNNInternalLayer.theta:
                detached_ibnnlayer_theta[key] = IBNNInternalLayer.theta[key].detach().clone().to(computation_device)
                detached_ibnnlayer_theta[key].requires_grad = False
            pass

            ##########
            # For DA (corresponding to the Jacobian matrix of A evaluated at I)
            ##########

            def transformation_A(input):
                return conv2d_adapted(
                    input,
                    detached_ibnnlayer_theta['m'], bias=-detached_ibnnlayer_theta['b'],
                    padding=IBNNInternalLayer.m_padding, padding_mode=IBNNInternalLayer.m_padding_mode, groups=IBNNInternalLayer.m_groups
                )

            pass

            DA, A_I = manual_jacobian_matrix_black_box_vec_format(transformation_A, I, h=h,
                                                                  order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DA']' and delete 'A_I'
            dict_component_jacobians['DA'] = DA.to(backup_device)
            del DA, A_I
            gc.collect()
            torch.cuda.empty_cache()

            ##########
            # For DB (corresponding to the Jacobian matrix of B evaluated at U_ast)
            ##########

            def transformation_B(input):
                return conv2d_crossdiff(
                    input,
                    weight=detached_ibnnlayer_theta['w'], activation_function=IBNNInternalLayer.sigma_activation,
                    padding='same', padding_mode=IBNNInternalLayer.w_padding_mode, groups=IBNNInternalLayer.w_groups,
                    calculation_mode=IBNNInternalLayer.calculation_mode,
                    memory_saving_version=IBNNInternalLayer.memory_saving_version,
                    **IBNNInternalLayer._fixed_kwargs
                )

            pass

            DB, B_U_ast = manual_jacobian_matrix_black_box_vec_format(transformation_B, U_ast, h=h,
                                                                      order=order, sparse_result=sparse_result)
            # Backup 'dict_component_jacobians['DB']' and delete 'B_U_ast'
            dict_component_jacobians['DB'] = DB.to(backup_device)
            del DB, B_U_ast
            gc.collect()
            torch.cuda.empty_cache()

            ##########
            # For DPhi (corresponding to the Jacobian matrix of Phi evaluated at A(I)-lambda*B(U*):
            ##########

            def transformation_Phi(input):
                return IBNNInternalLayer.phi_activation(input)

            pass

            A_I_minus_lambda_B = f_modified_RF(
                I, detached_ibnnlayer_theta, u=U_ast,
                phi_activation='identity', sigma_activation=IBNNInternalLayer.sigma_activation,
                m_padding=IBNNInternalLayer.m_padding, m_padding_mode=IBNNInternalLayer.m_padding_mode, m_groups=IBNNInternalLayer.m_groups,
                w_padding_mode=IBNNInternalLayer.w_padding_mode, w_groups=IBNNInternalLayer.w_groups,
                calculation_mode=IBNNInternalLayer.calculation_mode, memory_saving_version=IBNNInternalLayer.memory_saving_version,
                **IBNNInternalLayer._fixed_kwargs
            )

            DPhi, Phi_I_U_ast = manual_jacobian_matrix_black_box_vec_format(transformation_Phi, A_I_minus_lambda_B, h=h,
                                                                            order=order, sparse_result=sparse_result)

            # Backup 'dict_component_jacobians['DPhi']' and delete 'Phi_I_U_ast'
            dict_component_jacobians['DPhi'] = DPhi.to(backup_device)
            del DPhi, Phi_I_U_ast
            gc.collect()
            torch.cuda.empty_cache()

            if detached_ibnnlayer_theta['lambda'].numel() == 1:
                #
                identity_tensor = torch.eye(dict_component_jacobians['DB'].size(-1), device=computation_device)
                if sparse_result:
                    identity_tensor = identity_tensor.to_sparse()
                pass
                #
                print(f"----- Pre-multiplication -----")
                memo.print_memory_progress()
                #
                dict_component_jacobians['K'] = torch.inverse(
                    (identity_tensor + \
                     detached_ibnnlayer_theta['lambda'].item() * torch.matmul(
                                dict_component_jacobians['DPhi'].to(computation_device),
                                dict_component_jacobians['DB'].to(computation_device)
                            )
                     ).to_dense()
                )
                gc.collect()
                torch.cuda.empty_cache()
                #
                print(f"----- Pre-sparse -----")
                memo.print_memory_progress()
                #
                # Backup 'dict_component_jacobians['K']'
                if sparse_result:
                    dict_component_jacobians['K'] = dict_component_jacobians['K'].to_sparse().to(backup_device)
                else:
                    dict_component_jacobians['K'] = dict_component_jacobians['K'].to(backup_device)
                pass
                gc.collect()
                torch.cuda.empty_cache()
                #
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = torch.matmul(
                    dict_component_jacobians['K'].to(computation_device),
                    torch.matmul(
                        dict_component_jacobians['DPhi'].to(computation_device),
                        dict_component_jacobians['DA'].to(computation_device)
                    ).to_dense()
                )
                #
                print(f"----- After-multiplication -----")
                memo.print_memory_progress()
                #
                gc.collect()
                torch.cuda.empty_cache()
                #
                print(f"----- After cleaning, before sparse -----")
                memo.print_memory_progress()
                #
                # Get to "sparse" if that was the choice
                if sparse_result and \
                        (not dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].is_sparse):
                    #
                    print(f"----- Just before getting '.to_sparse()' -----")
                    memo.print_memory_progress()
                    #
                    dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                        dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to_sparse()
                pass
                gc.collect()
                torch.cuda.empty_cache()

                # Backup 'dict_component_jacobians['jacobian_vec_tensor_multiplied_manually']
                dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'] = \
                    dict_component_jacobians['jacobian_vec_tensor_multiplied_manually'].to(backup_device)
                gc.collect()
                torch.cuda.empty_cache()
                #
            pass

            # Move to the device of the result
            for key in dict_component_jacobians:
                dict_component_jacobians[key] = dict_component_jacobians[key].to(device_result)
            pass
            torch.cuda.empty_cache()
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            U_ast.to(device_result),
                            dict_component_jacobians)
            #
        else:
            #
            output_tuple = (jacobian_vec_tensor.to(device_result),
                            U_ast.to(device_result))
        pass
        #
    pass

    return output_tuple
