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
import numpy as np
import warnings


def _memory_size_from_byte_to_prefix(memory_size, units='MB'):
    """
    Turn a number in bytes into a number in another unit multiplier.

    Parameters
    ----------
    units : str
        Value among ``'B'``, ``'KB'``, ``'MB'``, and ``'GB'``. Default: ``'MB'``

    Returns
    -------
    float
    """
    factor = 1
    if units == 'B':
        factor = 1
    elif units == 'KB':
        factor = 1024
    elif units == 'MB':
        factor = 1024 * 1024
    elif units == 'GB':
        factor = 1024 * 1024 * 1024
    else:
        raise Exception(f"Unknown 'units' argument received: {units}")
    pass
    #
    return float(float(memory_size) / float(factor))


######################################################
######################################################
#
# Memory tracers
#
######################################################
######################################################

def print_memory_progress(units='MB'):
    """

    Parameters
    ----------
    computation_device : str
    units : str
        Value among ``'B'``, ``'KB'``, ``'MB'``, and ``'GB'``. Default: ``'MB'``
    """
    occupied_memory = _memory_size_from_byte_to_prefix(
        torch.cuda.memory_allocated(), units=units
    )
    free_memory_unformatted, existing_memory_unformatted = torch.cuda.mem_get_info()
    free_memory = _memory_size_from_byte_to_prefix(free_memory_unformatted, units=units)
    existing_memory = _memory_size_from_byte_to_prefix(existing_memory_unformatted, units=units)

    dict_extra_offset = {'B': 18, 'KB': 15, 'MB': 12, 'GB': 9}
    str_format = f"{dict_extra_offset[units]}.2f"

    print(f"Occupied memory: {occupied_memory:{str_format}} {units} (device: 'cuda')")
    print(f"Free memory:     {free_memory:{str_format}} {units} (device: 'cuda')")
    print(f"Total memory:    {existing_memory:{str_format}} {units} (device: 'cuda')")
    # print(f"Reserved memory: {reserved_memory:{str_format}} {units} (device: {computation_device})")


def tensor_memory_size(query_tensor, units='B'):
    """
    Estimated tensor size. It provides three values: *(1)* real (approximated) memory occupancy; *(2)* type of the \
    tensor ``is_sparse``: 'dense' (``False``) or 'sparse' (``True``); *(3)* (approximated) memory occupancy if dense. \
    Both memory values provided are in ``'units'``.

    - If ``query_tensor`` is dense: (1) and (3) are coincident.

    - If ``query_tensor`` is sparse, two results: (1) approximate memory that its composing structures actually occupy.

    Parameters
    ----------
    query_tensor : torch.Tensor
    units : str
        Value among ``'B'``, ``'KB'``, ``'MB'``, and ``'GB'``. Default: ``'MB'``

    Returns
    -------
    memory_occupancy_in_units : float
    is_sparse : bool
    memory_occupancy_if_dense_in_units : float
    """

    is_sparse = query_tensor.is_sparse

    memory_occupancy_if_dense = query_tensor.element_size()*query_tensor.nelement()

    if not is_sparse:
        memory_occupancy = memory_occupancy_if_dense
    else:
        layout = query_tensor.layout
        if layout == torch.sparse_coo:  # no row or col compression: methods .values(), .indices()
            memory_occupancy = \
                query_tensor.values().element_size() * query_tensor.values().numel() + \
                query_tensor.indices().element_size() * query_tensor.indices().numel()
        elif layout in [torch.sparse_csc, torch.sparse_bsc]:
            # col-compressed: methods .values(), .ccol_indices(), .row_indices()
            memory_occupancy = \
                query_tensor.values().element_size() * query_tensor.values().numel() + \
                query_tensor.ccol_indices().element_size() * query_tensor.ccol_indices().numel() + \
                query_tensor.row_indices().element_size() * query_tensor.row_indices().numel()
        elif layout in [torch.sparse_csr, torch.sparse_bsr]:
            # row-compressed: methods .values(), .col_indices(), .crow_indices()
            memory_occupancy = \
                query_tensor.values().element_size() * query_tensor.values().numel() + \
                query_tensor.col_indices().element_size() * query_tensor.col_indices().numel() + \
                query_tensor.crow_indices().element_size() * query_tensor.crow_indices().numel()
        else:
            warnings.warn(f"Unknown type of sparse tensor layout: {layout} found. Estimated size: as dense.")
            memory_occupancy = memory_occupancy_if_dense
        pass
    pass

    return (_memory_size_from_byte_to_prefix(memory_occupancy, units=units),
            is_sparse,
            _memory_size_from_byte_to_prefix(memory_occupancy_if_dense, units=units))


def print_tensor_memory_size(query_tensor, units='MB'):
    """
    Estimated tensor size... if dense: if ``query_tensor`` is sparse the result would indicate the memory that the \
    tensor, with its defined data type and size, would occupy as dense.

    Parameters
    ----------
    query_tensor : torch.Tensor
    units : str
        Value among ``'B'``, ``'KB'``, ``'MB'``, and ``'GB'``. Default: ``'MB'``

    """
    real_memory, is_sparse, memory_as_dense = tensor_memory_size(query_tensor, units=units)
    sparsity_str = 'sparse' if is_sparse else 'dense'
    print((f"Tensor size: real, {real_memory:.2f} {units} {sparsity_str} tensor, " +
           f"{memory_as_dense:.2f} {units} as dense (device: {query_tensor.device.type})"))


######################################################
######################################################
#
# Memory planning
#
######################################################
######################################################

def desired_tensor_memory_size(query_tensor_size, torch_dtype=None, units='B'):
    """
    It returns the memory space that a potential tensor of size ``query_tensor_size`` and torch.dtype ``torch_dtype``, \
    dense, would require. If no ``torch_dtype`` is provided the current :py:meth:`.torch.get_default_dtype` is presumed.

    Parameters
    ----------
    query_tensor_size : int or float or list[int] or list[float] or tuple[int] or list[float] or torch.Size
    torch_dtype : torch.dtype
    units : str
        Value among ``'B'``, ``'KB'``, ``'MB'``, and ``'GB'``. Default: ``'MB'``

    Returns
    -------
    int

    """
    #
    if torch_dtype is None:
        torch_dtype = torch.get_default_dtype()
    pass
    element_dtype_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    #
    num_elements_query_tensor = None
    if isinstance(query_tensor_size, int) or isinstance(query_tensor_size, float):
        num_elements_query_tensor = int(query_tensor_size)
    elif isinstance(query_tensor_size, torch.Size) or isinstance(query_tensor_size, tuple) or \
            isinstance(query_tensor_size, list):
        num_elements_query_tensor = np.array(list(query_tensor_size)).prod()
    else:
        print(f"Unacceptable type for 'query_tensor_size': {type(query_tensor_size)}")
    pass
    #
    return _memory_size_from_byte_to_prefix(element_dtype_bytes * num_elements_query_tensor,
                                            units=units)


def fraction_desired_tensor_fitting_available_memory(
        query_tensor_size, torch_dtype=None, computation_device='cuda'
):
    """
    It returns the ratio between the memory that the desired tensor would occupy, calculated internally using the \
    function :py:func:`.desired_tensor_memory_size`, and the available memory in the indicated computation device. \
    The returned ratio provides a guidance about the need for resizing or reducing the type resolution of the desired \
    tensor.

    Parameters
    ----------
    query_tensor_size : int or float or list[int] or list[float] or tuple[int] or list[float] or torch.Size
    torch_dtype : torch.dtype
    computation_device : str
        Default: ``'cuda'``

    Returns
    -------
    float

    """
    #
    if computation_device == 'cpu':
        ratio_memory_desired_tensor2free_memory = 0.0
    else:
        memory_desired_tensor = desired_tensor_memory_size(query_tensor_size, torch_dtype=torch_dtype, units='B')
        free_memory, _ = torch.cuda.mem_get_info()
        #
        ratio_memory_desired_tensor2free_memory = float(memory_desired_tensor) / float(free_memory)
    pass

    return ratio_memory_desired_tensor2free_memory
