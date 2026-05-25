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

import os
import gc
import datetime
import copy
import inspect
import socket
# import collections
import re

import numpy as np
import pandas as pd
import math
from random import randint

import torch
import torch.nn.functional as F


import mlflow
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.lines as mlines


from experimental_evaluation.interaction_with_mlflow import (connect_to_mlflow,
                                                             artifact_uri_from_experiment_and_run,
                                                             recreate_classifier_from_run_artifacts_uri,
                                                             experiment_metrics_for_run,
                                                             load_same_dataset_of_run_artifacts_uri)
from experimental_evaluation.operations_for_datasets import (obtain_classification_dataset_loaders_from_point_data,
                                                             _multiclass_mask_from_image,
                                                             _find_image_custom_dataset)
from experimental_evaluation.experiment_utils import classifier_training, formatted_log_base_name
from modified_rf import SMLayer
from applications import _dict_classifiers_as_in_conf_file



##############################################################################################################
##############################################################################################################
##############################################################################################################


###################################################################################################
# Creation of a string describing the real state (parameters) of a certain parameter of a classifier
###################################################################################################


def _text_from_real_param_from_classifier(classifier_nn:torch.nn.Module,
                                          param_name: str):
    """
    Parameters
    ----------
    classifier_nn : torch.nn.Module
    param_name: str
        For now we accept only "p", "lambda", and "b"

    Returns
    -------
    str
    """

    assert isinstance(classifier_nn, torch.nn.Module), \
        f"'classifier_nn' must be a torch.nn.Module: {type(classifier_nn)} instead!"
    assert isinstance(param_name, str), \
        f"'param_name' must be a string: {param_name} instead!"

    dict_parameter_names = {
        "lambda": "lambda",
        "p": "sigma_x_compress",
        "b": "b"
    }
    assert param_name in dict_parameter_names, \
        f"'param_name' must be one of " + ", ".join(list(dict_parameter_names.keys())) + f"; {param_name} provided!"

    ################################################################
    # Extract the name of the conv_i layers of the classifier
    ################################################################

    pattern = re.compile("block_conv_[0-9]+.conv_[0-9]+")
    list_conv_layer_names = []
    for (name, param) in classifier_nn._nn.backbone.named_parameters():
        match = pattern.match(name)
        if match is not None:
            list_conv_layer_names.append(match.group(0))
        pass
    pass
    list_conv_layer_names = list(set(list_conv_layer_names))
    list_conv_layer_names.sort()

    ################################################################
    # Extract the parameter of each conv_i layer corresponding to the desired param name
    ##########################+######################################

    dict_parameters_of_interest = {}
    for conv_layer_name in list_conv_layer_names:
        key = conv_layer_name + "._theta." + dict_parameter_names[param_name]
        dict_parameters_of_interest[key] = None
    pass
    for (name, param) in classifier_nn._nn.backbone.named_parameters():
        if name in dict_parameters_of_interest:
            dict_parameters_of_interest[name] = param.detach().clone().cpu()
        pass
    pass

    ################################################################
    # Format the result
    ################################################################

    dict_strings_of_interest = {}
    for key in dict_parameters_of_interest:
        if dict_parameters_of_interest[key] is None:
            dict_strings_of_interest[key] = "-"
        elif isinstance(dict_parameters_of_interest[key], torch.Tensor):
            dict_strings_of_interest[key] = ", ".join([f"{e.item():.3f}" for e in dict_parameters_of_interest[key].flatten()])
        else:
            raise Exception(f"Unknown content type: {type(dict_parameters_of_interest[key])}")
    pass

    formatted_params = " | ".join(list(dict_strings_of_interest.values()))
    if formatted_params == "":
        formatted_params = None
    pass

    return formatted_params


###################################################################################################
# Creating of a string describing a classifier
###################################################################################################


def _text_description_from_classifier(classifier_nn:torch.nn.Module,
                                      flag_for_title:bool=False,
                                      flag_conv_like_type:bool=True,
                                      flag_num_layers:bool=True, flag_num_neurons_per_layer:bool=True,
                                      flag_initial_lambda:bool=None, flag_real_lambda:bool=None,
                                      flag_initial_p:bool=None, flag_real_p:bool=None
                                      ):
    """
    Parameters
    ----------
    classifier_nn : torch.nn.Module
    flag_for_title : bool, optional
        Default: ``False``
    flag_conv_like_type : bool, optional
        Default: ``True``
    flag_num_layers : bool, optional
        Default: ``True``
    flag_num_neurons_per_layer : bool, optional
        Default: ``True``
    flag_initial_lambda : bool, optional
        Its default state depends on whether `flag_for_title` is ``True`` or ``False``: \
        if `flag_for_title` if ``True`` then it is ``False``; \
        if `flag_for_title` if ``False`` then it is ``True``;.
        Default: ``None``
    flag_real_lambda : bool, optional
        Unlike `flag_initial_lambda`, which refers to the lambda indicated in the constructor of the net, \
        this flag reads the lambda from the trained parameters of the net. \
        Its default state depends on whether `flag_for_title` is ``True`` or ``False`` and is inverse to that \
        of `flag_initial_lambda`:
        if `flag_for_title` if ``True`` then it is ``True``; \
        if `flag_for_title` if ``False`` then it is ``False``;.
        Default: ``None``
    flag_initial_p : bool, optional
        Its default state depends on whether `flag_for_title` is ``True`` or ``False``: \
        if `flag_for_title` if ``True`` then it is ``False``; \
        if `flag_for_title` if ``False`` then it is ``True``;.
        Default: ``None``
        Default: ``None``
    flag_real_p : bool, optional
        Unlike `flag_initial_p`, which refers to the p (sigma_x_compress) indicated in the constructor of the net, \
        this flag reads the p from the trained parameters of the net. \
        Its default state depends on whether `flag_for_title` is ``True`` or ``False`` and is inverse to that \
        of `flag_initial_lambda`:
        if `flag_for_title` if ``True`` then it is ``True``; \
        if `flag_for_title` if ``False`` then it is ``False``;.
        Default: ``None``

    Returns
    -------
    str
    """

    assert isinstance(flag_for_title, bool), f"The argument 'flag_for_title' must be bool: '{flag_for_title}' given!"

    #####
    # Set the default state of the indecisive parameters
    #####
    if flag_initial_lambda is None:
        flag_initial_lambda = not flag_for_title
    if flag_initial_p is None:
        flag_initial_p = not flag_for_title
    if flag_real_lambda is None:
        flag_real_lambda = flag_for_title
    if flag_real_p is None:
        flag_real_p = flag_for_title

    #####
    # Extract fields from the classifier: from its constructor_kwargs, mostly
    #####

    assert 'constructor_kwargs' in classifier_nn._extra_state_dict, \
        f"'constructor_kwargs' not present in classifier_nn._extra_state_dict!!!"
    constructor_kwargs = classifier_nn._extra_state_dict['constructor_kwargs']

    conv_like_type = constructor_kwargs.get('conv_like_type', None) if flag_conv_like_type else None
    #
    num_layers = None
    if flag_num_layers and 'm_kernel_size_per_conv_layer' in constructor_kwargs:
        num_layers = len(constructor_kwargs['m_kernel_size_per_conv_layer'])
    #
    num_neurons_per_layer = None
    if flag_num_neurons_per_layer and 'm_kernel_size_per_conv_layer' in constructor_kwargs:
        num_neurons_per_layer = constructor_kwargs['m_kernel_size_per_conv_layer'][0][0]
    #
    initial_lambda = None
    if flag_initial_lambda and 'initial_lambda' in constructor_kwargs:
        initial_lambda = constructor_kwargs['initial_lambda'][0]
    #
    initial_p = None
    if flag_initial_p and 'sigma_x_compress' in constructor_kwargs:
        initial_p = constructor_kwargs['sigma_x_compress']

    string_real_lambda = None
    if flag_real_lambda:
        string_real_lambda = _text_from_real_param_from_classifier(classifier_nn, 'lambda')

    string_real_p = None
    if flag_real_p:
        string_real_p = _text_from_real_param_from_classifier(classifier_nn, 'p')

    #####
    # Create the text description
    #####

    description_str = ""
    description_str += (f"({str(conv_like_type)})" if conv_like_type is not None else "")
    description_str += (f"x{num_layers}" if num_layers is not None else "")
    description_str += (f" ({num_neurons_per_layer}u)" if num_neurons_per_layer is not None else "")
    if flag_for_title:
        description_str += (f" $\\lambda_0$={str(initial_lambda)}" if initial_lambda is not None else "")
        description_str += (f" $\\lambda$={string_real_lambda}" if string_real_lambda is not None else "")
        description_str += (f" p_0={str(initial_p)}" if initial_p is not None else "")
        description_str += (f" p={string_real_p}" if string_real_p is not None else "")
    else:
        description_str += (f"_lambda0_{str(initial_lambda)}" if initial_lambda is not None else "")
        description_str += (f"_lambda_{string_real_lambda}" if string_real_lambda is not None else "")
        description_str += (f"_p0_{str(initial_p)}" if initial_p is not None else "")
        description_str += (f"_p_{string_real_lambda}" if string_real_lambda is not None else "")
    pass
    #
    return description_str



###################################################################################################
###################################################################################################
#
# VISUALIZATION FUNCTIONS
#
###################################################################################################
###################################################################################################



def _define_grid_as_batched_images(grid=None, x1_range=None, x2_range=None, points=None,
                                   delta_x=None, num_points_grid=None):
    """
    It defines the grid were the class assigned to each input 2D point will be represented. More precisely: \
    it defines said points... but in the form of batched images (each point a 1-ch image of 2 pixels set vertically).

    The portion of the 2D plane where those decision boundaries are visualized can be set in three different ways \
    whose priority is taken in the order stated next:

    - By providing a `grid`, which is a list/tuple of two 1D or 2D :py:class: `~numpy.ndarray` or torch.Tensor \
    containing the coordinates of the points where the decision boundaries will be evaluated (see the format \
    of the function from Matplotlib `contourf` for reference)

    - By providing `x1_range` and `x2_range`, which are 2-tuples/lists containing the minimum and maximum \
    coordinates along each axis, to be transformed into a `grid` through either `delta_x` or `num_points_grid`.

    - By providing `points`, which is a 2D :py:class:`~numpy.ndarray` or torch.Tensor containing the coordinates \
    of the points in the dataset used to train the model. The portion of the 2D plane to be visualized will be \
    automatically set, in the form of `x1_range` and `x2_range`, to a rectangle containing all the points \
    (plus a margin) and, again, such ranges will be transformed into a `grid` through either \
    `delta_x` or `num_points_grid`.

    Parameters
    ----------
    grid : list/tuple of two 1D or 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    x1_range, x2_range : list/tuple of two floats, optional
        Default: `None`
    points : 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    labels : 1D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    delta_x : float, optional
        Grid step along each axis, if `grid` is not provided.
        Default: None
    num_points_grid : int, optional
        (Approximate) Number of total points (points along each axis will be, approx., its squared root).
        Default: 1000

    Returns
    -------
    grid : torch.Tensor
    grid_as_batched_images : torch.Tensor
    """


    ####################################################################################################################
    # Check and reformat the grids
    ####################################################################################################################

    # Check, if provided, 'points' and 'labels'
    if points is not None:
        # Make them tensor
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32).squeeze()
        pass
        # Check types and shapes
        assert isinstance(points, torch.Tensor), \
            f"'points' must be a numpy.ndarray or torch.Tensor; got {type(points)}."
        assert points.ndim == 2 and points.shape[-2] == 2, \
            f"'points' must be a 2D array-like with shape (2, num_points): {points.shape}."
    pass

    if grid is not None:
        ####################################################
        # If grid is provided:
        ####################################################
        # If tuple -> list
        grid = list(grid) if isinstance(grid, tuple) else grid
        assert isinstance(grid, list) and len(grid) == 2, \
            (f"'grid' must be a list or tuple of two 1D or 2D array-like objects; " +
             f"got {type(grid)} with length {len(grid) if isinstance(grid, (list, tuple)) else 'N/A'}.")
        # Make numpy arrays be tensors and squeeze their unnecessary dimensions
        for i in range(len(grid)):
            if isinstance(grid[i], np.ndarray):
                grid[i] = torch.tensor(grid[i], dtype=torch.float32)
            pass
            if isinstance(grid[i], torch.Tensor):
                grid[i] = grid[i].squeeze()
            pass
        pass
        assert all(isinstance(grid[i], torch.Tensor) for i in range(len(grid))), \
            (f"Both elements of 'grid' must be either numpy.ndarray or torch.Tensor. " +
             f"Got {[type(grid[i]) for i in range(len(grid))]}.")
        # The input grid can be either the two 1D arrays of coordinates along each axis (of possibly different lengths
        # for x1 and x2), or two 2D arrays of the same shape, containing the coordinates of each point.
        # So, if 1D -> convert to 2D using meshgrid!
        if grid[0].ndim == 1:
            assert grid[1].ndim != 1, \
                (f"If the first element of 'grid' is 1D, the second one must be 2D. " +
                 f"Got shapes {grid[0].shape} and {grid[1].shape}.")
            # Use meshgrid to generate the 2D arrays
            x1_grid, x2_grid = torch.meshgrid(grid[0], grid[1], indexing='ij')
            grid = [x1_grid, x2_grid]
        elif grid[0].ndim == 2:
            assert grid[1].ndim == 2, \
                (f"If the first element of 'grid' is 2D, the second one must be 2D too. " +
                 f"Got shapes {grid[0].shape} and {grid[1].shape}.")
        else:
            raise ValueError((f"Both elements of 'grid' must be either 1D or 2D; " +
                              f"got shapes {grid[0].shape} and {grid[1].shape}."))
        pass
    else:
        ####################################################
        # If 'grid' is NOT provided we will need to use, or generate, 'x1_range' and 'x2_range':
        ####################################################
        if x1_range is not None:
            ####################################################
            # If 'grid' is NOT provided but 'x1_range' is provided:
            ####################################################
            assert x2_range is not None, "When 'x1_range' is provided 'x2_range' must be provided too."
            # Make both lists of 2D elements or give an error
            if isinstance(x1_range, tuple):
                x1_range = list(x1_range)
            if isinstance(x2_range, tuple):
                x2_range = list(x2_range)
            pass
            assert len(x1_range) == 2 and all(isinstance(x1_range[i], (int, float)) for i in range(2)), \
                f"'x1_range' must be a list or tuple of two numbers; got {x1_range}."
            assert len(x2_range) == 2 and all(isinstance(x2_range[i], (int, float)) for i in range(2)), \
                f"'x2_range' must be a list or tuple of two numbers; got {x2_range}."
        elif points is not None:
            x1_range = (points[0, :].min().item(), points[0, :].max().item())
            x2_range = (points[1, :].min().item(), points[1, :].max().item())
            # Add a margin around  of the range
            margin = 0.2
            extra_x1 = margin * (x1_range[1] - x1_range[0])
            extra_x2 = margin * (x2_range[1] - x2_range[0])
            x1_range = (x1_range[0] - extra_x1, x1_range[1] + extra_x1)
            x2_range = (x2_range[0] - extra_x2, x2_range[1] + extra_x2)
        else:
            raise ValueError("One of 'grid', 'x1_range' or 'points' must be provided!")
        pass
        # Now, generate the grid from the ranges
        assert x1_range[0] < x1_range[1], f"The range 'x1_range' is not ordered: {x1_range}."
        assert x2_range[0] < x2_range[1], f"The range 'x2_range' is not ordered: {x2_range}."
        # Check that either 'delta_x' or 'num_points_grid' is provided
        if delta_x is None and num_points_grid is None:
            raise Exception("When 'grid' is not provided, one of 'delta_x' or 'num_points_grid' must be provided.")
        elif delta_x is not None:
            assert isinstance(delta_x, (int, float)) and delta_x > 0.0, \
                f"'delta_x' must be a positive number; got {delta_x}."
            # Generate the points along each axis
            x1_values = torch.arange(x1_range[0], x1_range[1], delta_x, dtype=torch.float32)
            x2_values = torch.arange(x2_range[0], x2_range[1], delta_x, dtype=torch.float32)
        else:  # This means that 'num_points_grid' is not None
            assert isinstance(num_points_grid, int) and num_points_grid > 0, \
                f"'num_points_grid' must be a positive integer; got {num_points_grid}."
            # Calculate the number of points along each axis, approximately the squared root of 'num_points_grid'
            # but taking into account the complete range of each dimension (NOTE: the formula is inexact but reasonable)
            length1 = float(x1_range[1] - x1_range[0])
            length2 = float(x2_range[1] - x2_range[0])
            ratio_points = math.sqrt(length2 / length1)
            num_points_x1 = max(2, round(1 / ratio_points * math.sqrt(num_points_grid)))
            num_points_x2 = max(2, round(ratio_points * math.sqrt(num_points_grid)))
            # Generate the points along each axis
            x1_values = torch.linspace(x1_range[0], x1_range[1], num_points_x1, dtype=torch.float32)
            x2_values = torch.linspace(x2_range[0], x2_range[1], num_points_x2, dtype=torch.float32)
        pass
        # For the provided or just-calculated 'x1_values' and 'x2_values', generate the grid
        x1_grid, x2_grid = torch.meshgrid(x1_values, x2_values, indexing='ij')
        grid = [x1_grid, x2_grid]
    pass

    # Now that 'grid' is defined: (i) pack it into a tensor representing "a batch of images of 1-ch and size 2x2" and
    # (ii) evaluate the output of each "image" using the model
    flattened_grid = [elem.flatten() for elem in grid]
    grid_as_batched_images = torch.stack(flattened_grid, dim=1).unsqueeze(-1).unsqueeze(-3)

    #####
    # Try to free some memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####

    return grid, grid_as_batched_images


###################################################################################################
# Visualization of the boundaries of an experiment
###################################################################################################


def visualize_2D_classifier_linear_regions(model: (torch.nn.Module, list, tuple),
                                           grid=None, x1_range=None, x2_range=None, points=None, labels=None,
                                           delta_x=None, num_points_grid=None,
                                           ax=None, cmap=None, cbarlocation='bottom', title=None):
    """
    Visualize the regions of the input 2D space of a classifier that correspond, for a first SM hidden layer with \
    a non-linearity of the type ReLU, to different regimes of its output units. That is: for an affine map inside \
    the SM layer from $R^2$ to $R^n$, each input 2D point can result, after the ReLU, into any of the output neurons \
    being active (positive output) or inactive (zero output). Therefore, and consequently, this function generates \
    for each point the binary value of the possible $2^n$ corresponding to the number of active/inactive outputs.

    - If `model` is a single py:class:`~torch.nn.Module` the plot: (i) represents filled scores of the regions \
      using *contourf* from Matplotlib.

    - If `model` is a list/tuple of py:class:`~torch.nn.Module`, the plot represents, for each model, the limits \
      of the above regions using one single color, and with a different color per model.

    The portion of the 2D plane where those regions are visualized can be set in three different ways \
    whose priority is taken in the order stated next:

    - By providing a `grid`, which is a list/tuple of two 1D or 2D :py:class: `~numpy.ndarray` or torch.Tensor \
    containing the coordinates of the points where the decision boundaries will be evaluated (see the format \
    of the function from Matplotlib `contourf` for reference)

    - By providing `x1_range` and `x2_range`, which are 2-tuples/lists containing the minimum and maximum \
    coordinates along each axis, to be transformed into a `grid` through either `delta_x` or `num_points_grid`.

    - By providing `points`, which is a 2D :py:class:`~numpy.ndarray` or torch.Tensor containing the coordinates \
    of the points in the dataset used to train the model. The portion of the 2D plane to be visualized will be \
    automatically set, in the form of `x1_range` and `x2_range`, to a rectangle containing all the points \
    (plus a margin) and, again, such ranges will be transformed into a `grid` through either \
    `delta_x` or `num_points_grid`.

    Additionally, if `points` are provided, `labels` must be provided too, and the points are plotted too.

    Parameters
    ----------
    model : torch.nn.Module or list/tuple of torch.nn.Module
        The 2D classifier model to be visualized, with the respective behaviour as explained above
    grid : list/tuple of two 1D or 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    x1_range, x2_range : list/tuple of two floats, optional
        Default: `None`
    points : 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    labels : 1D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    delta_x : float, optional
        Grid step along each axis, if `grid` is not provided.
        Default: None
    num_points_grid : int, optional
        (Approximate) Number of total points (points along each axis will be, approx., its squared root).
        Default: 1000
    ax : matplotlib.axes.Axes, optional
        Axes where to plot the decision boundaries.
        Default: `None`
    cmap : str or ~matplotlib.colors.Colormap, optional
        Colormap to be used for the visualization. The default colormap is different depending on whether \
        a single model (and thus contourf is used) or multiple models (and thus only the boundary of each model is \
        shown) are provided:. in the former case, the default is a `'gray'` colormap; in the latter case,
        the default `'tab20b'` otherwise.
        Default: `'gray'` for a single model, `'tab20b'` for multiple models
    cbarlocation : str, optional
        Location of the colorbar. Options are: 'left', 'right', 'top', 'bottom'.
        Default: `'bottom'`
    title : str, optional
        Title of the plot. In case of `None`, the title is taken from the parameters of the model in the form of: \
        "type of layers + lambdas, if relevant".
        Default: `None`

    Returns
    -------
    matplotlib.axes.Axes
    """

    flag_multiple_models = False
    models = None
    #
    if isinstance(model, (list, tuple)):
        assert all(isinstance(model[i], torch.nn.Module) for i in range(len(model))), \
            (f"If 'model' is a list or tuple, all its elements must be torch.nn.Module; " +
             f"got {[type(model[i]) for i in range(len(model))]}.")
        pass
        assert all([model[i]._extra_state_dict['out_classes'] == 2 for i in range(len(model))]), \
            (f"If 'model' is a list or tuple, all its elements must be binary classifiers; " +
             f"got {[model[i]._extra_state_dict['out_classes'] for i in range(len(model))]}.")
        pass
        flag_multiple_models = True
        models = list(model)
        cmap = 'tab20b' if cmap is None else cmap
    elif isinstance(model, torch.nn.Module):
        flag_multiple_models = False
        models = [model]
        cmap = 'gray' if cmap is None else cmap
    else:
        raise ValueError(f"'model' must be either a torch.nn.Module or a list/tuple of them; got {type(model)}.")
    pass

    # Computation device: 'cuda' if possible
    computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################################################################################################################
    # Check and reformat the grids
    ####################################################################################################################

    # Check, if provided, 'points' and 'labels'
    if points is not None:
        # Labels must also be provided!
        assert labels is not None, \
            f"Since 'points' is provided, 'labels' must also be provided!"
        # Make them tensor
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32).squeeze()
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.int8).squeeze()
        pass
        # Check types and shapes
        assert isinstance(points, torch.Tensor), \
            f"'points' must be a numpy.ndarray or torch.Tensor; got {type(points)}."
        assert points.ndim == 2 and points.shape[-2] == 2, \
            f"'points' must be a 2D array-like with shape (2, num_points): {points.shape}."
        assert isinstance(labels, torch.Tensor), \
            f"'labels' must be a numpy.ndarray or torch.Tensor; got {type(labels)}."
        assert labels.ndim == 1 and labels.shape[0] == points.shape[-1], \
            (f"'labels' must be a 1D array-like with shape (num_points,), " +
             f"matching the number of points: {labels.shape} vs {points.shape}.")
    pass

    ### Define the grid

    grid, grid_as_batched_images = _define_grid_as_batched_images(
        grid=grid, x1_range=x1_range, x2_range=x2_range, points=points,
        delta_x=delta_x, num_points_grid=num_points_grid)

    #####
    # Try to free some memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####

    ### Evaluate the model or models
    #
    models_binary_code = [None] * len(models)
    grid_as_batched_images = grid_as_batched_images.to(computation_device)
    #
    for i, model_i in enumerate(models):
        print(f"Model {i + 1}-th (of {len(models)})...")
        try:
            args_affine_layer = model_i[i]._nn.backbone.block_conv_0.conv_0.get_extra_state()
            first_sm_layer = SMLayer(**args_affine_layer)
            first_sm_layer.set_m(model_i[i]._nn.backbone.block_conv_0.conv_0.theta_copy['m'])
            first_sm_layer.set_b(model_i[i]._nn.backbone.block_conv_0.conv_0.theta_copy['b'])
            #
            # NOTE: since each "image" is very small (2x1) there seem not to exist a reason to use batches
            model_i.to(computation_device)
            model_i.eval()
            with torch.no_grad():
                models_binary_code[i] = model_i(grid_as_batched_images).detach().cpu()
            pass
            #
            #####
            # Try to free some memory
            model_i.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            #####
            #
        except Exception as e:
            raise Exception(f"Error when evaluating the model on the grid points:\n{e}.")
        pass
    pass

    print(f"Reshaping model results...", flush=True)

    # Transform the 'models_outputs' into 'grid_values', in the manner that each case requires (classes or scores)

    grid_max_class = [None] * len(models)
    grid_prob_last_class = [None] * len(models)
    for i in range(len(models_outputs)):
        grid_prob_last_class[i] = F.softmax(models_outputs[i], dim=-1)[:, -1]
        grid_max_class[i] = torch.argmax(models_outputs[i], dim=-1)
        # Reshape the resulting values to the shape of the grid
        grid_max_class[i] = grid_max_class[i].reshape(grid[0].shape)
        grid_prob_last_class[i] = grid_prob_last_class[i].reshape(grid[0].shape)
        #
        #####
        # The 'models_outputs' are not needed any more: try to free some memory.
        # Additionally: everything could be pushed now to CPU, so we do so
        models_outputs[i] = None
        grid_max_class[i] = grid_max_class[i].detach().cpu()
        grid_prob_last_class[i] = grid_max_class[i].detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        #####
        #
    pass

    #
    #####
    # The 'models_outputs' are not needed any more: try to free some memory.
    # Additionally: everything could be pushed now to CPU, so we do so
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####
    #

    print(f"Plotting the decision boundaries...", flush=True)

    #########################################################################################
    # Plot the results, with points if necessary
    #########################################################################################

    vmin, vmax = None, None

    if ax is None:
        fig, ax = plt.subplots()
    pass

    if flag_multiple_models:
        #
        legend_elems = []
        for i in range(len(models)):
            color_i = cm.get_cmap(cmap)(i / len(models))
            contour = ax.contour(grid[0].numpy(), grid[1].numpy(), grid_max_class[i].numpy(),
                                 levels=[0.5], colors=color_i, zorder=5)
            # Add the info to the legend handles
            name_legend = _text_description_from_classifier(models[i], flag_for_title=True)
            if name_legend is None or not isinstance(name_legend, str):
                name_legend = f"Model {i + 1}"
            pass
            legend_elems.append(mlines.Line2D([], [], color=color_i, label=name_legend))
        pass
        #
        # Add a legend to the figure
        ax.legend(handles=legend_elems)
        #
        # Set the desired title
        if title is None:
            title = "Decision boundary for the last class"
        pass
        #
    else:  # Single model
        # Display both contours and points (the latter if so requested)
        vmin, vmax = 0, 1
        contourf = ax.contourf(grid[0].numpy(), grid[1].numpy(), grid_prob_last_class[0].numpy(),
                               vmin=vmin, vmax=vmax, cmap=cmap, levels=levels, zorder=2)
        # Add the colorbar of the classes
        cbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_mappable = cm.ScalarMappable(norm=cbar_norm, cmap=cmap)
        # plt.colorbar(cbar_mappable, location=cbarlocation, shrink=0.33, label="Class color code", ax=ax)
        plt.colorbar(cbar_mappable, location=cbarlocation, shrink=0.33, ax=ax)
        # And add the contour of the change of class
        contour = ax.contour(grid[0].numpy(), grid[1].numpy(), grid_max_class[0].numpy(),
                             levels=[0.5], colors='r', zorder=5)

        # Set the desired title
        if title is None:
            title = _text_description_from_classifier(models[0], flag_for_title=True)
        pass
        #
    pass
    #
    vmin_points, vmax_points = np.min(labels.numpy()), np.max(labels.numpy())
    if points is not None:
        print(f"Plotting {points.shape[-1]} points on top of the decision boundaries...")
        scatter = ax.scatter(points[0, :].numpy(), points[1, :].numpy(), c=labels.numpy(),
                             vmin=vmin_points, vmax=vmax_points, cmap="summer", s=10,
                             edgecolors='k', linewidths=0.75,
                             zorder=4)
    pass
    #
    ax.set_title(title)
    ax.axis('equal')
    #
    #####
    # Try to free some memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####
    #
    return ax



###################################################################################################
# Visualization of the boundaries of an experiment
###################################################################################################



def visualize_2D_classifier_decision_boundaries(model: (torch.nn.Module, list, tuple),
                                                grid=None, x1_range=None, x2_range=None,
                                                points=None, labels=None,
                                                display_point_mode=True,
                                                palette_points=None, size_points=10, alpha_points=0.5,
                                                delta_x=None, num_points_grid=10**6,
                                                xlims:tuple=None, ylims:tuple=None,
                                                class_mask_set=None,
                                                ax=None, cmap=None, levels=10, cbarlocation='bottom', title=None):
    """
    Visualize the decision boundaries of the single 2D classifier `model` or for the list/tuple of classifiers given
    in `model`. The behavior in both cases is different:
    
    - If `model` is a single py:class:`~torch.nn.Module` the plot: (i) represents the scores of the classifier \
      if the classifier the decision boundaries \
    of the model, using *contourf* from Matplotlib.
    
    - If `model` is a list/tuple of py:class:`~torch.nn.Module`, the plot represents the decision boundaries \
    of each model presuming that they are all binary classifier (0 versus 1) using only *contour* from Matplotlib,
    using different colors.

    The portion of the 2D plane where those decision boundaries are visualized can be set in three different ways \
    whose priority is taken in the order stated next:

    - By providing a `grid`, which is a list/tuple of two 1D or 2D :py:class: `~numpy.ndarray` or torch.Tensor \
    containing the coordinates of the points where the decision boundaries will be evaluated (see the format \
    of the function from Matplotlib `contourf` for reference)

    - By providing `x1_range` and `x2_range`, which are 2-tuples/lists containing the minimum and maximum \
    coordinates along each axis, to be transformed into a `grid` through either `delta_x` or `num_points_grid`.

    - By providing `points`, which is a 2D :py:class:`~numpy.ndarray` or torch.Tensor containing the coordinates \
    of the points in the dataset used to train the model. The portion of the 2D plane to be visualized will be \
    automatically set, in the form of `x1_range` and `x2_range`, to a rectangle containing all the points \
    (plus a margin) and, again, such ranges will be transformed into a `grid` through either \
    `delta_x` or `num_points_grid`.

    Additionally, if `points` are provided, `labels` must be provided too, and the points are plotted too.

    Parameters
    ----------
    model : torch.nn.Module or list/tuple of torch.nn.Module
        The 2D classifier model to be visualized, with the respective behaviour as explained above
    grid : list/tuple of two 1D or 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    x1_range, x2_range : list/tuple of two floats, optional
        Default: `None`
    points : 2D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    labels : 1D numpy.ndarray or torch.Tensor, optional
        Default: `None`
    display_point_mode : bool or int or bool, optional
        If ``False``, ``0``, or ``0.0``, (points are loaded to determine ranges but) no points are shown. \
        If ``True`` or ``1.0``, all points are shown. \
        If `display_point_mode` is a float in the range (0.0, 1.0) it will be understood as a percentage of points \
        to display, while an int > 0 would be understood as the maximum points to plot: in both cases points are \
        selected randomly.
        If no points are provided, \
        even if the flag is set to a value where points would be shown no points would be displayed.
        Default: ``True``
    palette_points : str, optional
        Default: ``None``, assigned depending on the number of models for display
    size_points : int, optional
        Default: ``10``
    alpha_points : float, optional
        A float in the range [0.0, 1.0].
        Default: ``0.5``
    delta_x : float, optional
        Grid step along each axis, if `grid` is not provided.
        Default: None
    num_points_grid : int, optional
        (Approximate) Number of total points (points along each axis will be, approx., its squared root).
        Default: 1000000
    xlims : tuple, optional
        If not ``None``, a 2-element tuple of floats indicating the inf-sup limits to display.
        Default: ``None`` (automatically defined by the data to plot)
    ylims : tuple, optional
        If not ``None``, a 2-element tuple of floats indicating the inf-sup limits to display.
        Default: ``None`` (automatically defined by the data to plot)
    class_mask_set : tuple[torch.Tensor], optional
        Triplet having the mask, and the grid_x and grid_y corresponding to the meshgrid for the mask. \
        If provided, no background grid will be displayed (beyond the decision regions).
        Default: ``None``
    ax : matplotlib.axes.Axes, optional
        Axes where to plot the decision boundaries.
        Default: `None`
    cmap : str or ~matplotlib.colors.Colormap, optional
        Colormap to be used for the visualization. The default colormap is different depending on whether \
        a single model (and thus contourf is used) or multiple models (and thus only the boundary of each model is \
        shown) are provided:. in the former case, the default is a `'gray'` colormap; in the latter case,
        the default `'tab20b'` otherwise.
        Default: `'gray'` for a single model, `'tab20b'` for multiple models
    levels : int, optional
        Number of levels to be used in the contourf plot. Only relevant when a single model is provided.
        Default: `10`
    cbarlocation : str, optional
        Location of the colorbar. Options are: 'left', 'right', 'top', 'bottom'.
        Default: `'bottom'`
    title : str, optional
        Title of the plot. In case of `None`, the title is taken from the parameters of the model in the form of: \
        "type of layers + lambdas, if relevant".
        Default: `None`

    Returns
    -------
    matplotlib.axes.Axes
    """

    if xlims is not None:
        assert isinstance(xlims, tuple) and len(xlims)==2, \
            f"If 'xlims' is not None, it must be a tuple of two floats; got {xlims}."
        assert all([isinstance(elem, (float,int)) for elem in xlims]), \
            f"If 'xlims' is not None, all its elements must be floats; got {xlims}."
    pass
    if ylims is not None:
        assert isinstance(ylims, tuple) and len(ylims) == 2, \
            f"If 'ylims' is not None, it must be a tuple of two floats; got {ylims}."
        assert all([isinstance(elem, (float, int)) for elem in ylims]), \
            f"If 'ylims' is not None, all its elements must be floats; got {ylims}."
    pass


    models = None
    # Flag: whether models (1 or more) where indicated as a list or one single model was indicated in isolation
    flag_isolated_model = False
    #
    if isinstance(model, (list, tuple)):
        assert all(isinstance(model[i], torch.nn.Module) for i in range(len(model))), \
            (f"If 'model' is a list or tuple, all its elements must be torch.nn.Module; " +
             f"got {[type(model[i]) for i in range(len(model))]}.")
        pass
        assert all([model[i]._extra_state_dict['out_classes'] == 2 for i in range(len(model))]), \
            (f"If 'model' is a list or tuple, all its elements must be binary classifiers; " +
             f"got {[model[i]._extra_state_dict['out_classes'] for i in range(len(model))]}.")
        pass
        flag_isolated_model = False
        models = list(model)
        cmap = 'tab20b' if cmap is None else cmap
        palette_points = 'gray' if palette_points is None else palette_points
    elif isinstance(model, torch.nn.Module):
        flag_isolated_model = True
        models = [model]
        cmap = 'gray' if cmap is None else cmap
        palette_points = 'Wistia' if palette_points is None else palette_points
    else:
        raise ValueError(f"'model' must be either a torch.nn.Module or a list/tuple of them; got {type(model)}.")
    pass

    # Computation device: 'cuda' if possible
    computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################################################################################################################
    # Check the mask set
    ####################################################################################################################

    if class_mask_set is not None:
        assert isinstance(class_mask_set, tuple), f"'class_mask_set' must be a tuple; got {type(class_mask_set)}."
        assert len(class_mask_set) == 3, f"'class_mask_set' must have exactly 3 elements, got {len(class_mask_set)}."
        assert all([isinstance(elem, torch.Tensor) for elem in class_mask_set]), \
            f"All the elements in 'class_mask_set' must be torch.Tensor; got {[type(elem) for elem in class_mask_set]}."
    pass

    ####################################################################################################################
    # Check and reformat the grids
    ####################################################################################################################

    # Check, if provided, 'points' and 'labels'
    number_of_points_to_display = 0
    if points is not None:
        # Display the points?
        number_of_points_to_display = points.shape[-1]
        if isinstance(display_point_mode, bool):
            number_of_points_to_display = number_of_points_to_display if display_point_mode else 0
        elif isinstance(display_point_mode, int):
            assert display_point_mode >= 0, \
                f"If 'display_point_mode' is an int, it must be non-negative; got {display_point_mode}."
            number_of_points_to_display = min(display_point_mode, number_of_points_to_display)
        elif isinstance(display_point_mode, float):
            assert 0.0 <= display_point_mode <= 1.0, \
                f"If 'display_point_mode' is a float, it must be in the range [0.0, 1.0]; got {display_point_mode}."
            number_of_points_to_display = int(display_point_mode * number_of_points_to_display)
        else:
             raise ValueError(f"'display_point_mode' must be a bool, int, or float; got {type(display_point_mode)}.")
        pass

        # Labels must also be provided!
        assert labels is not None, \
            f"Since 'points' is provided, 'labels' must also be provided!"
        # Make them tensor
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32).squeeze()
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.int8).squeeze()
        pass
        # Check types and shapes
        assert isinstance(points, torch.Tensor), \
            f"'points' must be a numpy.ndarray or torch.Tensor; got {type(points)}."
        assert points.ndim == 2 and points.shape[-2] == 2, \
            f"'points' must be a 2D array-like with shape (2, num_points): {points.shape}."
        assert isinstance(labels, torch.Tensor), \
            f"'labels' must be a numpy.ndarray or torch.Tensor; got {type(labels)}."
        assert labels.ndim == 1 and labels.shape[0] == points.shape[-1], \
            (f"'labels' must be a 1D array-like with shape (num_points,), " +
             f"matching the number of points: {labels.shape} vs {points.shape}.")
    else:
        display_point_mode = False
        number_of_points_to_display = 0
    pass

    ### Define the grid

    grid, grid_as_batched_images = _define_grid_as_batched_images(
        grid=grid, x1_range=x1_range, x2_range=x2_range, points=points,
        delta_x=delta_x, num_points_grid=num_points_grid)

    #####
    # Try to free some memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####

    ### Evaluate the model or models
    #
    models_outputs = [None]*len(models)
    grid_as_batched_images = grid_as_batched_images.to(computation_device)
    #

    for i, model_i in enumerate(models):
        print(f"Model {i+1}-th (of {len(models)})...")
        try:
            # NOTE: since each "image" is very small (2x1) there seem not to exist a reason to use batches
            model_i.to(computation_device)
            model_i.eval()
            with torch.no_grad():
                models_outputs[i] = model_i(grid_as_batched_images).detach().cpu()
            pass
            #
            #####
            # Try to free some memory
            model_i.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            #####
            #
        except Exception as e:
            raise Exception(f"Error when evaluating the model on the grid points:\n{e}.")
        pass
    pass
    
    # Transform the 'models_outputs' into 'grid_values', in the manner that each case requires (classes or scores)

    print(f"Reshaping model results...", flush=True)

    # ############################
    # # TO DELETE!
    # ############################
    # # FIND INDICES CORRESPONDING TO THE POINTS CLOSE TO THE (0,0)
    # p00 = torch.Tensor([0.0, 0.0]).unsqueeze(0).unsqueeze(-1).to(computation_device)
    # distance_to_p00 = torch.norm(grid_as_batched_images-p00, p=2, dim=(1, 2, 3))
    # indices_almost_p00 = (distance_to_p00 < 0.05).nonzero().flatten()
    # ############################
    # # TO DELETE!
    # ############################


    grid_max_class = [None] * len(models)
    grid_prob_last_class = [None] * len(models)
    for i in range(len(models_outputs)):
        grid_prob_last_class[i] = F.softmax(models_outputs[i], dim=-1)[:, -1]
        grid_max_class[i] = torch.argmax(models_outputs[i], dim=-1)

        # ############################
        # # TO DELETE!
        # ############################
        # print(f"Size 'grid_as_batched_images': {grid_as_batched_images.shape}")
        # print(f"Size 'models_outputs[i]': {models_outputs[i].shape}")
        # print(f"Size 'grid_prob_last_class[i]': {grid_prob_last_class[i].shape}")
        # print(f"Size 'grid_max_class[i]': {grid_max_class[i].shape}")
        # # PRINT THE MODEL OUTPUT, THE PROBABILITY (of the last class) AND THE CLASS FOR THE POINTS CLOSE TO (0,0)
        # for idx in indices_almost_p00:
        #     print(f"\t\tp00: {grid_as_batched_images[idx].cpu().numpy()}" +
        #           f", scores: {models_outputs[i][idx].cpu().numpy()}" +
        #           f", prob. last class: {grid_prob_last_class[i][idx].cpu().numpy()}" +
        #           f", class: {grid_max_class[i][idx].cpu().numpy()}")
        # ############################
        # # TO DELETE!
        # ############################

        # Reshape the resulting values to the shape of the grid
        grid_prob_last_class[i] = grid_prob_last_class[i].reshape(grid[0].shape)
        grid_max_class[i] = grid_max_class[i].reshape(grid[0].shape)
        #
        #####
        # The 'models_outputs' are not needed any more: try to free some memory.
        # Additionally: everything could be pushed now to CPU, so we do so
        models_outputs[i] = None
        grid_max_class[i] = grid_max_class[i].detach().cpu()
        grid_prob_last_class[i] = grid_prob_last_class[i].detach().cpu()
        #
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        #####
        #
    pass

    #
    #####
    # The 'models_outputs' are not needed any more: try to free some memory.
    # Additionally: everything could be pushed now to CPU, so we do so
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####
    #

    #########################################################################################
    # Plot the results, with points if necessary
    #########################################################################################

    print(f"Plotting the decision boundaries...", flush=True)

    # General params
    lw_boundaries = 2.5

    vmin, vmax = None, None

    if ax is None:
        fig, ax = plt.subplots()
    pass

    if class_mask_set is not None:
        print(f"Plotting class masks...", flush=True)
        class_mask, mask_x1_grid, mask_x2_grid = class_mask_set
        min_mask = torch.min(class_mask.flatten()).item()
        max_mask = torch.max(class_mask.flatten()).item()
        print(f"min_mask: {min_mask}, max_mask: {max_mask}")
        mask = ax.contourf(mask_x1_grid.numpy(), mask_x2_grid.numpy(), class_mask.numpy(),
                           vmin=min_mask, vmax=1.5*max_mask, cmap='Greys', zorder=0)
    pass

    if not flag_isolated_model:
        #
        legend_elems = []
        for i in range(len(models)):
            color_i = cm.get_cmap(cmap)(i / len(models))
            contour = ax.contour(grid[0].numpy(), grid[1].numpy(), grid_max_class[i].numpy(),
                                 levels=[0.5], colors=color_i, linewidths=lw_boundaries, zorder=5)
            # Add the info to the legend handles
            name_legend = _text_description_from_classifier(models[i], flag_for_title=True)

            if name_legend is None or not isinstance(name_legend, str):
                name_legend = f"Model {i+1}"
            pass
            legend_elems.append(mlines.Line2D([], [], color=color_i, label=name_legend))
        pass
        #
        # Add a legend to the figure
        ax.legend(handles=legend_elems)
        #
        # Set the desired title
        if title is None:
            title = "Decision boundary for the last class"
        pass
        #
    else: # Single model (flag_isolated_model == True)
        # Display both contours and points (the latter if so requested)
        vmin, vmax = 0, 1
        #
        # beta = 5
        # def _forward(x):
        #     y = np.where(
        #         x >= 1 / float(2),
        #         0.5 * (1 + np.pow(+(2 * x - 1), beta)),
        #         0.5 * (1 - np.pow(-(2 * x - 1), beta)),
        #     )
        #     return y
        # pass
        # def _inverse(y):
        #     x = np.where(
        #         y >= 1 / float(2),
        #         0.5 * (1 + np.pow(+(2 * y - 1), 1 / float(beta))),
        #         0.5 * (1 - np.pow(-(2 * y - 1), 1 / float(beta))),
        #     )
        #     return x
        # pass
        # norm = mcolors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
        #
        if class_mask_set is None:
            norm = mcolors.NoNorm()
            contourf = ax.contourf(grid[0].numpy(), grid[1].numpy(), grid_prob_last_class[0].numpy(),
                                  vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, levels=levels, zorder=2)
            # ax.contour(contourf, colors='k', linewidths=0.1)
            # Add the colorbar of the classes
            cbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cbar_mappable = cm.ScalarMappable(norm=cbar_norm, cmap=cmap)
            # plt.colorbar(cbar_mappable, location=cbarlocation, shrink=0.33, label="Class color code", ax=ax)
            plt.colorbar(cbar_mappable, location=cbarlocation, shrink=0.33, pad=0.05, ax=ax)
        pass
        # And add the contour of the change of class
        color_0 = 'r'
        contour = ax.contour(grid[0].numpy(), grid[1].numpy(), grid_max_class[0].numpy(),
                             levels=[0.5], colors=color_0, linewidths=lw_boundaries, zorder=5)
        #
        # Add the info to the legend handles
        name_legend = _text_description_from_classifier(models[0], flag_for_title=True)
        if name_legend is None or not isinstance(name_legend, str):
            name_legend = f"Model"
        pass
        #
        legend_elems = []
        legend_elems.append(mlines.Line2D([], [], color=color_0, label=name_legend))
        # Add a legend to the figure
        ax.legend(handles=legend_elems)
        #
        # Set the desired title
        if title is None:
            title = _text_description_from_classifier(models[0], flag_for_title=True)
        pass
        #
    pass
    #
    if points is not None and number_of_points_to_display > 0:

        # Randomly sample the points `points` and the labels `labels` to be displayed, if necessary
        if number_of_points_to_display < points.shape[-1]:
            indices_to_display = torch.randperm(points.shape[-1])[:number_of_points_to_display]
            final_points = points[:, indices_to_display]
            final_labels = labels[indices_to_display]
        else:
            final_points = points
            final_labels = labels
        pass
        vmin_points, vmax_points = np.min(final_labels.numpy()), np.max(final_labels.numpy())
        print(f"Plotting {final_points.shape[-1]} points (of a total {points.shape[-1]}) " +
              f"on top of the decision boundaries...")
        scatter = ax.scatter(final_points[0, :].numpy(), final_points[1, :].numpy(), c=final_labels.numpy(),
                             vmin=vmin_points, vmax=vmax_points, cmap=palette_points, s=size_points,
                             edgecolors='face', alpha=alpha_points, linewidths=0.75,
                             zorder=4)
    pass

    #####
    ax.set_title(title)
    #####

    print(f"xlims = {xlims}")
    print(f"ylims = {ylims}")

    #####
    ax.axis('equal')
    #
    if xlims is not None:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim((-1.0, +1.0))
    pass
    #
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim((-1.0, +1.0))
    pass
    #####

    #####
    # Try to free some memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #####
    #
    return ax


###################################################################################################
# Visualization of the boundaries of an experiment
###################################################################################################


def visualize_2D_classifier_decision_boundaries_from_experiment_run_label_tuple(
        tuple_experiment_run_labels: tuple|list,
        display_point_mode=True, palette_points=None, size_points=10, alpha_points=0.5,
        flag_display_mask=False,
        num_points_grid=10**6, xlims:tuple=None, ylims:tuple=None,
        ax=None, cmap=None, levels=10, cbarlocation='bottom',
        title=None, save_figure=False, base_path_figure=None, name_figure=None):
    """
    Identical to the function :py:func:`visualize_2D_classifier_decision_boundaries` but providing the model, or each \
    model, using a tuple of (experiment name/id, run name/id).

    The points for the display will be loaded from the dataset of the model, or from the dataset of the first model \
    if several, and displayed.

    Parameters
    ----------
    tuple_experiment_run_labels : tuple or list[tuple] or tuple[tuple]
        2D-tuple, or list/tuple of 2D tuples, containing the experiment name/id and run name/id of the model(s) to \
        load. torch.nn.Module or list/tuple of torch.nn.Module
        The 2D classifier model to be visualized, with the respective behaviour as explained above.
    display_point_mode : bool or int or bool, optional
        If ``False``, ``0``, or ``0.0``, (points are loaded to determine ranges but) no points are shown. \
        If ``True`` or ``1.0``, all points are shown. \
        If `display_point_mode` is a float in the range (0.0, 1.0) it will be understood as a percentage of points \
        to display, while an int > 0 would be understood as the maximum points to plot: in both cases points are \
        selected randomly.
        If no points are provided, \
        even if the flag is set to a value where points would be shown no points would be displayed.
        Default: ``True``
    palette_points : str, optional
        Default: ``None``, assigned depending on the number of models for display
    size_points : int, optional
        Default: ``10``
    alpha_points : float, optional
        A float in the range [0.0, 1.0].
        Default: ``0.5``
    num_points_grid : int, optional
        (Approximate) Number of total points (points along each axis will be, approx., its squared root).
        Default: 1000000
    xlims : tuple or list, optional
        If not ``None``, a 2-element tuple of floats indicating the inf-sup limits to display.
        Default: ``None`` (automatically defined by the data to plot)
    ylims : tuple or list, optional
        If not ``None``, a 2-element tuple of floats indicating the inf-sup limits to display.
        Default: ``None`` (automatically defined by the data to plot)
    ax : matplotlib.axes.Axes, optional
        Axes where to plot the decision boundaries.
        Default: `None`
    cmap : str or ~matplotlib.colors.Colormap, optional
        Colormap to be used for the visualization. The default colormap is different depending on whether \
        a single model (and thus contourf is used) or multiple models (and thus only the boundary of each model is \
        shown) are provided:. in the former case, the default is a `'gray'` colormap; in the latter case,
        the default `'tab20b'` otherwise.
        Default: `'gray'` for a single model, `'tab20b'` for multiple models
    levels : int, optional
        Number of levels to be used in the contourf plot. Only relevant when a single model is provided.
        Default: `10`
    cbarlocation : str, optional
        Location of the colorbar. Options are: 'left', 'right', 'top', 'bottom'.
        Default: `'bottom'`
    title : str, optional
        Title of the plot. In case of `None`, the title is taken from the name of the dataseet as... \
        "Multiple classifiers for dataset <dataset name>".
        Default: `None`
    save_figure : bool, optional
        Default: `False`
    base_path_figure: str, optional
        If `None`, the base path is formed as: "../figures/datasets_2d/<dataset name>".
        Default: `None`
    name_figure: str, optional
        If `None`, the name of the figure is formed as: \
        "multiple_classifiers_for_dataset_<dataset name> plus a 5 digit random number.
        Default: `None`

    Returns
    -------
    matplotlib.axes.Axes
    """

    ################################################################
    # Check if `tuple_experiment_run_labels` refers to 1 model or to several
    ################################################################

    # Flag: whether models (1 or more) where indicated as a list or one single model was indicated in isolation
    flag_isolated_model = False

    if not isinstance(tuple_experiment_run_labels, (tuple, list)):
        raise Exception(
            f"'tuple_experiment_run_labels' must be a tuple or list, {type(tuple_experiment_run_labels)} provided!"
        )
    else:
        #
        if len(tuple_experiment_run_labels) == 2 and all([isinstance(e, (str,int)) for e in tuple_experiment_run_labels]):
            # IN THIS CASE THE REQUEST WAS FOR ONE SINGLE MODEL INDICATED "WITHOUT THE WRAP OF A LIST":
            # then get it within a list (where it will be the unique model) BUT INDICATE IT WAS ISOLATED!
            tuple_experiment_run_labels = [tuple(tuple_experiment_run_labels)]
            flag_isolated_model = True
        #
        tuple_experiment_run_labels = list(tuple_experiment_run_labels)
        for elem in tuple_experiment_run_labels:
            assert len(elem) == 2 and all([isinstance(e, (str, int)) for e in elem]), \
                f"'tuple_experiment_run_labels' must be formed of 2D tuples of str/int: element {elem} found!"
        pass
    pass

    ################################################################
    # Get the models from the experiment+run labels, and get their metrics too!
    ################################################################

    artifact_uris = [None]*len(tuple_experiment_run_labels)
    recreated_models = [None] * len(tuple_experiment_run_labels)
    recreated_model_metrics = [None] * len(tuple_experiment_run_labels)
    for i, (experiment_label_i, run_label_i) in enumerate(tuple_experiment_run_labels):
        print(f"----------------------------------------------------------------------")
        print(f"| Model {i+1:3d} (out of {len(tuple_experiment_run_labels):3d})")
        print(f"| Experiment label: {experiment_label_i}")
        print(f"| Run label:        {run_label_i}")
        print(f"----------------------------------------------------------------------")
        artifact_uris[i] = artifact_uri_from_experiment_and_run(experiment_label_i, run_label_i)
        recreated_models[i] = recreate_classifier_from_run_artifacts_uri(artifact_uris[i])
        recreated_model_metrics[i] = experiment_metrics_for_run(experiment_label_i, run_label_i)
    pass

    ##############################################################################################################
    # Load the points of the dataset
    # The dataset will be loaded only once, for the first element of `tuple_experiment_run_labels`
    ##############################################################################################################

    recreated_dataset_loader_dict = load_same_dataset_of_run_artifacts_uri(artifact_uris[0])

    print(f"----------------------------------------------------------------------")
    print(f"| Loading validation points...")
    print(f"----------------------------------------------------------------------")

    point_images = None
    labels = None
    for test_batch, (images_batch, labels_batch) in enumerate( recreated_dataset_loader_dict['dataloader_test']):
        point_images = torch.cat((point_images, images_batch.to('cpu')), 0) if point_images is not None \
            else images_batch.to('cpu')
        labels = torch.cat((labels, labels_batch.to('cpu')), 0) if labels is not None \
            else labels_batch.to('cpu')
    pass
    points = point_images.squeeze().transpose(0, 1)

    ##############################################################################################################
    # Check, if the dataset is one such that there exists a mask, load the mask
    ##############################################################################################################

    class_mask_set = None
    if flag_display_mask:
        image_path = _find_image_custom_dataset(recreated_dataset_loader_dict['dataset_name'])
        _, class_mask = _multiclass_mask_from_image(image_path)
        mask_height, mask_width = class_mask.size(-2), class_mask.size(-1)
        mask_x1_vector = torch.Tensor(range(0, mask_width)).to(dtype=torch.float32)
        mask_x2_vector = torch.Tensor(range(0, mask_height)).to(dtype=torch.float32)
        mask_x1_grid, mask_x2_grid = torch.meshgrid(mask_x1_vector, mask_x2_vector, indexing='xy')
        if True: # Because it is normalized
            max_value_normalization = 1.0
            offset_pixel_dimensions = torch.tensor([mask_width / 2.0, mask_height / 2.0], dtype=torch.float32)
            scale_factor = max_value_normalization * 2.0 / (max(mask_height, mask_width) - 1)
            mask_x1_grid = (mask_x1_grid - offset_pixel_dimensions[0]) * scale_factor
            mask_x2_grid = (mask_x2_grid - offset_pixel_dimensions[1]) * scale_factor
        pass
        class_mask_set = (class_mask, mask_x1_grid, mask_x2_grid)
    pass

    ##############################################################################################################
    # Use the function to display the already loaded modules
    ##############################################################################################################

    if title is None:
        title = f"Multiple classifiers for dataset '{recreated_dataset_loader_dict['dataset_name']}'"
    pass

    try:
        #
        recreated_models_cpu = [recreated_classifier.to('cpu') for recreated_classifier in recreated_models]
        #
        ax = visualize_2D_classifier_decision_boundaries(
            model= recreated_models_cpu[0] if flag_isolated_model else recreated_models_cpu,
            points=points, labels=labels,
            display_point_mode=display_point_mode,
            palette_points=palette_points, size_points=size_points, alpha_points=alpha_points,
            num_points_grid=num_points_grid, xlims=xlims, ylims=ylims,
            class_mask_set=class_mask_set,
            ax=ax, levels=levels, title=title, cmap=cmap
        )
        #
        # Modify the existing legend to include the best accuracy
        for (metrics_model_i, legend_elem_i) in zip(recreated_model_metrics, ax.get_legend().get_texts()):
            acc_value = metrics_model_i.get('best_acc', None)
            if acc_value is not None:
                text_acc = f"val acc: {100 * acc_value:5.2f} %"
                # print(text_acc)
                legend_elem_i.set_text(legend_elem_i.get_text() + "  ||  " + text_acc)
            else:
                print(f"Best acc not accessible!")
            pass
        pass
        #
        if save_figure:
            if base_path_figure is None:
                base_path_figure = f"../figures/datasets_2d/{recreated_dataset_loader_dict['dataset_name']}"
            pass
            if not os.path.exists(base_path_figure):
                os.makedirs(base_path_figure)
            pass
            if name_figure is None:
                name_figure = f"multiple_classifiers_for_dataset_{recreated_dataset_loader_dict['dataset_name']}" + \
                              f"_{randint(0, 99999):05d}"
            pass
            plt.tight_layout(pad=0.25)
            for ext in ["eps", "svg", "png"]:
                plt.savefig(os.path.join(base_path_figure, name_figure) + f".{ext}", dpi=300, pad_inches=0.0)
            pass
        pass
    except Exception as e:
        print(f"EXCEPTION!!!\n{e}")
    pass
    #
    return ax


###################################################################################################
# Train a 2D classifier according to the defined parameters
###################################################################################################

def train_2D_classifier(net: str = 'multi_layer',
                        kwargs_classifier: dict = None, kwargs_dataset: dict = None,
                        kwargs_optim_scheduler: dict = None,
                        acc_threshold:float=None, number_of_retries:int=10,
                        mlflow_logging: str = "vmg", suffix_experiment:str=None, verbose='medium'):
    """

    Parameters
    ----------
    net : str
        Name of the classifier to be used. Options are: 'multi_layer', 'alexnet'...
    kwargs_classifier : dict
        Set of keyword arguments to be passed to the constructor of the classifier
    kwargs_dataset : dict
        Set of keyword arguments to be passed to \
        :py:func:`experimental_evaluation.operations_for_datasets import obtain_classification_dataset_loaders_from_point_data` \
        for creating the dataset and the loaders
    kwargs_optim_scheduler : dict
        Set of keyword arguments ruling the used optimized and scheduler, as understood by the function \
        :py:func:`experimental_evaluation.experiment_utils.classifier_training`, that is: \
        'maximum_epochs' (int), 'loss_function' (~collections.abc.Callable), \
        'optimizer_class' (torch.optim.Optimizer), 'optimizer_arg_dict' (dict), \
        'scheduler_class' (torch.optim.lr_scheduler.LRScheduler), 'scheduler_arg_dict' (dict), \
        'epochs_sm_based_warmup' (int), \
        'early_stop_epochs' (int), 'validations_per_epoch' (int), \
        'validation_on_test_subset' (bool), 'pre_validation' (bool), 'random_initialization' (bool)
    acc_threshold : float, optional
        Float ``0.0`` < `acc_threshold` < ``1.0`` marking the minimum accuracy achieved to accept the training as \
        'satisfactory', and under which the training is deleted and repeated.
        Default: ``None``
    number_of_retries : int, optional
        Number of times that the experiment will be repeated if `acc_threshold` is given and the acc. stays below.
        Default: ``10``
    mlflow_logging : str, optional
        URL of the MLflow tracking server to be used for logging.
        Default: ``"vmg"``, which loads the system credentials and addresses
    verbose : str, optional
        Level of verbosity of the training process. Options are: None, 'low', 'medium', 'high'.
        Default: ``'medium'``

    Returns
    -------
    nn.Module
    """

    # Computation device: 'cuda' if possible
    computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # (Very) Initial checks
    assert isinstance(net, str) and net in _dict_classifiers_as_in_conf_file, \
        (f"'net' must be a string with the name of the classifier to be used. " +
         f"Options are: {list(_dict_classifiers_as_in_conf_file.keys())}; got {net}.")
    assert isinstance(kwargs_classifier, dict), f"'kwargs_classifier' must be a dict; got {type(kwargs_classifier)}."
    assert isinstance(kwargs_dataset, dict), f"'kwargs_dataset' must be a dict; got {type(kwargs_dataset)}."
    assert isinstance(kwargs_optim_scheduler, dict), \
        f"'kwargs_optim_scheduler' must be a dict; got {type(kwargs_optim_scheduler)}."
    if acc_threshold is not None:
        assert isinstance(acc_threshold, (float)) and 0.0 < acc_threshold < 1.0, \
            f"The provided variable 'acc_threshold' must be a float between 0 and 1; got {acc_threshold}."
        assert isinstance(number_of_retries, int) and number_of_retries > 1, \
            f"Number of retries must be an integer greater than or equal to 1, got {number_of_retries}."
    assert mlflow_logging is None or isinstance(mlflow_logging, str), \
        f"'mlflow_logging' must be None or a string with the URL of the MLflow tracking server; got {mlflow_logging}."
    assert suffix_experiment is None or isinstance(suffix_experiment, str), \
        f"'suffix_experiment' must be None or string, got {suffix_experiment} of type {type(suffix_experiment)}."
    assert verbose in [None, 'low', 'medium', 'high'], \
        f"'verbose' must be None, 'low', 'medium' or 'high'; got {verbose}."

    # Load the dataset
    dataset_loader_dict = obtain_classification_dataset_loaders_from_point_data(**kwargs_dataset)
    print(
        f"Dataset {dataset_loader_dict['dataset_name']} {dataset_loader_dict['colorspace']} " +
        f"(B={dataset_loader_dict['batch_size']}, " +
        f"H={dataset_loader_dict['im_height']},W={dataset_loader_dict['im_width']}) created successfully!")

    # Make sure that the fields 'in_size' (tuple), 'in_channels' (int) and 'out_classes' (int), admissible/compulsory \
    # for the classifier, are in 'kwargs_classifier' and appear only once.
    dict_fields_to_push_once_in_kwargs_classifier = {
        'in_size': (dataset_loader_dict['im_height'], dataset_loader_dict['im_width']),
        'in_channels': dataset_loader_dict['channels'],
        'out_classes': dataset_loader_dict['classes']
    }
    for key in dict_fields_to_push_once_in_kwargs_classifier:
        if key in kwargs_classifier:
            print(f"Warning: field '{key}' already present in 'kwargs_classifier' with value " +
                  f"{kwargs_classifier[key]}; it will be overwritten with " +
                  f"{dict_fields_to_push_once_in_kwargs_classifier[key]}.")
        pass
        kwargs_classifier[key] = dict_fields_to_push_once_in_kwargs_classifier[key]
    pass


    number_of_retries = 10
    for ind_attempt_training in range(number_of_retries):

        if ind_attempt_training > 0:
            print(f"\n·············································")
            print((f"Re-attempt to training: {ind_attempt_training+1}-th attempt " +
                   f"(of max {number_of_retries})!"))
            print(f"·············································\n")
        pass

        # Create the model
        classifier_nn = _dict_classifiers_as_in_conf_file[net](**kwargs_classifier)
        classifier_nn.logging_compliance_checker()
        print(f"{_dict_classifiers_as_in_conf_file[net]} created successfully!")

        ###############################################################################################
        # Do the training!!!
        ###############################################################################################

        # Set other parameters that the function 'experimental_evaluation.experiment_utils import classifier_training' \
        # accepts and are not input arguments to the present function

        classifier_training_arg_mlflow_run_id = None
        classifier_training_arg_run_name = None

        # The run name:
        run_name = formatted_log_base_name(current_date=datetime.datetime.now(),
                                           host=socket.gethostname(),
                                           dataset_name=dataset_loader_dict['dataset_name'],
                                           net_name=net, extra_field=None, flag_random_id=True)

        if mlflow_logging is not None and mlflow_logging != False:
            #
            ######################################
            # SET THE TRACKING URI FOR MLFLOW
            ######################################
            connect_to_mlflow(mlflow_logging)

            # The name of the experiment in MLflow: the dataset name plus all the options which are not default in the \
            # dataset creation function
            experiment_name = f"examples_2D_classifier_{dataset_loader_dict['dataset_name']}"
            all_acceptable_args_dataset_creation = \
                inspect.signature(obtain_classification_dataset_loaders_from_point_data).parameters
            remaining_kwargs_dataset = copy.deepcopy(kwargs_dataset)
            for key in all_acceptable_args_dataset_creation:
                remaining_kwargs_dataset.pop(key, None)
            pass
            experiment_name += ''.join(
                [f"_{key}-{remaining_kwargs_dataset[key]}" for key in sorted(remaining_kwargs_dataset)])
            experiment_name += "" if suffix_experiment is None else f"_{suffix_experiment}"

            # Create the experiment and run and extract what could be necessary for the function 'classifier_training(...)'
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=run_name,
                                          experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
                                          log_system_metrics=True)
            classifier_training_arg_mlflow_run_id = mlflow_run.info.run_id
            classifier_training_arg_run_name = None
            #
        else:
            classifier_training_arg_mlflow_run_id = None
            classifier_training_arg_run_name = run_name
        pass

        # Run the training

        classifier_nn.to_device(computation_device)
        result_classifier_training = classifier_training(
            classifier_nn, dataset_loader_dict, **kwargs_optim_scheduler,
            mlflow_run_id=classifier_training_arg_mlflow_run_id, run_name=classifier_training_arg_run_name,
            local_log_folder=None, verbose=verbose
        )

        # Stop the logging
        if mlflow_logging is not None and mlflow_logging != False:
            mlflow.end_run()
        pass

        # Evaluate the acc of the training and decide whether retraining is needed
        if acc_threshold is None or result_classifier_training.best_acc >= acc_threshold:
            break
        else:
            print(f"\n·············································")
            print((f"Training with insufficient acc: {100*result_classifier_training.best_acc:.2f} % " +
                   f"(th={100*acc_threshold:.2f})"))
            # Delete the run (if existing)
            if mlflow_logging is not None and mlflow_logging != False:
                print(f"Deletion of the unsuccessful run...", end="")
                mlflow.delete_run(mlflow_run.info.run_id)
                print(f" DONE!")
            pass
            print(f"·············································\n")
            # And stay in the "for" for another iteration
        pass
        #
    pass

    # Print a short summary of the results
    print("Training finished!")
    print(f"Best model corresponds to:")
    print(f"\tacc  = {100 * result_classifier_training.best_acc:8.2f} %")
    print(f"\tloss = {result_classifier_training.best_loss:10.4f}")
    #
    # Return the result of the training: to this end...
    # - we create an identical classifier to the trained one...
    # - and load into it the parameters of the best model
    best_classifier_nn = _dict_classifiers_as_in_conf_file[net](**kwargs_classifier)
    best_classifier_nn.load_state_dict(result_classifier_training.best_model)

    return best_classifier_nn
