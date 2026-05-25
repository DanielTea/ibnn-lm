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

import tomli
import socket
import datetime
import os

import traceback

import argparse
import pathlib
import itertools

import json
import numpy as np

import copy

import torch
import torch.nn as nn
from sklearn.utils.multiclass import type_of_target
from torch import optim
from torch.utils.data import DataLoader, random_split

from applications import _dict_classifiers_as_in_conf_file
from experimental_evaluation import _dict_optimizer_classes, _dict_scheduler_classes, _dict_loss_functions
from experimental_evaluation.operations_for_datasets import _dict_dataset_info_and_constructor

import mlflow
import mlflow.pytorch


######################################################
######################################################
# Auxiliary functions addressing convenient multilevel dictionary access
######################################################
######################################################

def replace_dict_entries(dict_modifier, dict_modified):
    """
    It uses the entries of the dictionary ``dict_modifier`` to substitute the corresponding entries of the \
    dictionary ``dict_modified``. In a way, the first ``dict_modifier`` can be regarded (and needs to be) an \
    incomplete version of the bigger, second dictionary ``dict_modified`` that substitutes only \
    specific values of the latter.
    It acts **inplace** on the second dictionary ``dict_modified``.

    Parameters
    ----------
    dict_modifier : dict
    dict_modified : dict
    """

    for key in dict_modifier:
        if isinstance(dict_modifier[key], dict):
            replace_dict_entries(dict_modifier[key], dict_modified[key])
        else:
            dict_modified[key] = dict_modifier[key]
        pass
    pass


def get_keypaths_exisiting_dict_elements(dictionary):
    """
    It returns a list with the hierarchical list, gone through sequentially, indicating the path to each field of \
    the input ``dictionary`` effectively filled.

    Parameters
    ----------
    dictionary : dict

    Returns
    -------
    list[list[str]]
        Each element *i* of the returned list is a list of the (string) keys leading to the effective fields
    """
    list_of_paths = []
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            list_subpaths = get_keypaths_exisiting_dict_elements(dictionary[key])
            for subpath in list_subpaths:
                completed_path = subpath
                completed_path.insert(0, key)
                list_of_paths.append(completed_path)
        else:
            new_path = [key]
            list_of_paths.append(new_path)
        pass
    pass
    return list_of_paths


def get_multilevel_dict_element(dictionary, dict_path, default_returned_element=None):
    """
    Obtain the value contained, in the multi-level dictionary `dictionary`, in the path of keys indicated by the \
    list of strings/keys `dict_path`.
    If the ``dict_path`` does not exist, it returns `default_returned_element`

    Parameters
    ----------
    dictionary : dict
    dict_path : list[str]
        List containing the keys leading to the requested element of `dictionary`
    default_returned_element : object
    Returns
    -------
    object
    """

    returned_element = default_returned_element
    try:
        current_level = dictionary
        for key in dict_path:
            current_level = current_level[key]
        pass
        returned_element = current_level
    except KeyError as e:
        pass
    except:
        raise

    return returned_element


def set_multilevel_dict_element(dictionary, dict_path, value):
    """
    Set the object ``value`` in the ``dictionary`` in the path of keys indicated by the \
    list of strings/keys ``dict_path``.
    It acts **inplace** on the dictionary ``dictionary``.
    If the path does not exist already in ``dictionary`` the function raises an exception.

    Parameters
    ----------
    dictionary : dict
    dict_path : list[str]
        List containing the keys leading to the requested element of ``dictionary``
    value : object
    """
    current_level_dictionary = dictionary
    for key in dict_path[:-1]:
        if key not in current_level_dictionary:
            complete_keypath_str = "[" + "][".join(dict_path) + "]"
            raise Exception(f"The requested keypath {complete_keypath_str} not reachable in the dictionary. " +
                            f"In particular: key '{key}' not found!")
        else:
            current_level_dictionary = current_level_dictionary[key]
        pass
    pass
    current_level_dictionary[dict_path[-1]] = value


def indented_print_dict(dictionary, indent_str=""):
    """
    It prints a potentially multilevel dictionary, with the given general indent ``indent_str`` if desired.

    Parameters
    ----------
    dictionary : dict
        The (potentially multilevel) dictionary to display.
    indent_str : str, optional
        General indentation of the printed scheme. \
        Default: ``""`` (no initial indentation)
    """
    increment_indent = "     "
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            print(f"{indent_str}========================================")
            print(f"{indent_str}{key}")
            print(f"{indent_str}========================================")
            indented_print_dict(dictionary[key], indent_str=indent_str + increment_indent)
        else:
            print(f"{indent_str}{key} {str(type(dictionary[key]))} = " + str(dictionary[key]))
        pass
    pass


def value_if_bool_false(value, value_if_false):
    """
    It checks if ``value`` is bool and is ``False``, which in the experiment specifications means usually ``None`` \
    (for that case, ``value_if_false`` should be set to ``None``) or other meaning.

    Parameters
    ----------
    value : object
        The intended and queried value
    value_if_false : object
        The value in case ``value`` is ``bool`` and is ``False``.
    """
    value_to_return = value_if_false if isinstance(value, bool) and (value == False) else value
    return value_to_return


def resolve_path_from_base_dict_and_group_dict(field_dict_path, abs_group_file, group_dict, abs_base_file, base_dict):
    """
    This function adapts folder/path-related fields present in the base and group configuration files (more precisely, \
    their resulting dictionaries) to the absolute path of the correct file. That is:

    - When the field of interest is present and not `None` or `False` in the group dictionary, then such \
      path is taken from said group configuration file and is resolved \
      relative to the absolute path of the group file.

    - When the field of interest is not present or is `None` or `False` in the group dictionary, then such \
      path is taken from the base configuration file and is resolved \
      relative to the absolute path of the base file.

    Parameters
    ----------
    field_dict_path : list[str]
        How to reach the field containing the desired path in both `group_dict and `base_dict` dictionaries
    abs_group_file : str
        The absolute path to the group experiment specification file from which `group_dict` was loaded
    group_dict : dict
        The dictionary containing the group experiment specification
    abs_base_file
        The absolute path to the group experiment specification file pointed at \
        by the group experiment specification file, whose content is in `base_dict`
    base_dict
        The dictionary containing the base experiment specification


    Returns
    -------
    :py:class:`pathlib.PosixPath`
        The resolved path
    """

    rel_path_from_base_dict = get_multilevel_dict_element(base_dict, field_dict_path)
    resolved_path_according_to_base_file = \
        pathlib.Path(abs_base_file).parents[0] / pathlib.Path(rel_path_from_base_dict)
    resolved_path = resolved_path_according_to_base_file

    rel_path_from_group_dict = get_multilevel_dict_element(group_dict, field_dict_path, default_returned_element=None)
    if (rel_path_from_group_dict is not None) and (rel_path_from_group_dict is not False):
        resolved_path = pathlib.Path(abs_group_file).parents[0] / pathlib.Path(rel_path_from_group_dict)
    pass

    return str(resolved_path)


#########################################################################################
#########################################################################################
# Experiment specification file reading
#########################################################################################
#########################################################################################


def read_experiment_group_specification_file(experiment_group_specification_file):
    """
    It reads the provided `experiment_group_specification_file`, a TOML according to the format set in \
    the file ``CURRENT_FORMATS.md``, and provides a list where each element \
    is the dictionary of each individual experiment resulting from the expansion of the hyperparameters indicated \
    in the group file.

    The function additionally adapts the fields related to folders (e.g. dataset root folder, logging folders; \
    by assessing fields containing the word "folder" or "path") to the absolute path of the group file \
    or the base file referred to by the group file, depending on the case.

    Parameters
    ----------
    experiment_group_specification_file : str
        The path to the group experiment specification file, which is a TOML file containing the \
        hyperparameters to be used in the experiments of the group

    Returns
    -------
    dict_general_info_group_experiment : dict
        Dictionary containing the general information of the group experiment, namely `'experiment_name'` (str), \
        `'purpose'` (str), `'base_experiment_specification_file'` (str, a path to a file), \
        `'mlflow_logging'` (bool), and `'local_log_folder'` (str a path to a folder or `None` if not provided), \
        as provided in the group specification file
    list_of_individual_dict_individual_experiment_specifications_pre_filtering : list[dict]
        List where each element is the dictionary of each individual experiment resulting from \
        the expansion of the hyperparameters indicated in the group file `experiment_group_specification_file`
    list_of_paths_to_existing_hyperparameter_elements : list[list[str]]
        List wherein each element contain the path to each parameter requested as hyperparameter by the group file, \
        in the form of a list of strings/path to child in the dictionary
    """

    ######################################
    # Load the group experiment specification in 'experiment_group_specification_file'
    ######################################

    abs_group_experiment_specification_file = pathlib.Path(experiment_group_specification_file)
    print(f"Read the group TOML file {abs_group_experiment_specification_file}   ...")
    with open(abs_group_experiment_specification_file, "rb") as f:
        dict_hyperparam_specification = tomli.load(f)

    ######################################
    # We well perform these tasks at this point:
    #    1. We will extract all the fields of the group specification dictionary 'dict_hyperparam_specification' which
    #       are in fact not hyperparameters, and check them. Those fields are actually the outmost fields of the file.
    #    2. We store those which are definitive in the resulting 'dict_general_info_group_experiment'
    #    3. We will REMOVE those fields from the dictionary so the remaining fields are all hyperparameters
    ######################################
    # Regarding extracting the outmost fields of the group experiment file, which are (not hyperparameters):
    #    - experiment_name (str)
    #    - purpose (str)
    #    - base_experiment_specification_file (str, a path to a file)
    #    - mlflow_logging (bool)
    #    - local_log_folder (str, a path to a folder or false)
    ######################################

    # Check that the group experiment specification file is a dictionary
    dict_general_info_group_experiment = {}

    # Extract the experiment name for logging in MLFlow
    if 'experiment_name' not in dict_hyperparam_specification:
        raise Exception(f"Expected 'experiment_name' in group file '{experiment_group_specification_file}', " +
                        f"but it is not present!")
    else:
        experiment_name = dict_hyperparam_specification.pop('experiment_name', None)
        if isinstance(experiment_name, str) and experiment_name:  # Non-empty string
            print(f"MLFlow EXPERIMENT NAME: {experiment_name}")
            dict_general_info_group_experiment['experiment_name'] = experiment_name
        else:
            raise Exception(f"Expected 'experiment_name' in group file '{experiment_group_specification_file}' to be " +
                            f"a non-empty string, but it is {type(experiment_name)} with value {experiment_name}!")
        pass
    pass

    # Extract the purpose name for logging in MLFlow
    if 'purpose' not in dict_hyperparam_specification:
        raise Exception(f"Expected 'purpose' in group file '{experiment_group_specification_file}', " +
                        f"but it is not present!")
    else:
        purpose = dict_hyperparam_specification.pop('purpose', None)
        print(f"EXPERIMENT PURPOSE: {purpose}")
        dict_general_info_group_experiment['purpose'] = purpose
    pass

    # Extract the 'base_experiment_specification' file path/name for the experiment group
    if 'base_experiment_specification_file' not in dict_hyperparam_specification:
        raise Exception(f"Expected 'base_experiment_specification_file' in group file " +
                        f"'{experiment_group_specification_file}', but it is not present!")
    else:
        base_experiment_specification_file = \
            dict_hyperparam_specification.pop('base_experiment_specification_file', None)
        if isinstance(base_experiment_specification_file,
                      str) and base_experiment_specification_file:  # Non-empty string
            dict_general_info_group_experiment['base_experiment_specification_file'] = \
                base_experiment_specification_file
        else:
            raise Exception(f"Expected 'base_experiment_specification_file' in group file " +
                            f"'{experiment_group_specification_file}' to be " +
                            f"a non-empty string, but it is {type(base_experiment_specification_file)} " +
                            f"with value {base_experiment_specification_file}!")
        pass
    pass

    # Extract the bool option 'mlflow_logging'
    if 'mlflow_logging' not in dict_hyperparam_specification:
        raise Exception(f"Expected 'mlflow_logging' in group file " +
                        f"'{experiment_group_specification_file}', but it is not present!")
    else:
        mlflow_logging = dict_hyperparam_specification.pop('mlflow_logging', None)
        flag_enabled_mlflow_logging = False
        if isinstance(mlflow_logging, bool):
            flag_enabled_mlflow_logging = mlflow_logging
        elif isinstance(mlflow_logging, str):
            flag_enabled_mlflow_logging = True
        else:
            raise Exception(f"Expected 'mlflow_logging' in group file " +
                            f"'{experiment_group_specification_file}' to be a boolean or str, but it is " +
                            f"{type(mlflow_logging)} with value {mlflow_logging}!")
        pass
        #
        str_status = "enabled" if flag_enabled_mlflow_logging else "disabled"
        print(f"MLFlow logging {str_status} for the group experiment '{experiment_name}' in " +
              f"file '{experiment_group_specification_file}' (mlflow_logging = {mlflow_logging})")
        dict_general_info_group_experiment['mlflow_logging'] = mlflow_logging
    pass

    # Extract the 'local_log_folder", if provided; if false, set as 'None'
    if 'local_log_folder' not in dict_hyperparam_specification:
        raise Exception(f"Expected 'local_log_folder' in group file " +
                        f"'{experiment_group_specification_file}', but it is not present!")
    else:
        local_log_folder = value_if_bool_false(
            dict_hyperparam_specification.pop('local_log_folder', None),
            None
        )
        if local_log_folder is not None and not isinstance(local_log_folder, str):
            raise Exception(
                f"Expected 'local_log_folder' in group file " +
                f"'{experiment_group_specification_file}' to be a string or None, but it is " +
                f"{type(local_log_folder)} with value {local_log_folder}!")
        else:
            dict_general_info_group_experiment['local_log_folder'] = local_log_folder
        pass
    pass

    # Now all the remaining parameters/fields are hyperparameters

    ######################################
    ######################################
    # Load the base experiment (adapting paths if necessary),
    # and add the hyperparameter combinations within the group file (correcting paths if necessary) into it
    ######################################
    ######################################

    fields_name_hints_to_make_abs = ['folder', 'path', 'file', 'directory', 'root']

    ######################################
    # Regarding the base experiment:
    #    - load the base specification, and
    #    - adapt all the paths of the base file to the absolute path of the base file
    ######################################

    if (dict_general_info_group_experiment['base_experiment_specification_file'] is None) or \
            (dict_general_info_group_experiment['base_experiment_specification_file'] is False) or \
            (dict_general_info_group_experiment['base_experiment_specification_file'] == ''):
        raise Exception(f"No 'base_experiment_specification' field in the group experiment specification file!")
    else:
        # Make absolute path and load it!
        abs_base_experiment_specification_file = \
            abs_group_experiment_specification_file.parents[0] / \
            pathlib.Path(dict_general_info_group_experiment['base_experiment_specification_file'])
        # Provisional group experiment specification, pending to add
        # the modifications in "dict_raw_group_specification['general_modifications_to_base']"
        print(f"Read the base TOML file {abs_base_experiment_specification_file}   ...")
        with open(abs_base_experiment_specification_file, "rb") as f:
            dict_base_experiment_specification = tomli.load(f)
        pass

        ## EXCEPTION! WE WILL TRANSFORM ALL THE FIELDS SUGGESTING A PATH OR FILE
        ## (guided by the list "fields_name_hints_to_make_abs" defined above, listing names suggesting paths)
        ## SO THEY ARE MADE RELATIVE TO THE PATH OF THE FILE
        list_of_keypaths = get_keypaths_exisiting_dict_elements(dict_base_experiment_specification)
        for keypath in list_of_keypaths:
            # Check if it is likely that the key corresponds to a folder of a path
            if any(elem in keypath[-1] for elem in fields_name_hints_to_make_abs):
                param_to_make_abs = get_multilevel_dict_element(
                    dict_base_experiment_specification, keypath, default_returned_element=None
                )
                if isinstance(param_to_make_abs, str):
                    new_param_to_make_abs = abs_base_experiment_specification_file.parents[0] / \
                                            pathlib.Path(param_to_make_abs)
                    set_multilevel_dict_element(dict_base_experiment_specification,
                                                keypath, str(new_param_to_make_abs))
                elif param_to_make_abs is None:
                    pass
                elif isinstance(param_to_make_abs, bool) and (param_to_make_abs == False):
                    pass
                else:
                    # field_to_make_abs_to_str = ".".join(field_to_make_abs)
                    raise Exception(f"Field '{keypath}' in the base experiment specification file " +
                                    f"is not a string: type {type(param_to_make_abs)} found " +
                                    f"with value {param_to_make_abs}!")
                pass
            pass
        pass
    pass

    ######################################
    # Read the hyperparameters indicated in the 'experiment_group_specification_file' and plug all the
    # hyperparameter combinations into the base
    ######################################

    list_of_paths_to_existing_hyperparameter_elements = get_keypaths_exisiting_dict_elements(
        dict_hyperparam_specification
    )

    # Make sure that all the contained hyperparameters are lists! Otherwise raise an exception
    for path_to_hyperparam in list_of_paths_to_existing_hyperparameter_elements:
        hyperparam_value = get_multilevel_dict_element(dict_hyperparam_specification, path_to_hyperparam)
        if not isinstance(hyperparam_value, list):
            raise Exception(f"All parameters in the experiment configuration 'group' file must be " +
                            f"hyperparameters (e.g. lists); however '{'.'.join(path_to_hyperparam)}' is not a list! " +
                            f"type {type(hyperparam_value)} found with value {hyperparam_value}!")
        pass
    pass

    # Make correct absolute paths for the hyperparameters that are folders/paths (remember: all elements are lists!!!)
    for keypath in list_of_paths_to_existing_hyperparameter_elements:
        # Check if it is likely that the key corresponds to a folder of a path
        if any(elem in keypath[-1] for elem in fields_name_hints_to_make_abs):
            list_params_to_make_abs = get_multilevel_dict_element(
                dict_hyperparam_specification, keypath, default_returned_element=None
            )
            # In the group experiment specification file, the hyperparameter MUST BE a list of strings
            for ind, param_to_make_abs in enumerate(list_params_to_make_abs):
                if isinstance(param_to_make_abs, str):
                    new_param_to_make_abs = abs_group_experiment_specification_file.parents[0] / \
                                            pathlib.Path(param_to_make_abs)
                    list_params_to_make_abs[ind] = str(new_param_to_make_abs)
                elif param_to_make_abs is None:
                    pass
                elif isinstance(param_to_make_abs, bool) and (param_to_make_abs == False):
                    pass
                else:
                    # field_to_make_abs_to_str = ".".join(field_to_make_abs)
                    raise Exception(f"Field '{keypath}' in the group experiment specification file " +
                                    f"is not a string: type {type(param_to_make_abs)} found " +
                                    f"with value {param_to_make_abs}!")
                pass
            pass
            set_multilevel_dict_element(dict_hyperparam_specification,
                                        keypath, list_params_to_make_abs)
        pass
    pass

    # Generate the cartesian products of all the combinations of the hyperparameters
    list_of_value_alternatives_in_existing_hyperparameter_elements = [
        get_multilevel_dict_element(dict_hyperparam_specification, path_existing_element)
        for path_existing_element in list_of_paths_to_existing_hyperparameter_elements
    ]
    list_of_cartesian_product_existing_hyperparameter_elements = itertools.product(
        *list_of_value_alternatives_in_existing_hyperparameter_elements)

    ######################################
    # Pack and return the resolved hyperparameter combinations
    ######################################

    # List to accumulate and return combinations
    list_of_individual_dict_individual_experiment_specifications_pre_filtering = []

    for i, hyperparameter_combination in enumerate(list_of_cartesian_product_existing_hyperparameter_elements):
        #
        # Raw dictionary specification: before eliminating unnecessary fields
        raw_dict_individual_experiment_specification_i = copy.deepcopy(dict_base_experiment_specification)
        for j, path_field in enumerate(list_of_paths_to_existing_hyperparameter_elements):
            set_multilevel_dict_element(raw_dict_individual_experiment_specification_i, path_field,
                                        hyperparameter_combination[j])
        pass
        #
        # Add/append to the list
        list_of_individual_dict_individual_experiment_specifications_pre_filtering.append(
            raw_dict_individual_experiment_specification_i)
    pass

    ######################################
    # Print a summary of all the combinations
    ######################################

    print(f"SUMMARY:")

    # Summary of the raw parameter combinations to consider in the experiments of the group
    print(f"    - Hyperparameter values included in the group experimental specification:")
    num_raw_combinations = 1
    for p, v in zip(list_of_paths_to_existing_hyperparameter_elements,
                    list_of_value_alternatives_in_existing_hyperparameter_elements):
        num_raw_combinations *= len(v)
        print(f"          {' > '.join(p)}  =  {str(v)}")
    pass
    print(f"    - Total number of hyperparameter combinations before filtering: {num_raw_combinations:6d}")

    ######################################
    # Return the list
    ######################################

    return dict_general_info_group_experiment, \
        list_of_individual_dict_individual_experiment_specifications_pre_filtering, \
        list_of_paths_to_existing_hyperparameter_elements


def duplicated_experiment_specification_removal(list_of_experiment_dicts):
    """
    It removes the duplicated experiment specifications from the list of dictionaries \
    `list_of_experiment_dicts`, which is a list of dictionaries, each one containing the \
    hyperparameters of an individual experiment.

    Parameters
    ----------
    list_of_experiment_dicts : dict
        List of dictionaries, each one containing the hyperparameters of an individual experiment

    Returns
    -------
    dict
        List of dictionaries, each one containing the hyperparameters of an individual experiment, \
        with the duplicated ones removed

    """

    # Create a dict where the keys are the string representation of the corresponding dictionary!
    hashed_key_dict_of_dict_individual_experiment_specifications = {}
    for raw_dict_individual_experiment_specification_i in list_of_experiment_dicts:
        hashed_key_dict_of_dict_individual_experiment_specifications[
            str(raw_dict_individual_experiment_specification_i)] = \
            raw_dict_individual_experiment_specification_i

    # Use those keys to filter the elements!
    list_of_experiment_dicts_after_filtering = \
        list(hashed_key_dict_of_dict_individual_experiment_specifications.values())

    return list_of_experiment_dicts_after_filtering


######################################################
# Auxiliary functions to 'extract_classifier_kwargs_from_experiment_specification'
######################################################


def _overall_field_conv_layers_or_field_per_conv_layer(dict_architecture_specific,
                                                       field_name, overall_field, num_conv_like_layers,
                                                       error_if_not_present=True):
    """
    Auxiliary function.
    The role of this function is the following: it checks whether the field `'field_name'` \
    (e.g. `'m_kernel_size_per_conv_layer'`) is present in 'dict_architecture_specific', corresponding precisely \
    to the part of the [classifier] field of the conf. files aimed at the architecture-specific parameters. \
    Then the possible options are the following:
    - If that field is not present at all then `None` is returned if not error_if_not_present but an exception is \
      raised if it is true.
    - If that field is present:
      - If it is False (not an array or tuple of Falses, but a bool False) then `overall_field` is used to fill it.
      - If it is not False then it is used as it is (after checks)
    """
    #
    assert isinstance(field_name, str), "Expected 'field_name' to be a string!"
    #
    field_per_conv_layer = None
    if not field_name in dict_architecture_specific:
        if error_if_not_present:
            raise Exception(f"Expected '{field_name}' or its 'overall_..._conv_layers' version in " +
                            f"the architecture-specific part of the configuration file but they it is not present!")
        else:
            pass
        pass
    else:
        value_field_name = dict_architecture_specific[field_name]
        if isinstance(value_field_name, bool) and (value_field_name is False):
            if overall_field is not None:
                assert isinstance(num_conv_like_layers, int) and (num_conv_like_layers > 0), \
                    "Expected 'num_conv_like_layers' to be a positive integer: " + \
                    f"it is {type(num_conv_like_layers)} with value {num_conv_like_layers}!"
                field_per_conv_layer = [overall_field] * num_conv_like_layers
                field_per_conv_layer = tuple(field_per_conv_layer)
            elif error_if_not_present:
                raise Exception(f"Expected '{field_name}' or its 'overall_..._conv_layers' version in " +
                                f"the configuration file but they are not present!")
            pass
        elif isinstance(value_field_name, (list, tuple)):
            if (num_conv_like_layers is not None) and (len(value_field_name) != num_conv_like_layers):
                raise Exception(f"Expected '{field_name}' to have length {num_conv_like_layers} " +
                                f"since there are {num_conv_like_layers} conv-like layers, " +
                                f"but it has length {len(value_field_name)}!")
            pass
            field_per_conv_layer = tuple(copy.deepcopy(value_field_name))
        else:
            raise Exception(f"Expected '{field_name}' to be a list or tuple " +
                            (f"of length {num_conv_like_layers}, " if (num_conv_like_layers is not None) else "") +
                            f"or the boolean False, but it is {type(value_field_name)} with value " +
                            f"{value_field_name}!")
        pass
    pass
    #
    return field_per_conv_layer

######################################################


def extract_classifier_kwargs_from_experiment_specification(dict_individual_experiment_specification,
                                                            in_size=None, in_channels=None, out_classes=None,
                                                            net=None,
                                                            error_if_not_present=True):
    """
    It returns the keyword arguments for the construction of the classifier object of the type specified in \
    `dict_individual_experiment_specification['classifier']['net']`, if no other `net` is provided, or by `net`.
    If an expected variable or parameter is not present in `dict_individual_experiment_specification['classifier']`, \
    it raises an exception if `error_if_not_present` is `True`, or simply ignores it if `error_if_not_present` is \
    `False`.

    **NOTES on this version:** The following modifications have been introduced:
    - The networks "single_layer" and "double_layer" have been directly discontinued! The flexibility that \
      the "multi_layer" net offers makes the other two redundant and even confusing.
    - The variable
      - ``[classifier.architecture_specific.multi_layer][conv_block_specification]`` has been \
      substituted for the variable
      - ``[classifier.architecture_specific.multi_layer][num_conv_like_layers]`` since only blocks with \
      one single conv-like layer (e.g. [1, 1, ..., 1]) were allowed in the ``multi_layer`` anyway. (The argument \
      ``conv_block_specification`` is still used for VGGx, AlexNet, and EfficientNet, though).
    - The variables...
      - ``[classifier][maxpool_reduction]``
      - ``[classifier][batch_normalization]``
      - ``[classifier.conv_like_layer.traditional][phi_activation]``
      - ``[classifier.conv_like_layer.traditional][m_kernel_size]``
      are now reformulated to have, instead, their name changed to 'overall_<previous_name>_conv_layers', and are to \
      be understood as non-compulsory and applying to all the convolution-like layers of the classifier \
      if no more specific values were indicated in the architecture-specific block per-layer. Additionally, the have \
      been all moved to ``[classifier.conv_like_layer.traditional]``.
    - Additionally, the new block
      - ``[classifier.fully_connected]``
      have been introduced, containing the compulsory variables...
      - ``[classifier.fully_connected][fc_num_layers]``
      - ``[classifier.fully_connected][fc_num_units_intermediate_layers]``
      - ``[classifier.fully_connected][fc_batch_normalization]``
      - ``[classifier.fully_connected][fc_dropout]``
      - ``[classifier.fully_connected][penciled_decision]``
      - ``[classifier.fully_connected][softmax_output]``
      The two latter have been removed from the initial part of the file.

    Parameters
    ----------
    dict_individual_experiment_specification : dict
    in_size : tuple[int]
        2D tuple, input size
        Default: `None`
    in_channels : int
        Number of input channels.
        Default: `None`
    out_classes : int
        Number of output classes
        Default: `None`
    net : str, optional
        Value among 'single_layer', 'double_layer', 'multi_layer' 'vggx', 'alexnet', 'efficientnetv2s'. \
        The type of classifier to create, overriding the one specified in \
        `dict_individual_experiment_specification['classifier']['net']`. \
        If `None`, it uses the one specified in `dict_individual_experiment_specification['classifier']['net']`.
        Default: `None`
    error_if_not_present : bool, optional
        If `True`, it raises an exception if the classifier specification is not present in \
        `dict_individual_experiment_specification['classifier']`. If `False`, it returns `None` in that case. \
        Default: `True`

    Returns
    -------
    kwargs_arguments_classifier : dict
        Kwargs in the format that the constructor of the class of the corresponding net \
        would take to create the corresponding object
    net : str
        One of the values ``'single_layer'``, ``'double_layer'``, ``'vggx'``, ``'alexnet'``, \
        ``'efficientnetv2s'``, the classifier types considered so far.
    """

    ######################################
    # Classifier creation: extraction of parameters (as keyword arguments) for the construction of the classifier
    ######################################
    # - Extraction of common parameters for different network types, shared.
    # - Specific parameters for each classifier type, branched depending on the
    #   'dict_individual_experiment_specification['classifier']['net']'
    ######################################

    kwargs_arguments_classifier = {}

    ######################################
    ######################################
    # Size parameters, if provided
    ######################################
    ######################################

    if in_size is not None:
        assert isinstance(in_size, (tuple, list)) and len(in_size) == 2, \
            f"Expected 'in_size' to be a 2D tuple or list, but received {in_size} of type {type(in_size)}!"
        assert all([isinstance(elem, int) for elem in in_size]), \
            f"Expected 'in_size' to be a 2D tuple or list of integers, but received {in_size} of type {type(in_size)}!"
        kwargs_arguments_classifier['in_size'] = in_size
    pass
    if in_channels is not None:
        assert isinstance(in_channels, int) and (in_channels > 0), \
            f"Expected 'in_channels' to be a positive integer, but received {in_channels} of type {type(in_channels)}!"
        kwargs_arguments_classifier['in_channels'] = in_channels
    pass
    if out_classes is not None:
        assert isinstance(out_classes, int) and (out_classes > 0), \
            f"Expected 'out_classes' to be a positive integer, but received {out_classes} of type {type(out_classes)}!"
        kwargs_arguments_classifier['out_classes'] = out_classes
    pass

    ######################################
    ######################################
    # Common parameters for all classifier types
    ######################################
    ######################################

    # The net to "presume" for the parameter extraction: we need it to extract the parameters specific for the
    # classifier/architecture type
    net = dict_individual_experiment_specification['classifier']['net'] if net is None else net

    # It could only be used if it is defined in the experiment specification
    overall_m_kernel_size_conv_layers = None
    overall_phi_activation_conv_layers = None
    overall_batch_normalization_conv_layers = None
    overall_maxpool_reduction_conv_blocks = None

    # Check the rest of COMPULSORY first-level order in 'dict_individual_experiment_specification['classifier']'
    for key in ['conv_like_type', 'conv_like_type_position', 'prenormalization']:
        if key in dict_individual_experiment_specification['classifier']:
            kwargs_arguments_classifier[key] = dict_individual_experiment_specification['classifier'][key]
        elif error_if_not_present:
            raise Exception(f"Expected {key} in 'dict_individual_experiment_specification['classifier']', " +
                            f"but it is not present!")
        pass
    pass

    ######################################
    ######################################
    # Conv-like layer parameters (except for those corresponding to the specific classifier architecture)
    ######################################
    ######################################

    dict_conv_like_layer = None
    if 'conv_like_layer' in dict_individual_experiment_specification['classifier']:
        dict_conv_like_layer = dict_individual_experiment_specification['classifier']['conv_like_layer']
    else:
        if error_if_not_present:
            raise Exception(
                f"Expected {'conv_like_layer'} in 'dict_individual_experiment_specification['classifier']', " +
                f"but it is not present!")
        pass
    pass

    dict_overall_field_conv_layers = {}
    if dict_conv_like_layer is not None:
        #
        if kwargs_arguments_classifier.get('conv_like_type', None) in \
                ['sm', 'inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn', None]:
            #
            ###############################################
            dict_traditional = None
            if 'traditional' in dict_individual_experiment_specification['classifier']['conv_like_layer']:
                dict_traditional = dict_individual_experiment_specification['classifier']['conv_like_layer'][
                    'traditional']
            else:
                if error_if_not_present:
                    raise Exception(
                        f"Expected {'traditional'} in 'dict_individual_experiment_specification['classifier']['conv_like_layer']', " +
                        f"but it is not present!")
                pass
            pass
            ###############################################
            #
            # The parameters in the list defined next are "special": they are compulsory, although their value \
            # could be `False` meaning that they are left blank deliberately "to fill them with defaults".
            dict_overall_field_conv_layers = {}
            for name in ['m_kernel_size', 'phi_activation', 'batch_normalization', 'maxpool_reduction']:
                # Name for the "overall_...", e.g. default, value for the parameter
                overall_name_conv_layers = f"overall_{name}_conv_layers" \
                    if name != 'maxpool_reduction' else f"overall_{name}_conv_blocks"
                # Name for the "..._per_conv_layer/block", e.g. specific, value for the parameter
                name_per_conv_layer = f"{name}_per_conv_layer" \
                    if name != 'maxpool_reduction' else f"{name}_per_conv_block"
                # If the "overall_..." version is provided, copy it (not compulsory)
                if (dict_traditional is not None) and (overall_name_conv_layers in dict_traditional):
                    # FALSE if present and `false`, to distinguish it from None, corresponding no presence
                    dict_overall_field_conv_layers[name_per_conv_layer] = \
                        value_if_bool_false(dict_traditional[overall_name_conv_layers], False) \
                            if name != 'batch_normalization' \
                            else dict_traditional[overall_name_conv_layers] # batch_normalization can be boolean!!!
                else:
                    # If no "overall_..._conv_layers" is provided, we still create the key "..._per_conv_layer"
                    # so it will be processed later
                    dict_overall_field_conv_layers[name_per_conv_layer] = None
                    if error_if_not_present:
                        raise Exception(f"Expected '{overall_name_conv_layers}' or '{name_per_conv_layer}' in " +
                                        f"the 'traditional' part of the configuration file but none is present!")
                pass
            pass

            if dict_traditional is not None:
                # Check the rest of NON-COMPULSORY first-level order in 'dict_individual_experiment_specification['classifier']'
                # if 'overall_batch_normalization_conv_layers' in dict_traditional:
                #     overall_batch_normalization_conv_layers = \
                #         value_if_bool_false(dict_traditional['overall_batch_normalization_conv_layers'], None)
                # if 'overall_maxpool_reduction_conv_blocks' in dict_traditional:
                #     overall_maxpool_reduction_conv_blocks = \
                #         value_if_bool_false(dict_traditional['overall_maxpool_reduction_conv_blocks'], None)
                # if 'overall_phi_activation_conv_layers' in dict_traditional:
                #     # If 'overall_phi_activation' was set to false=None we do not include it in the arguments
                #     overall_phi_activation_conv_layers = \
                #         value_if_bool_false(dict_traditional['overall_phi_activation_conv_layers'], None)
                # if 'overall_m_kernel_size_conv_layers' in dict_traditional:
                #     # If 'overall_phi_activation' was set to false=None we do not include it in the arguments
                #     overall_m_kernel_size_conv_layers = \
                #         value_if_bool_false(dict_traditional['overall_m_kernel_size_conv_layers'], None)
                #
                for key in ['m_independent_channels', 'm_padding', 'm_padding_mode',
                            'm_initialization', 'm_trainable', 'b_type', 'initial_b', 'b_trainable']:
                    if key in dict_traditional:
                        kwargs_arguments_classifier[key] = dict_traditional[key]
                    elif error_if_not_present:
                        raise Exception(f"Expected {key} in " +
                                        f"dict_individual_experiment_specification['classifier']['conv_like_layer']['traditional'], " +
                                        f"but it is not present!")
                    pass
                pass
                if 'initial_b' in kwargs_arguments_classifier and \
                        isinstance(kwargs_arguments_classifier['initial_b'], list):
                    kwargs_arguments_classifier['initial_b'] = torch.Tensor(kwargs_arguments_classifier['initial_b'])
                pass
            pass    # END OF: if dict_traditional is not None:
        pass
        #
        ############################################################################################################
        #
        if kwargs_arguments_classifier.get('conv_like_type', None) in \
                ['inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn', None]:
            #
            ###############################################
            dict_nonlinear_bias = None
            if 'nonlinear_bias' in dict_individual_experiment_specification['classifier']['conv_like_layer']:
                dict_nonlinear_bias = dict_individual_experiment_specification['classifier']['conv_like_layer'][
                    'nonlinear_bias']
            else:
                if error_if_not_present:
                    raise Exception(f"Expected {'nonlinear_bias'} in " +
                                    f"'dict_individual_experiment_specification['classifier']['conv_like_layer']', " +
                                    f"but it is not present!")
                pass
            pass
            ###############################################
            #
            if dict_nonlinear_bias is not None:
                #
                for key in ['sigma_activation',
                            'sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset',
                            'sigma_x_compress_trainable', 'sigma_y_stretch_trainable',
                            'sigma_x_offset_trainable', 'sigma_y_offset_trainable',
                            'lambda_type', 'initial_lambda', 'lambda_trainable']:
                    if key in dict_nonlinear_bias:
                        kwargs_arguments_classifier[key] = dict_nonlinear_bias[key]
                    elif error_if_not_present:
                        raise Exception(
                            f"Expected {key} in 'dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias']', " +
                            f"but it is not present!")
                    pass
                pass
                #
                # if 'initial_lambda' in kwargs_arguments_classifier and \
                #         isinstance(kwargs_arguments_classifier['initial_lambda'], list):
                #     kwargs_arguments_classifier['initial_lambda'] = torch.Tensor(
                #         kwargs_arguments_classifier['initial_lambda']
                #     )
                # pass
                #
                ###############################################
                dict_cross_conv_computation = None
                if 'cross_conv_computation' in dict_nonlinear_bias:
                    dict_cross_conv_computation = dict_nonlinear_bias['cross_conv_computation']
                else:
                    if error_if_not_present:
                        raise Exception(f"Expected {'cross_conv_computation'} in " +
                                        f"dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias'], " +
                                        f"but it is not present!")
                    pass
                pass
                ###############################################
                #
                if dict_cross_conv_computation is not None:
                    for key in ['calculation_mode', 'num_sampling_points', 'memory_saving_version']:
                        if key in dict_cross_conv_computation:
                            kwargs_arguments_classifier[key] = dict_cross_conv_computation[key]
                        elif error_if_not_present:
                            raise Exception(f"Expected {key} in " +
                                            f"dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias']['cross_conv_computation'], " +
                                            f"but it is not present!")
                        pass
                    pass
                    if 'range_std_sigma' in dict_cross_conv_computation:
                        non_constant_range = dict_cross_conv_computation['range_std_sigma']
                        kwargs_arguments_classifier['start_range_std_activation'] = non_constant_range[0]
                        kwargs_arguments_classifier['end_range_std_activation'] = non_constant_range[1]
                    elif error_if_not_present:
                        raise Exception(f"Expected {'range_std_sigma'} in " +
                                        f"dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias']['cross_conv_computation'], " +
                                        f"but it is not present!")
                    pass
                    #
                pass
                #
                if kwargs_arguments_classifier.get('conv_like_type', None) in \
                        ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn', None]:
                    #
                    for key in ['w_kernel_size', 'w_independent_channels', 'w_padding_mode',
                                'w_initialization', 'w_trainable']:
                        if key in dict_nonlinear_bias:
                            kwargs_arguments_classifier[key] = dict_nonlinear_bias[key]
                        elif error_if_not_present:
                            raise Exception(f"Expected {key} in " +
                                            f"dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias'], " +
                                            f"but it is not present!")
                        pass
                    pass
                pass
            pass    # END OF: if dict_nonlinear_bias is not None
        pass
        #
        #####################################################################################################
        #
        if kwargs_arguments_classifier.get('conv_like_type', None) in ['ibnn_internal', 'ibnn', None]:
            #
            ###############################################
            dict_fixed_point = None
            if 'fixed_point' in dict_individual_experiment_specification['classifier']['conv_like_layer']:
                dict_fixed_point = \
                    dict_individual_experiment_specification['classifier']['conv_like_layer']['fixed_point']
            else:
                if error_if_not_present:
                    raise Exception(f"Expected {'fixed_point'} in " +
                                    f"dict_individual_experiment_specification['classifier']['conv_like_layer'], " +
                                    f"but it is not present!")
                pass
            pass
            ###############################################
            #
            if dict_fixed_point is not None:
                for key in ['batched_fixed_point',
                            'f_solver', 'f_max_iter', 'f_tol',
                            'f_tau',
                            'b_solver', 'b_max_iter', 'b_tol',
                            'abs_error_threshold']:
                    if key in dict_fixed_point:
                        value_for_key_resolving_false = value_if_bool_false(dict_fixed_point[key], None)
                        if value_for_key_resolving_false is not None:
                            kwargs_arguments_classifier[key] = value_for_key_resolving_false
                        pass
                    elif error_if_not_present:
                        raise Exception(
                            f"Expected {key} in " +
                            f"dict_individual_experiment_specification['classifier']['conv_like_layer']" +
                            f"['fixed_point'], " +
                            f"but it is not present!")
                    pass
                pass
                #
                # The f_solver "broyden" does not use 'f_tau': set to None if that is the case
                if kwargs_arguments_classifier.get('f_solver', None) == 'broyden':
                    kwargs_arguments_classifier['f_tau'] = None
                pass
            pass    # END OF: if dict_fixed_point is not None:
        pass
    pass    # END OF: if dict_conv_like_layer is not None:

    ######################################
    ######################################
    # FC-related
    ######################################
    ######################################

    ###############################################
    dict_fully_connected = None
    if 'fully_connected' in dict_individual_experiment_specification['classifier']:
        dict_fully_connected = dict_individual_experiment_specification['classifier']['fully_connected']
    else:
        if error_if_not_present:
            raise Exception(
                f"Expected {'fully_connected'} in 'dict_individual_experiment_specification['classifier']', " +
                f"but it is not present!")
        pass
    pass
    ###############################################
    #
    if dict_fully_connected is not None:
        for key in ['softmax_output', 'penciled_decision', 'fc_batch_normalization']: # booleans
            if key in dict_fully_connected:
                kwargs_arguments_classifier[key] = dict_fully_connected[key]
            elif error_if_not_present:
                raise Exception(
                    f"Expected {key} in 'dict_individual_experiment_specification['classifier']['fully_connected']', " +
                    f"but it is not present!")
            pass
        pass
        #
        for key in ['fc_num_layers', 'fc_num_units_intermediate_layers', 'fc_dropout']:  # scalars
            if key in dict_fully_connected:
                kwargs_arguments_classifier[key] = value_if_bool_false(dict_fully_connected[key], None)
            elif error_if_not_present:
                raise Exception(
                    f"Expected {key} in 'dict_individual_experiment_specification['classifier']['fully_connected']', " +
                    f"but it is not present!")
            pass
        pass
    pass

    ######################################
    ######################################
    # Address the architecture-specific parameters of each classifier type
    ######################################
    ######################################
    # Certain common notes for some classifier types:
    # - All classifiers have one or more (tunable) layer, and therefore all of them have:
    #   - a field 'm_kernel_size_per_conv_layer', which indicates, if not None,
    #     the size of the kernel in each (tunable) layer, OVERRIDING THE GENERAL "overall_m_kernel_size_conv_layers";
    #     if "false", the general "overall_m_kernel_size_conv_layers" is used for all layers.
    #   - a field 'phi_activation_per_conv_layers', which indicates, if not None,
    #     the size of the kernel in each (tunable) layer, OVERRIDING THE GENERAL "overall_m_kernel_size_conv_layers";
    #     if "false", the general "overall_m_kernel_size_conv_layers" is used for all layers.
    #   - a field 'batch_normalization_per_conv_layer'... OVERRIDING ... "overall_batch_normalization_conv_layers";
    #   - a field 'maxpool_reduction_per_conv_layer'... OVERRIDING ... "overall_maxpool_reduction_conv_blocks".
    ######################################
    ######################################

    dict_architecture_specific = None
    if 'architecture_specific' in dict_individual_experiment_specification['classifier'] and \
            isinstance(dict_individual_experiment_specification['classifier']['architecture_specific'], dict) and \
            (net in dict_individual_experiment_specification['classifier']['architecture_specific']):
        dict_architecture_specific = dict_individual_experiment_specification['classifier']['architecture_specific'][
            net]
    elif error_if_not_present:
        raise Exception(
            f"Expected {'architecture_specific'} in 'dict_individual_experiment_specification['classifier']', " +
            f"but it is not present or does not contain the key '{net}'!")
    pass

    if dict_architecture_specific is not None:
        if net == 'multi_layer':
            #
            num_conv_like_layers = None
            num_conv_like_blocks = None
            if 'num_conv_like_layers' in dict_architecture_specific:
                num_conv_like_layers = dict_architecture_specific['num_conv_like_layers']
                num_conv_like_blocks = num_conv_like_layers
                if isinstance(num_conv_like_layers, int) and (num_conv_like_layers > 0):
                    kwargs_arguments_classifier['conv_block_specification'] = \
                        tuple([1]*num_conv_like_layers)
                else:
                    raise Exception(f"Expected 'num_conv_like_layers' in " +
                                    f"dict_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"to be a list or tuple, but it is {type(dict_architecture_specific['num_conv_like_layers'])}!")
                pass
            elif error_if_not_present:
                raise Exception(f"Expected 'conv_block_specification' in " +
                                f"dict_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                f"but it is not present!")
            pass
            #
            for key in ['channels_per_conv_layer']:
                if key in dict_architecture_specific:
                    kwargs_arguments_classifier[key] = dict_architecture_specific[key]
                elif error_if_not_present:
                    raise Exception(f"Expected '{key}' in " +
                                    f"dict_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"but it is not present!")
                pass
            pass
            #
            # Resolving the content of the '..._per_conv_layer/block' fields in the 'dict_overall_field_conv_layers' \
            # by means of the aux function '_overall_field_conv_layers_or_field_per_conv_layer'.
            # Remember: when 'error_if_not_present'=True, the fields where the "default" behavior is intended MUST be \
            # explicitly indicated with (a single) `False`
            for key in dict_overall_field_conv_layers:
                if key in dict_architecture_specific:
                    num_items = num_conv_like_layers if 'maxpool_reduction' not in key else num_conv_like_blocks
                    kwargs_arguments_classifier[key] = _overall_field_conv_layers_or_field_per_conv_layer(
                        dict_architecture_specific,
                        field_name=key, overall_field=dict_overall_field_conv_layers[key],
                        num_conv_like_layers=num_items, error_if_not_present=False
                    )
                elif error_if_not_present:
                    raise Exception(f"Expected '{key}' or its 'overall_..._conv_layers' version in " +
                                    f"dict_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"but they are not present!")
            pass
            #
            # if 'm_kernel_size_per_conv_layer' in dict_architecture_specific and \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer'] is not False:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer']
            # elif m_kernel_size is not None:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         [m_kernel_size] * num_conv_like_layers
            # elif error_if_not_present:
            #     raise Exception(f"Expected 'm_kernel_size_per_conv_layer' in " +
            #                     f"dict_experiment_specification['classifier']['architecture_specific']['{net}'], " +
            #                     f"but it is not present!")
            # pass
            #
        elif net == 'vggx':
            #
            for unused_unallowed_key in []:
                if unused_unallowed_key in kwargs_arguments_classifier:
                    # Argument not yet used nor allowed for VGGx: remove it if present
                    kwargs_arguments_classifier.pop(unused_unallowed_key, None)
                    warnings.warn(
                        f"Argument '{unused_unallowed_key}' not yet used nor allowed for {net} networks: removing it from the arguments.",
                        RuntimeWarning
                    )
                pass
            pass
            #
            num_conv_like_layers = None
            if 'conv_block_specification' in dict_architecture_specific:
                if isinstance(dict_architecture_specific['conv_block_specification'], (list, tuple)):
                    kwargs_arguments_classifier['conv_block_specification'] = \
                        tuple(dict_architecture_specific['conv_block_specification'])
                    num_conv_like_layers = sum(num for num in kwargs_arguments_classifier['conv_block_specification'])
                else:
                    raise Exception(f"Expected 'conv_block_specification' in " +
                                    f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"to be a list or tuple, but it is {type(dict_architecture_specific['conv_block_specification'])}!")
                pass
            elif error_if_not_present:
                raise Exception(f"Expected 'conv_block_specification' in " +
                                f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                f"but it is not present!")
            pass
            #
            for key in dict_overall_field_conv_layers:
                kwargs_arguments_classifier[key] = _overall_field_conv_layers_or_field_per_conv_layer(
                    dict_architecture_specific,
                    field_name=key, overall_field=dict_overall_field_conv_layers[key],
                    num_conv_like_layers=num_conv_like_layers, error_if_not_present=False
                )
            pass
            # if 'm_kernel_size_per_conv_layer' in dict_architecture_specific and \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer'] is not False:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer']
            # elif m_kernel_size is not None:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         [m_kernel_size] * num_conv_like_layers
            # elif error_if_not_present:
            #     raise Exception(f"Expected 'm_kernel_size_per_conv_layer' in " +
            #                     f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
            #                     f"but it is not present!")
            # pass
            #
            for key in ['base_channels']:
                if key in dict_architecture_specific:
                    kwargs_arguments_classifier[key] = dict_architecture_specific[key]
                elif error_if_not_present:
                    raise Exception(f"Expected '{key}' in " +
                                    f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"but it is not present!")
                pass
            pass
            #
        elif net == 'alexnet':
            #
            for unused_unallowed_key in []:
                if unused_unallowed_key in kwargs_arguments_classifier:
                    # Argument not yet used nor allowed for VGGx: remove it if present
                    kwargs_arguments_classifier.pop(unused_unallowed_key, None)
                    warnings.warn(
                        f"Argument '{unused_unallowed_key}' not yet used nor allowed for {net} networks: removing it from the arguments.",
                        RuntimeWarning
                    )
                pass
            pass
            #
            num_conv_like_layers = None
            num_conv_like_blocks = None
            if 'conv_block_specification' in dict_architecture_specific:
                if isinstance(dict_architecture_specific['conv_block_specification'], (list, tuple)):
                    kwargs_arguments_classifier['conv_block_specification'] = \
                        tuple(dict_architecture_specific['conv_block_specification'])
                    num_conv_like_layers = sum(num for num in kwargs_arguments_classifier['conv_block_specification'])
                    num_conv_like_blocks = len(kwargs_arguments_classifier['conv_block_specification'])
                else:
                    raise Exception(f"Expected 'conv_block_specification' in " +
                                    f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"to be a list or tuple, but it is {type(dict_architecture_specific['conv_block_specification'])}!")
                pass
            elif error_if_not_present:
                raise Exception(f"Expected 'conv_block_specification' in " +
                                f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                f"but it is not present!")
            pass
            #
            for key in dict_overall_field_conv_layers:
                num_items = num_conv_like_layers if 'maxpool_reduction' not in key else num_conv_like_blocks
                kwargs_arguments_classifier[key] = _overall_field_conv_layers_or_field_per_conv_layer(
                    dict_architecture_specific,
                    field_name=key, overall_field=dict_overall_field_conv_layers[key],
                    num_conv_like_layers=num_items, error_if_not_present=False
                )
            pass
            # if 'm_kernel_size_per_conv_layer' in dict_architecture_specific and \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer'] is not False:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer']
            # elif m_kernel_size is not None:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         [m_kernel_size] * num_conv_like_layers
            # elif error_if_not_present:
            #     raise Exception(f"Expected 'm_kernel_size_per_conv_layer' in " +
            #                     f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
            #                     f"but it is not present!")
            # pass
            #
            for key in ['channels_per_conv_layer']:
                if key in dict_architecture_specific:
                    kwargs_arguments_classifier[key] = dict_architecture_specific[key]
                elif error_if_not_present:
                    raise Exception(f"Expected '{key}' in " +
                                    f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"but it is not present!")
                pass
            pass
            #
        elif net == 'efficientnetv2s':
            #
            for unused_unallowed_key in []:
                if unused_unallowed_key in kwargs_arguments_classifier:
                    # Argument not yet used nor allowed for VGGx: remove it if present
                    kwargs_arguments_classifier.pop(unused_unallowed_key, None)
                    warnings.warn(
                        f"Argument '{unused_unallowed_key}' not yet used nor allowed for {net} networks: removing it from the arguments.",
                        RuntimeWarning
                    )
                pass
            pass
            #
            num_conv_like_layers = 2
            for key in dict_overall_field_conv_layers:
                kwargs_arguments_classifier[key] = _overall_field_conv_layers_or_field_per_conv_layer(
                    dict_architecture_specific,
                    field_name=key, overall_field=dict_overall_field_conv_layers[key],
                    num_conv_like_layers=num_conv_like_layers, error_if_not_present=False
                )
            pass
            # if 'm_kernel_size_per_conv_layer' in dict_architecture_specific and \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer'] is not False:
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         dict_architecture_specific['m_kernel_size_per_conv_layer']
            # elif m_kernel_size is not None:
            #     num_conv_like_layers = 2
            #     kwargs_arguments_classifier['m_kernel_size_per_conv_layer'] = \
            #         [m_kernel_size] * num_conv_like_layers
            # elif error_if_not_present:
            #     raise Exception(f"Expected {'m_kernel_size_per_conv_layer'} in " +
            #                     f"dict_individual_experiment_specification['classifier']['architecture_specific']['double_layer'], " +
            #                     f"but it is not present!")
            # pass
            #
            for key in ['num_hidden_channels_2']:
                if key in dict_architecture_specific:
                    kwargs_arguments_classifier[key] = dict_architecture_specific[key]
                elif error_if_not_present:
                    raise Exception(f"Expected '{key}' in " +
                                    f"dict_individual_experiment_specification['classifier']['architecture_specific']['{net}'], " +
                                    f"but it is not present!")
                pass
            pass
            #
        else:
            raise Exception(
                f"Invalid option for 'dict_individual_experiment_specification['classifier']['net']': valid " +
                f"'single_layer', 'double_layer', 'multi_layer', 'vggx', 'alexnet', 'efficientnetv2s', but '{net}', received " +
                f"{dict_individual_experiment_specification['classifier']['net']}")
        pass
        #
    pass
    #
    return kwargs_arguments_classifier, net


def extract_dataset_loader_kwargs_from_experiment_specification(dict_individual_experiment_specification,
                                                                error_if_not_present=True):
    """
    It returns the parameters regarding the dataset contained in the `dict_individual_experiment_specification` \
    formatted already in the kwarg structure which the function \
    :py:func:`.experimental_evaluation.operations_for_datasets.obtain_classification_dataset_loaders` \
    would use.

    If an expected variable or parameter is not present in `dict_individual_experiment_specification`, \
    it raises an exception if `error_if_not_present` is `True`, or simply ignores it if `error_if_not_present` is \
    `False`.

    Parameters
    ----------
    dict_individual_experiment_specification : dict
    error_if_not_present : bool, optional
        If `True`, it raises an exception if the classifier specification is not present in \
        `dict_individual_experiment_specification['classifier']`. If `False`, it returns `None` in that case. \
        Default: `True`

    Returns
    -------
    dict
        The kwargs, apart from the classifier and the loaded dataset, relevant to call the function \
        :py:func:`.experimental_evaluation.operations_for_datasets.obtain_classification_dataset_loaders`.
    """

    kwargs_dataset_loader = {}

    ######################################
    # First: [dataset] in the dictionary?
    ######################################

    dict_dataset = None
    if 'dataset' not in dict_individual_experiment_specification:
        if error_if_not_present:
            raise Exception(
                f"Expected {'dataset'} in 'dict_individual_experiment_specification', but it is not present!")
        return None
    else:
        dict_dataset = dict_individual_experiment_specification['dataset']
    pass

    ######################################
    # Fields in 'dict_experiment_specification['dataset']:
    #   name, colorspace, force_im_size, train_proportion, val_proportion, batch_size, generator_seed
    ######################################

    dict_name_in_config_to_name_as_kwarg = {
        'name': 'dataset_name',
        'colorspace': 'loaded_im_colorspace',
        'force_im_size': 'desired_im_size',
        'train_proportion': 'train_proportion',
        'mislabeled_proportion': 'mislabeled_proportion',
        'val_proportion': 'val_proportion',
        'batch_size': 'batch_size',
        'num_workers': 'num_workers',
        'generator_seed': 'generator_seed',
        'root_folder': 'root_folder'
    }

    if dict_dataset is not None:
        if not isinstance(dict_dataset, dict):
            raise Exception(f"Expected 'dict_individual_experiment_specification['dataset'] to be a dictionary, " +
                            f"but it is {type(dict_dataset)}!")
        else:
            for key in dict_name_in_config_to_name_as_kwarg:
                if key not in dict_dataset:
                    if key in ['mislabeled_proportion', 'num_workers']:
                        kwargs_dataset_loader[key] = None
                    elif error_if_not_present:
                        raise Exception(f"Expected {key} in 'dict_individual_experiment_specification['dataset'], " +
                                        f"but it is not present!")
                    pass
                else:
                    kwargs_dataset_loader[dict_name_in_config_to_name_as_kwarg[key]] = dict_dataset[key]
                pass
            pass
        pass
    pass

    # Finally, the keyword fields 'desired_im_size' and 'generator_seed', if False in the configuration, are set to None
    for key in ['desired_im_size', 'generator_seed', 'num_workers']:
        if key in kwargs_dataset_loader:
            kwargs_dataset_loader[key] = value_if_bool_false(kwargs_dataset_loader[key], None)
        pass
    pass

    return kwargs_dataset_loader


def extract_training_kwargs_from_experiment_specification(dict_individual_experiment_specification,
                                                          error_if_not_present=True):
    """
    It returns the optimization and logging parameters for the training of a classifier as specified in the dictionary \
    `dict_individual_experiment_specification`.

    If an expected variable or parameter is not present in `dict_individual_experiment_specification['classifier']`, \
    it raises an exception if `error_if_not_present` is `True`, or simply ignores it if `error_if_not_present` is \
    `False`.

    Parameters
    ----------
    dict_individual_experiment_specification : dict
    error_if_not_present : bool, optional
        If `True`, it raises an exception if the classifier specification is not present in \
        `dict_individual_experiment_specification['classifier']`. If `False`, it returns `None` in that case. \
        Default: `True`

    Returns
    -------
    dict
        The kwargs, apart from the classifier and the loaded dataset, relevant to call the function \
        :py:func:`.classifier_training`.
    """

    kwargs_train_optim_scheduler_adv = {}

    ######################################
    # Training-optimization parameters
    ######################################
    # Fields in 'dict_individual_experiment_specification['training']:
    #   maximum_epochs, validations_per_epoch, early_stop_epochs, epochs_sm_based_warmup
    # Fields in 'dict_individual_experiment_specification['training']['optimizer']:
    #   type, initial_lr, arguments
    # Fields in 'dict_individual_experiment_specification['training']['scheduler']:
    #   type = false
    #     arguments = {}
    # Fields in 'dict_individual_experiment_specification['training']['adversarial']:
    #  type, arguments
    # Fields in 'dict_individual_experiment_specification['training']['logging']:
    #   local_log_folder
    ######################################

    ######################################
    # Is there a [training] block?
    ######################################

    dict_training = None
    if 'training' not in dict_individual_experiment_specification:
        if error_if_not_present:
            raise Exception(
                f"Expected {'training'} in 'dict_individual_experiment_specification', but it is not present!")
        pass
    else:
        dict_training = dict_individual_experiment_specification['training']
    pass

    ######################################
    # Explore the [training] block
    ######################################

    if dict_training is not None:

        ######################################
        # Highest level keys
        ######################################

        dict_keys_and_types = {
            'maximum_epochs': int,
            'loss_function': str,
            'validations_per_epoch': int,
            'validation_on_test_subset': bool,
            'early_stop_epochs': int,
            'epochs_sm_based_warmup': int,
            'min_acc_threshold': float,
            'max_number_of_retries': int,
        }
        for key in dict_keys_and_types:
            if key not in dict_training:
                if key in ['epochs_sm_based_warmup', 'min_acc_threshold', 'max_number_of_retries']:
                    kwargs_train_optim_scheduler_adv[key] = None
                elif error_if_not_present:
                    raise Exception(f"Expected {key} in 'dict_individual_experiment_specification['training'], " +
                                    f"but it is not present!")
                pass
            else:
                if dict_keys_and_types[key] is not bool:
                    kwargs_train_optim_scheduler_adv[key] = value_if_bool_false(dict_training[key], None)
                else:
                    kwargs_train_optim_scheduler_adv[key] = dict_training[key]
                pass
            pass
            if (kwargs_train_optim_scheduler_adv[key] is not None) and \
                    (not isinstance(kwargs_train_optim_scheduler_adv[key], dict_keys_and_types[key])):
                raise Exception(f"Expected {key} in 'dict_individual_experiment_specification['training'] " +
                                f"to be of type {dict_keys_and_types[key]}, but it is {type(dict_training[key])}!")
            pass
        pass

        # Actually: the kwarg corresponding to 'loss_function' is actually not the "text" label for it but the class \
        # as expressed in the dictionary `_dict_loss_functions`: resolve!
        if 'loss_function' in kwargs_train_optim_scheduler_adv and \
                isinstance(kwargs_train_optim_scheduler_adv['loss_function'], str):
            kwargs_train_optim_scheduler_adv['loss_function'] = \
                _dict_loss_functions[kwargs_train_optim_scheduler_adv['loss_function']]
        else:
            raise Exception(f"Expected 'loss_function' in 'dict_individual_experiment_specification['training'], " +
                            f"to be a string, but it is {type(dict_training['loss_function'])}!")
        pass

        ######################################
        # Explore the [training.optimizer] subblock
        ######################################

        dict_optimizer = None
        if 'optimizer' not in dict_training:
            if error_if_not_present:
                raise Exception(
                    f"Expected [training][optimizer] in 'dict_individual_experiment_specification', but it is not present!")
            pass
        else:
            dict_optimizer = dict_training['optimizer']
        pass

        if dict_optimizer is not None:
            #
            if 'type' not in dict_optimizer:
                if error_if_not_present:
                    raise Exception(
                        f"Expected [training][optimizer][type] in 'dict_individual_experiment_specification', " +
                        f"but it is not present!")
                pass
            else:
                optimizer_type = dict_optimizer['type']
                kwargs_train_optim_scheduler_adv['optimizer_class'] = _dict_optimizer_classes[optimizer_type]
            pass
            #
            # Regarding the fields 'initial_lr' and 'arguments': 'kwargs_train_optim_scheduler' will contain
            # a field called 'optimizer_arg_dict' which will contain the arguments for the optimizer, which includes
            # the initial learning rate under the keyword 'lr' (since this is what the optimizer classes expect): therefore
            # we will accumulate the content of the field 'arguments' and the content of the field 'initial_lr' in the
            # same dictionary
            retrieved_dictionary_arguments = {}
            if 'arguments' not in dict_optimizer:
                if error_if_not_present:
                    raise Exception(
                        f"Expected [training][optimizer][initial_lr] in 'dict_individual_experiment_specification', " +
                        f"but it is not present!")
                pass
            else:
                read_optimizer_arguments = value_if_bool_false(dict_optimizer['arguments'], None)
                if isinstance(read_optimizer_arguments, dict):
                    for key in dict_optimizer['arguments']:
                        retrieved_dictionary_arguments[key] = dict_optimizer['arguments'][key]
                    pass
                elif read_optimizer_arguments is None:
                    pass
                else:
                    raise Exception(
                        f"Expected [training][optimizer][arguments] in 'dict_individual_experiment_specification', " +
                        f"to be a dictionary, None, or False, but it is {type(dict_optimizer['arguments'])}!")
                pass
            pass
            if 'initial_lr' not in dict_optimizer:
                if error_if_not_present:
                    raise Exception(
                        f"Expected [training][optimizer][initial_lr] in 'dict_individual_experiment_specification', " +
                        f"but it is not present!")
                pass
            else:
                retrieved_dictionary_arguments['lr'] = dict_optimizer['initial_lr']
            pass
            kwargs_train_optim_scheduler_adv['optimizer_arg_dict'] = retrieved_dictionary_arguments
            #
        pass  # END OF: if dict_optimizer is not None

        ######################################
        # Explore the [training.scheduler] subblock
        ######################################

        dict_scheduler = None
        if 'scheduler' not in dict_training:
            if error_if_not_present:
                raise Exception(
                    f"Expected [training][scheduler] in 'dict_individual_experiment_specification', but it is not present!")
            pass
        else:
            dict_scheduler = dict_training['scheduler']
        pass

        if dict_scheduler is not None:
            #
            if 'type' not in dict_scheduler:
                if error_if_not_present:
                    raise Exception(
                        f"Expected [training][scheduler][type] in 'dict_individual_experiment_specification', " +
                        f"but it is not present!")
                pass
            else:
                scheduler_type = value_if_bool_false(dict_scheduler['type'], None)
                kwargs_train_optim_scheduler_adv['scheduler_class'] = None if scheduler_type is None \
                    else _dict_scheduler_classes[scheduler_type]
            pass
            #
            if 'arguments' not in dict_scheduler:
                if 'scheduler_class' in kwargs_train_optim_scheduler_adv and \
                        kwargs_train_optim_scheduler_adv['scheduler_class'] is None:
                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict'] = {}
                elif error_if_not_present:
                    raise Exception(
                        f"Expected [training][scheduler][arguments] in 'dict_individual_experiment_specification', " +
                        f"but it is not present!")
                pass
            else:
                scheduler_arg_dict = {} \
                    if 'scheduler_class' in kwargs_train_optim_scheduler_adv and \
                        kwargs_train_optim_scheduler_adv['scheduler_class'] is None \
                    else value_if_bool_false(dict_scheduler['arguments'], None)
                kwargs_train_optim_scheduler_adv['scheduler_arg_dict'] = {}
                if scheduler_arg_dict is None:
                    pass
                elif isinstance(scheduler_arg_dict, dict):
                    if len(scheduler_arg_dict) == 0:
                        pass
                    else:
                        if scheduler_type in ['linear_warmup', 'LinearLR']:
                            # For this scheduler we give the chance to use the arg name 'factor' instead of the arg \
                            # name 'start_factor' of the Pytorch function to make it analogous to the ConstanLR/'warmup'
                            if 'factor' in scheduler_arg_dict:
                                if 'start_factor' in scheduler_arg_dict:
                                    raise Exception(
                                        f"Expected either 'factor' or 'start_factor' in the arguments for the " +
                                        f"LinearLR scheduler, but not both: however both are present!")
                                pass
                                scheduler_arg_dict['start_factor'] = scheduler_arg_dict.pop('factor')
                            pass
                            # To the dictionary of kwargs!
                            for key in scheduler_arg_dict:
                                kwargs_train_optim_scheduler_adv['scheduler_arg_dict'][key] = scheduler_arg_dict[key]
                            pass
                        elif scheduler_type == 'CyclicLR':    # For this scheduler we might need to "translate" parameter names
                            list_original_names_args_cycliclr = ['base_lr', 'max_lr', 'step_size_up', 'step_size_down',
                                                                 'mode', 'gamma', 'scale_fn', 'scale_mode', 'cycle_momentum',
                                                                 'base_momentum', 'max_momentum', 'last_epoch']
                            if all([elem in list_original_names_args_cycliclr for elem in scheduler_arg_dict]):
                                # No translation
                                for key in scheduler_arg_dict:
                                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict'][key] = scheduler_arg_dict[key]
                                pass
                            else:
                                print(f"Detected non-Pytorch arguments for CyclicLR scheduler: {scheduler_arg_dict}")
                                # Translation: some params are "taken" or adapted from the optimizer
                                if not (('low_lr' in scheduler_arg_dict) and \
                                        (('num_iters_cycle' in scheduler_arg_dict) ^ ('num_cycles' in scheduler_arg_dict)) \
                                        ):
                                    raise Exception(
                                        f"Expected 'low_lr' and either 'num_iters_cycle'/'num_cycles' (but not both!) " +
                                        f"in the arguments for the " +
                                        f"CyclicLR scheduler under non-Pytorch format: however {scheduler_arg_dict.keys()} found!")
                                else:
                                    for elem in scheduler_arg_dict:
                                        if elem in list_original_names_args_cycliclr:
                                            raise Exception(
                                                f"Expected 'low_lr' and 'num_iters_cycle' in the arguments for the " +
                                                f"CyclicLR scheduler under non-Pytorch format, but '{elem}', belonging " +
                                                f"to the args of the Pytorch CyclicLR, is also present!")
                                        pass
                                    pass
                                    #
                                    cycle_length = scheduler_arg_dict['num_iters_cycle'] \
                                        if 'num_iters_cycle' in scheduler_arg_dict \
                                        else int(round(
                                            float(kwargs_train_optim_scheduler_adv['maximum_epochs']) / \
                                            float(scheduler_arg_dict['num_cycles'])
                                        ))
                                    #
                                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['base_lr'] = \
                                        scheduler_arg_dict['low_lr']
                                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['max_lr'] = \
                                        kwargs_train_optim_scheduler_adv['optimizer_arg_dict']['lr']
                                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['step_size_up'] = \
                                        cycle_length // 2
                                    kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['step_size_down'] = \
                                        cycle_length - \
                                        kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['step_size_up']
                                    if 'momentum' in kwargs_train_optim_scheduler_adv['optimizer_arg_dict']:
                                        kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['base_momentum'] = \
                                            kwargs_train_optim_scheduler_adv['optimizer_arg_dict']['momentum']
                                        kwargs_train_optim_scheduler_adv['scheduler_arg_dict']['max_momentum'] = \
                                            kwargs_train_optim_scheduler_adv['optimizer_arg_dict']['momentum']
                                    pass
                                pass
                            pass
                            # end of: if scheduler_type == 'CyclicLR'
                        else:
                            for key in scheduler_arg_dict:
                                kwargs_train_optim_scheduler_adv['scheduler_arg_dict'][key] = scheduler_arg_dict[key]
                            pass
                        pass
                    pass
                    # end of: if isinstance(scheduler_arg_dict, dict) and len(scheduler_arg_dict) > 0
                else:
                    raise Exception(
                        f"Expected [training][scheduler][arguments] in 'dict_individual_experiment_specification', " +
                        f"to be a dictionary, None, or False, but it is {type(dict_scheduler['arguments'])}!")
                pass
            pass
            #
        pass  # END OF: if dict_scheduler is not None
    pass

    ######################################
    # Explore the [training.adversarial] subblock
    ######################################
    if 'adversarial' not in dict_training:
        dict_adversarial = None
        # kwargs_train_optim_scheduler_adv['adversarial_type'] = None
        # kwargs_train_optim_scheduler_adv['adversarial_arg_dict'] = None
    else:
        dict_adversarial = dict_training['adversarial']
    pass
    if dict_adversarial is not None:
        if 'type' not in dict_adversarial:
            if error_if_not_present:
                raise Exception(
                    f"Expected [training][adversarial][type] in 'dict_individual_experiment_specification', " +
                    f"but it is not present!")
            pass
        else:
            adversarial_type = value_if_bool_false(dict_adversarial['type'], None)
            kwargs_train_optim_scheduler_adv['adversarial_type'] = None if adversarial_type is None \
                else adversarial_type
        pass
        #
        if 'proportion' not in dict_adversarial:
            if error_if_not_present:
                raise Exception(
                    f"Expected [training][adversarial][proportion] in 'dict_individual_experiment_specification', " +
                    f"but it is not present!")
            pass
        else:
            proportion = value_if_bool_false(dict_adversarial['proportion'], 1.0)
            kwargs_train_optim_scheduler_adv['adversarial_proportion'] = proportion
        pass
        #
        if 'arguments' not in dict_adversarial:
            if error_if_not_present:
                raise Exception(
                    f"Expected [training][adversarial][arguments] in 'dict_individual_experiment_specification', " +
                    f"but it is not present!")
            pass
        else:
            adversarial_arg_dict = value_if_bool_false(dict_adversarial['arguments'], None)
            if isinstance(adversarial_arg_dict, dict) and len(adversarial_arg_dict) > 0:
                kwargs_train_optim_scheduler_adv['adversarial_arg_dict'] = {}
                for key in adversarial_arg_dict:
                    kwargs_train_optim_scheduler_adv['adversarial_arg_dict'][key] = adversarial_arg_dict[key]
                pass
            elif adversarial_arg_dict is None:
                pass
            else:
                raise Exception(
                    f"Expected [training][adversarial][arguments] in 'dict_individual_experiment_specification', " +
                    f"to be a dictionary, None, or False, but it is {type(dict_adversarial['arguments'])}!")
            pass
        pass
        #
    ######################################
    # Return the processed parameters
    ######################################

    return kwargs_train_optim_scheduler_adv


def extract_aa_args_from_experiment_specification(dict_individual_experiment_specification):
    """
    It returns a refined version of the parameters regarding adversarial attacks contained in the \
    `dict_individual_experiment_specification`.

    The fields relevant to the adversarial attacks follow the format described in the \
    file `CURRENT_FORMATS.md <../../../CURRENT_FORMATS.md>`_ and refer to the blocks ``[attack]``, ``[params_to_filter]``, \
    ``[metrics_to_filter]`` and their respetive sub-blocks.

    Parameters
    ----------
    dict_individual_experiment_specification : dict

    Returns
    -------
    kwargs_adversarial_attack : dict
        Dictionary containing the clean parameters regarding adversarial attacks regarding 'type', \
        'loss_function' (resolved already), 'validation_on_test_subset' and 'attack_params'. \
        The latter is a dictionary \
        containing the parameters of the attack
    generate_using_sm : bool
        If `True`, it indicates that the adversarial examples should be generated using the \
        `torchattacks` library, otherwise it is `False`
    num_workers : int or None
        The number of workers to use for the dataset loading during the adversarial attack evaluation
    """

    #######################################################
    # Extract the parameters in [attacks] and store them in the format of the args of the function
    # `assess_classifier_for_white_box_aa_sm_with_torchattacks`
    # (which are:
    #   classifier_nn, dataset_dict, loss_function, attack_type, attack_params, classifier_sm,
    #   validation_on_test_subset=True, subbatch_size=None, mlflow_run_id=None, run_name=None, verbose='medium')
    #######################################################
    # It contains: [attacks].type, [attacks].loss, [attacks].generate_using_sm, [attacks].validation_on_test_subset
    # and then [attacks][parameters]
    #######################################################

    assert isinstance(dict_individual_experiment_specification, dict), \
        f"Expected 'dict_individual_experiment_specification' to be a dict, but received {type(dict_individual_experiment_specification)}!"

    kwargs_adversarial_attack = {}
    generate_using_sm = None
    validation_on_test_subset = None
    num_workers = None
    mlflow_logging = None
    local_log_folder = None
    #
    #######################################################
    dict_attacks = None
    if 'attack' in dict_individual_experiment_specification and \
            isinstance(dict_individual_experiment_specification['attack'], dict):
        dict_attacks = dict_individual_experiment_specification['attack']
    else:
        raise Exception(f"Expected 'attack' in 'dict_individual_experiment_specification', " +
                        f"but it is not present or is not a dictionary!")
    pass
    #######################################################

    # Type of attack
    if 'type' not in dict_attacks:
        raise Exception(f"Expected 'type' in 'dict_individual_experiment_specification['attacks']', " +
                        f"but it is not present!")
    elif not isinstance(dict_attacks['type'], str):
        raise Exception(f"Expected 'type' in 'dict_individual_experiment_specification['attacks']' to be a string, " +
                        f"but it is {type(dict_attacks['type'])}!")
    elif dict_attacks['type'].lower() not in ['fgsm', "onepixel", "gn", "deepfool", "pgd", "pixle"]:
        raise Exception(f"Expected 'type' in 'dict_individual_experiment_specification['attacks']' to be one of " +
                        f"'fgsm', 'onepixel', 'gn', 'pgd', 'pixle' or 'deepfool', but it is {dict_attacks['type']}!")
    else:
        kwargs_adversarial_attack['attack_type'] = dict_attacks['type']
    pass

    # Loss function
    loss_function = None
    dict_loss_name_to_loss_func = {'ce': torch.nn.functional.cross_entropy}
    if 'loss' not in dict_attacks:
        raise Exception(f"Expected 'loss' in 'dict_individual_experiment_specification['attacks']', " +
                        f"but it is not present!")
    elif not isinstance(dict_attacks['loss'], str):
        raise Exception(f"Expected 'loss' in 'dict_individual_experiment_specification['attacks']' to be a string, " +
                        f"but it is {type(dict_attacks['loss'])}!")
    elif dict_attacks['loss'].lower() not in ['ce']:
        raise Exception(f"Expected 'loss' in 'dict_individual_experiment_specification['attacks']' to be one of " +
                        f"'ce', 'dlr', but it is {dict_attacks['loss']}!")
    else:
        kwargs_adversarial_attack['loss_function'] = dict_loss_name_to_loss_func[dict_attacks['loss'].lower()]
    pass

    # Number of workers of the loaded dataset (if we want to override the value read from the dataset of the load model)
    num_workers = None
    if 'num_workers' not in dict_attacks:
        num_workers = None
    else:
        num_workers = value_if_bool_false(dict_attacks['num_workers'], None)
        if num_workers is not None:
            assert isinstance(num_workers, int) and num_workers >= 0, \
                f"Expected 'num_workers' in 'dict_individual_experiment_specification['attacks']' to be False " + \
                f"or a non-negative integer, but it is {dict_attacks['num_workers']}!"
    pass

    # 'validation_on_test_subset' field
    if 'validation_on_test_subset' not in dict_attacks:
        raise Exception(
            f"Expected 'validation_on_test_subset' in 'dict_individual_experiment_specification['attacks']', " +
            f"but it is not present!")
    elif not isinstance(dict_attacks['validation_on_test_subset'], bool):
        raise Exception(
            f"Expected 'validation_on_test_subset' in 'dict_individual_experiment_specification['attacks']' " +
            f"to be a boolean, but it is {type(dict_attacks['validation_on_test_subset'])}!")
    else:
        kwargs_adversarial_attack['validation_on_test_subset'] = dict_attacks['validation_on_test_subset']
    pass

    # NOTE: generate_using_sm does not go into kwargs_adversarial_attack because it is not used in the function \
    # `assess_classifier_for_white_box_aa_sm_with_torchattacks`
    generate_using_sm = None
    if 'generate_using_sm' not in dict_attacks:
        raise Exception(f"Expected 'generate_using_sm' in 'dict_individual_experiment_specification['attacks']', " +
                        f"but it is not present!")
    elif not isinstance(dict_attacks['generate_using_sm'], bool):
        raise Exception(f"Expected 'generate_using_sm' in 'dict_individual_experiment_specification['attacks']' " +
                        f"to be a boolean, but it is {type(dict_attacks['generate_using_sm'])}!")
    else:
        generate_using_sm = dict_attacks['generate_using_sm']
    pass

    #
    #######################################################
    dict_params_attacks = None
    if 'parameters' in dict_attacks and isinstance(dict_attacks['parameters'], dict):
        dict_params_attacks = dict_attacks['parameters']
    else:
        raise Exception(f"Expected 'parameters' in 'dict_individual_experiment_specification['attacks']', " +
                        f"but it is not present or is not a dictionary!")
    pass
    #######################################################
    #

    # Get its parameters
    kwargs_adversarial_attack['attack_params'] = {}

    list_valid_parameters = None
    if 'parameters' not in dict_attacks:
        raise Exception(f"Expected 'parameters' in 'dict_individual_experiment_specification['attacks']', " +
                        f"but it is not present!")
    else:
        if kwargs_adversarial_attack['attack_type'].lower() == 'fgsm':
            list_valid_parameters = ['epsilon']
        elif kwargs_adversarial_attack['attack_type'].lower() == 'onepixel':
            list_valid_parameters = ['pixel_count', 'max_iter', 'popsize']
        elif kwargs_adversarial_attack['attack_type'].lower() == 'pgd':
            list_valid_parameters = ['epsilon', 'alpha', 'num_iter']
        elif kwargs_adversarial_attack['attack_type'].lower() == 'pixle':
            list_valid_parameters = ['x_dimensions', 'y_dimensions', 'pixel_mapping', 'restarts', 'max_iter',
                                    'update_each_iteration']
        elif kwargs_adversarial_attack['attack_type'].lower() == 'gn':
            list_valid_parameters = ['mean', 'std']
        elif kwargs_adversarial_attack['attack_type'].lower() == 'deepfool':
            list_valid_parameters = ['overshoot', 'max_iter']
        else:
            raise Exception(f"Expected 'type' in 'dict_individual_experiment_specification['attacks']' to be one of " +
                            f"'fgsm', 'onepixel', 'gn', 'pgd', 'pixle' or 'deepfool', but it is {dict_attacks['type']}!")
        pass
    pass

    for key in list_valid_parameters:
        if key in dict_attacks['parameters']:
            kwargs_adversarial_attack['attack_params'][key] = dict_attacks['parameters'][key]
        else:
            raise Exception(
                f"Expected '{key}' in 'dict_individual_experiment_specification['attacks']['parameters']', " +
                f"but it is not present!")
        pass
    pass

    #######################################################

    return kwargs_adversarial_attack, generate_using_sm, num_workers


def extract_filtering_args_from_experiment_specification(dict_individual_experiment_specification):
    """
    It returns a refined version of the parameters regarding filtering MLFlow experiments, both \
    based on logged params and logged metrics.

    Parameters
    ----------
    dict_individual_experiment_specification : dict

    Returns
    -------
    params_to_filter : dict
    metrics_to_filter : dict
    """

    params_to_filter = {}
    metrics_to_filter = {}

    #######################################################
    # The input dictionary can contain:
    # [params_to_filter] with, possibly, several fields, with one single textual parameter each
    # [metrics_to_filter] with, possibly, several fields, with a vector of 2 values each
    #######################################################

    assert isinstance(dict_individual_experiment_specification, dict), \
        f"Expected 'dict_individual_experiment_specification' to be a dict, " + \
        f"but received {type(dict_individual_experiment_specification)}!"

    # Params!

    dict_params_to_filter = dict_individual_experiment_specification.get('params_to_filter', None)
    if dict_params_to_filter is not None:
        if not isinstance(dict_params_to_filter, dict):
            raise Exception(f"Expected 'params_to_filter' in 'dict_individual_experiment_specification', " +
                            f"to be a dictionary, but it is {type(dict_params_to_filter)}!")
        else:
            for key in dict_params_to_filter:
                if not isinstance(dict_params_to_filter[key], str):
                    raise Exception(f"Expected 'params_to_filter' in 'dict_individual_experiment_specification' " +
                                    f"to contain only strings, but the key '{key}' contains {type(dict_params_to_filter[key])}!")
                elif key == "net":
                    params_to_filter[key] = _dict_classifiers_as_in_conf_file[dict_params_to_filter[key]].__name__
                else:
                    params_to_filter[key] = dict_params_to_filter[key]
                pass
            pass
        pass
    pass    # end of: if dict_params_to_filter is not None ...

    # Metrics!

    # Since the filtering function substitutes false=None for +/- inf, depending on whether the value is \
    # a lower or upper bound, we need to make sure that the values are correctly interpreted

    dict_metrics_to_filter = dict_individual_experiment_specification.get('metrics_to_filter', None)
    if dict_metrics_to_filter is not None:
        if not isinstance(dict_metrics_to_filter, dict):
            raise Exception(f"Expected 'metrics_to_filter' in 'dict_individual_experiment_specification', " +
                            f"to be a dictionary, but it is {type(dict_metrics_to_filter)}!")
        else:
            for key in dict_metrics_to_filter:
                if not isinstance(dict_metrics_to_filter[key], (tuple, list)) or \
                        len(dict_metrics_to_filter[key]) != 2:
                    raise Exception(
                        f"Expected 'metrics_to_filter' in 'dict_individual_experiment_specification' " +
                        f"to contain only tuples or lists of length 2, but the key '{key}' contains " +
                        f"{type(dict_metrics_to_filter[key])} with length {len(dict_metrics_to_filter[key])}!"
                    )
                else:
                    metrics_to_filter[key] = list(dict_metrics_to_filter[key])
                    # Substitute "false" or "None" with +/- inf, depending on whether it is
                    for i in range(2):
                        interpreted_value_i = value_if_bool_false(metrics_to_filter[key][i], None)
                        if interpreted_value_i is None:
                            metrics_to_filter[key][i] = -np.inf if i == 0 else np.inf
                        elif isinstance(interpreted_value_i, (int, float)):
                            pass
                        else:
                            raise Exception(f"Invalid value for the key '{key}': {dict_metrics_to_filter[key]}. " +
                                            f"Expected a number, 'inf', or '-inf'.")
                        pass
                    pass
                    metrics_to_filter[key] = tuple(metrics_to_filter[key])
                pass
            pass
        pass
    pass  # end of: if dict_metrics_to_filter is not None ...

    return params_to_filter, metrics_to_filter


def extract_retraining_kwargs_from_experiment_specification(dict_individual_experiment_specification,
                                                            error_if_not_present=True):
    """
    It returns a dictionary with the processed version of the parameters regarding \
    retraining of a classifier as specified in the dictionary `dict_individual_experiment_specification` \
    and more specifically in the field [retraining].

    So far, the only expected subfield is 'trainable_section', which is a string value indicating which section of the \
    classifier should be retrained, with the following possible values: "head", "backbone", "all".

    If an expected variable or parameter is not present in `dict_individual_experiment_specification`, \
    it raises an exception if `error_if_not_present` is `True`, or simply ignores it if `error_if_not_present` is \
    `False`.

    Parameters
    ----------
    dict_individual_experiment_specification : dict
    error_if_not_present : bool, optional
        If `True`, it raises an exception if the classifier specification is not present in \
        `dict_individual_experiment_specification['classifier']`. If `False`, it returns `None` in that case. \
        Default: `True`

    Returns
    -------
    dict
    """

    kwargs_retraining = None

    #######################################################
    # The input dictionary can contain [retraining] with a set for possible options
    #######################################################

    assert isinstance(dict_individual_experiment_specification, dict), \
        f"Expected 'dict_individual_experiment_specification' to be a dict, " + \
        f"but received {type(dict_individual_experiment_specification)}!"

    dict_retraining = dict_individual_experiment_specification.get('retraining', None)
    if dict_retraining is not None:
        if not isinstance(dict_retraining, dict):
            raise Exception(f"Expected 'retraining' in 'dict_individual_experiment_specification', " +
                            f"to be a dictionary, but it is {type(dict_retraining)}!")
        else:
            if 'trainable_section' not in dict_retraining:
                if error_if_not_present:
                    raise Exception(
                        f"Expected 'trainable_section' in 'dict_individual_experiment_specification['retraining'], " +
                        f"but it is not present!"
                    )
                pass
            else:
                list_of_valid_sections = ['head', 'backbone', 'all', 'bn+head']
                if not isinstance(dict_retraining['trainable_section'], str):
                    raise Exception(
                        f"Expected 'trainable_section' in 'dict_individual_experiment_specification['retraining']' " +
                        f"to be a string, but it is {type(dict_retraining['trainable_section'])}!")
                elif dict_retraining['trainable_section'] not in list_of_valid_sections:
                    raise Exception(
                        f"Expected 'trainable_section' in 'dict_individual_experiment_specification['retraining']' " +
                        f"to be one of {list_of_valid_sections}, but it is {dict_retraining['trainable_section']}!")
                else:
                    kwargs_retraining = {'trainable_section': dict_retraining['trainable_section']}
                pass
            pass
        pass
    pass    # end of: if dict_retraining is not None ...

    return kwargs_retraining




