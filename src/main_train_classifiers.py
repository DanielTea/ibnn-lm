#!/usr/bin/env python
# coding: utf-8

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

import tomli
import socket
import datetime
import os
import warnings

import traceback

import argparse
import pathlib
import itertools
import json

import copy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

from applications import _dict_classifiers_as_in_conf_file

from experimental_evaluation import _dict_optimizer_classes, _dict_scheduler_classes, _dict_loss_functions

from experimental_evaluation.interaction_with_mlflow import connect_to_mlflow
from experimental_evaluation.experiment_utils import (
    main_entry_point_parser_to_experiment_function,
    run_experiment_group_from_toml_file,
    classifier_training,
    formatted_log_base_name,
    exception_display_and_log
)

from experimental_evaluation.configuration_file_reading_utils import (
    get_keypaths_exisiting_dict_elements,
    get_multilevel_dict_element,
    value_if_bool_false,
    extract_classifier_kwargs_from_experiment_specification,
    extract_dataset_loader_kwargs_from_experiment_specification,
    extract_training_kwargs_from_experiment_specification
)

from experimental_evaluation.operations_for_datasets import obtain_classification_dataset_loaders

from modified_rf.nn_layers import ModifiedRFLayer



##############################################################################################################
##############################################################################################################
##############################################################################################################



####################################################################################
# SPECIFIC FUNCTIONS FOR THE CLASSIFIER TRAINING EXPERIMENTS
####################################################################################


def run_classifier_training_experiment_group_from_toml_file(experiment_group_specification_file,
                                                            mlflow_logging="vmg", verbose='medium'):
    """
    Instance of the function :py:func:`experimental_evaluation.run_experiment_group_from_toml_file` for \
    running a group of classifier training experiments. The function \
    :py:func:`experimental_evaluation.run_experiment_group_from_toml_file` is called with the following \
    composing functions:

    - :py:func:`run_classifier_training_experiment` as `function_run_individual_experiment_from_dict`
    - :py:func:`deactivate_irrelevant_parameters_classifier_training_experiment` as \
        `function_deactivate_irrelevant_parameters_in_dict`

    Parameters
    ----------
    experiment_group_specification_file : str
        Path to the TOML file containing the group experiment specification
    mlflow_logging : str, optional
        URL of the MLflow tracking server to be used for logging.
        Default: ``"vmg"``, which loads the system credentials and addresses
    verbose : str, optional
        Default: ``'medium'``
    """

    ######################################
    # PROCESS AND EXECUTE THE GROUP FILE
    ######################################
    run_experiment_group_from_toml_file(
        experiment_group_specification_file,
        function_run_individual_experiment_from_dict=run_classifier_training_experiment,
        function_deactivate_irrelevant_parameters_in_dict=deactivate_irrelevant_parameters_classifier_training_experiment,
        verbose=verbose
    )


def run_classifier_training_experiment(
        dict_individual_experiment_specification, dict_general_info_group_experiment=None, verbose='medium'
):
    """
    Function controlling the training of a classifier from a dictionary, given in the \
    `dict_experiment_specification` argument, which contains the specification of the experiment.

    Parameters
    ----------
    dict_individual_experiment_specification : dict
    dict_general_info_group_experiment : dict, optional
        Dictionary containing the general information of the group experiment, namely `'experiment_name'` (str), \
        `'purpose'` (str), `'base_experiment_specification_file'` (str, a path to a file), \
        `'mlflow_logging'` (bool), and `'local_log_folder'` (str a path to a folder or `None` if not provided).
        Default: ``None``, which would result in an unnamed experiment with no logging
    verbose : str, optional
        Default: ``'medium'``

    Returns
    -------
    result_classifier_training : named tuple returned by :py:func:`~experimental_evaluation.experiment_utils.classifier_training`
    dataset_loader_dict : :py:class:`~experimental_evaluation.operations_for_datasets.LoadedDatasetDict`
    """

    ######################################
    # Logging flags and folders: taken from the argument `dict_general_info_group_experiment`
    ######################################

    if dict_general_info_group_experiment is None:
        dict_general_info_group_experiment = {}
    pass

    if not isinstance(dict_general_info_group_experiment, dict):
        raise Exception(
            f"Invalid type for 'dict_general_info_group_experiment': " +
            f"expected dict, received {type(dict_general_info_group_experiment)}")
    pass

    experiment_name = dict_general_info_group_experiment.get('experiment_name', 'unnamed')
    purpose = dict_general_info_group_experiment.get('purpose', None)
    base_experiment_specification_file = \
        dict_general_info_group_experiment.get('base_experiment_specification_file', None)

    mlflow_logging = dict_general_info_group_experiment.get('mlflow_logging', False)
    local_log_folder = \
        value_if_bool_false(dict_general_info_group_experiment.get('local_log_folder', None), None)

    ##############################################################################
    # "Declare" the variables which will be returned
    ##############################################################################

    result_classifier_training = None
    dataset_loader_dict = None

    ##############################################################################
    # Read all parameters (once) and load everything but the classifier itself
    ##############################################################################

    try:
        #######################################
        # Dataset loading
        #######################################

        kwargs_dataset_loader = extract_dataset_loader_kwargs_from_experiment_specification(
            dict_individual_experiment_specification)
        kwargs_dataset_loader['verbose'] = verbose

        dataset_loader_dict = obtain_classification_dataset_loaders(**kwargs_dataset_loader)
        print(
            f"Dataset {dataset_loader_dict['dataset_name']} {dict_individual_experiment_specification['dataset']['colorspace']} " +
            f"(" +
                f"B={dataset_loader_dict['batch_size']}, " +
                f"(H={dataset_loader_dict['im_height']},W={dataset_loader_dict['im_width']}), " +
                f"num_workers={dataset_loader_dict['num_workers']}" +
            f") created successfully!")

        # # FIX FOR DEBUGGING
        # dataset_loader_dict = {
        #     'im_height': 32, 'im_width': 32, 'channels': 3, 'classes': 10,
        # }

        #######################################
        # Classifier kwarg reading
        #######################################

        in_size = (dataset_loader_dict['im_height'], dataset_loader_dict['im_width'])
        kwargs_arguments_classifier, net = extract_classifier_kwargs_from_experiment_specification(
            dict_individual_experiment_specification,
            in_size=in_size, in_channels=dataset_loader_dict['channels'], out_classes=dataset_loader_dict['classes'],
            net=None, error_if_not_present=True
        )

        # classifier_nn = _create_classifier_from_experiment_group_from_dict(dict_individual_experiment_specification,
        #                                                                    dataset_loader_dict, verbose=verbose)
        #
        #######################################
        # Training-optimization parameters
        #######################################

        # kwargs_train_optim_scheduler = _load_optim_and_scheduler_parameters_from_experiment_group_from_dict(
        #     dict_individual_experiment_specification, verbose=verbose
        # )
        kwargs_train_optim_scheduler = extract_training_kwargs_from_experiment_specification(
            dict_individual_experiment_specification, error_if_not_present=True
        )
        min_acc_threshold = kwargs_train_optim_scheduler.pop('min_acc_threshold', None)
        max_number_of_retries = kwargs_train_optim_scheduler.pop('max_number_of_retries', None)
        if min_acc_threshold is not None:
            assert isinstance(min_acc_threshold, float) and 0.0 < min_acc_threshold < 1.0, \
                f"'min_acc_threshold', if provided, must be None or 0<float<1: {min_acc_threshold} found!"
        pass
        if max_number_of_retries is not None:
            assert isinstance(max_number_of_retries, int) and 0 < max_number_of_retries, \
                f"'max_number_of_retries', if provided, must be None or 0<int: {max_number_of_retries} found!"
        pass
        if (max_number_of_retries is None or max_number_of_retries == 1) and min_acc_threshold is not None:
            # This is equivalent to no retrying at all: so we set it so and give a warning!
            warnings.warn(f"Since 'max_number_of_retries'=1 has been provided, " + \
                          f"no effective re-attempts will be attempted!")
            min_acc_threshold = None
        pass
        #
        ############################################################################
        # Perform the training itself (repeating runs if so indicated)
        ############################################################################
        #
        effective_max_number_of_retries = max_number_of_retries if min_acc_threshold is not None else 1
        #
        for ind_attempt in range(effective_max_number_of_retries):

            ##############################################################################
            # Run the training, i-th attempt
            ##############################################################################

            if ind_attempt > 0:
                print(f"\n·············································")
                print((f"Re-attempt to training: {ind_attempt+1}-th attempt " +
                       f"(of max {effective_max_number_of_retries})!"))
                print(f"·············································\n")
            pass

            ##############################################################################
            # Creation of the run name
            ##############################################################################

            start_training = datetime.datetime.now()
            host = socket.gethostname()
            dataset_name = dict_individual_experiment_specification['dataset']['name']
            net = dict_individual_experiment_specification['classifier']['net']
            conv_like_type = dict_individual_experiment_specification['classifier']['conv_like_type']
            conv_like_type_position = dict_individual_experiment_specification['classifier']['conv_like_type_position']
            run_name = formatted_log_base_name(
                start_training, host=host, dataset_name=dataset_name, net_name=net,
                extra_field=f"{conv_like_type}_{conv_like_type_position}", flag_random_id=True
            )

            # Creation of the run name/run ID for MLFlow logging
            classifier_training_arg_run_name = None

            mlflow_run = None
            if mlflow_logging is None or mlflow_logging == False:
                #
                print(f"MLFlow logging not activated: local logging, if selected, under run name '{run_name}'")
                classifier_training_arg_run_name = run_name
                #
            else:
                #
                ######################################
                # SET THE TRACKING URI FOR MLFLOW
                ######################################
                if isinstance(mlflow_logging, bool) and mlflow_logging:
                    connect_to_mlflow()
                elif isinstance(mlflow_logging, str):
                    connect_to_mlflow(mlflow_logging)
                else:
                    raise Exception("'mlflow_logging' of the group config file must be None/false or bool or str: " + \
                                    f"{mlflow_logging} of type {type(mlflow_logging)} found!")
                pass
                #
                mlflow.set_experiment(experiment_name)
                mlflow_run = mlflow.start_run(run_name=run_name,
                                              experiment_id=mlflow.get_experiment_by_name(
                                                  experiment_name).experiment_id,
                                              log_system_metrics=True)
                print(f"MLFlow logging activated: experiment '{experiment_name}', run started with name '{run_name}'")
                #
            pass

            #####################################
            # Log the parameters of the specification "dict_individual_experiment_specification_i" and log them
            # (we do it because we might have deleted them when we repeated the experiment)
            ####################################
            if mlflow_run is not None:
                dict_list = get_keypaths_exisiting_dict_elements(dict_individual_experiment_specification)
                for item in dict_list:
                    value = get_multilevel_dict_element(dict_individual_experiment_specification, item)
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(item[-1], value, run_id=mlflow_run.info.run_id)
                        # print(f"\t***Logged:    {item[-1]} = {value}")
                    pass
                pass

            #####################################
            # (Re-)Create the classifier
            #####################################
            classifier_nn = _dict_classifiers_as_in_conf_file[net](**kwargs_arguments_classifier)
            classifier_nn.logging_compliance_checker()
            print(f"{_dict_classifiers_as_in_conf_file[net]} created successfully!")

            # Computation device
            computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            classifier_nn.to_device(computation_device)
            # computation_device = 'cpu'
            print(f"CUDA device{' ' if torch.cuda.is_available() else ' NOT '}available. " +
                  f"Classifier {type(classifier_nn).__name__} created on device '{classifier_nn.device}'.")

            mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else None
            result_classifier_training = classifier_training(
                classifier_nn, dataset_loader_dict, **kwargs_train_optim_scheduler,
                mlflow_run_id=mlflow_run_id, run_name=classifier_training_arg_run_name,
                local_log_folder=local_log_folder,
                verbose=verbose
            )

            ############################################################################
            # Log the current attempt (if it is not the last it will be deleted)
            # AND stop the logging of the run, if so indicated
            ############################################################################

            print(f"Finished training for the {ind_attempt+1}-th attempt!", flush=True)
            if mlflow_run is not None:
                mlflow.log_metric('attempt', ind_attempt+1)
                mlflow.end_run()
            pass
            print(f"Finished logging for the {ind_attempt+1}-th attempt!", flush=True)

            ##############################################################################
            # Check the result of the training, to see if re-training is needed
            ##############################################################################
            #
            # Evaluate the acc of the training and decide whether retraining is needed
            if min_acc_threshold is None or result_classifier_training.best_acc >= min_acc_threshold:
                break
            else:
                print(f"\n·············································")
                print((f"Training with insufficient acc: {100 * result_classifier_training.best_acc:.2f} % " +
                       f"(th={100 * min_acc_threshold:.2f})"))
                # Delete the run (if existing)
                if mlflow_logging is not None and mlflow_logging != False and \
                        ind_attempt < effective_max_number_of_retries-1:
                    print(f"Deletion of the unsuccessful run...", end="")
                    mlflow.delete_run(mlflow_run.info.run_id)
                    print(f" DONE!")
                pass
                print(f"·············································\n")
                # And stay in the "for" for another iteration
            pass
            #
        pass
        #
    except Exception as err:
        exception_display_and_log(err,
                                  dict_individual_experiment_specification=dict_individual_experiment_specification,
                                  mlflow_logging=mlflow_logging)
    pass

    ######################################
    # Check if the MLFlow run is still opened to close it (MLFlow gives problems with new runs if one is left open)
    ######################################
    if mlflow.active_run() is not None:
        mlflow.end_run()
    pass

    ######################################
    # Store the model
    ######################################

    return result_classifier_training, dataset_loader_dict


def deactivate_irrelevant_parameters_classifier_training_experiment(raw_dict_individual_experiment_specification):
    """
    This function takes a dictionary `dict_individual_experiment_specification` containing the specification of an \
    individual training experiment and deactivates the parameters which are not relevant for the specific experiment.

    Parameters
    ----------
    raw_dict_individual_experiment_specification : dict
        The dictionary containing the specification of an individual training experiment, before deactivation

    Returns
    -------
    processed_dict_individual_experiment_specification : dict
        The same dictionary, with the irrelevant parameters deactivated
    """

    processed_dict_individual_experiment_specification = copy.deepcopy(raw_dict_individual_experiment_specification)

    #########################################################################################
    # For the 'dataset' block
    #########################################################################################

    if 'dataset' in processed_dict_individual_experiment_specification:

        ######################
        # Eliminate unnecessary fields from the dictionary specification for the DATASET (e.g. bn datasets)
        ######################

        num_channels_from_dataset = 1 \
            if processed_dict_individual_experiment_specification['dataset']['colorspace'] == "gray" else 3

    pass  # END 'dataset' block

    #########################################################################################
    # For the 'classifier' block
    #########################################################################################

    if 'classifier' in processed_dict_individual_experiment_specification:

        ######################
        # Eliminate unnecessary fields related to the FC head
        ######################

        if processed_dict_individual_experiment_specification['classifier']['fully_connected']['fc_num_layers'] == 1:
            processed_dict_individual_experiment_specification['classifier']['fully_connected']['fc_num_units_intermediate_layers'] = -1
        pass

        ######################
        # Eliminate unnecessary fields from the dictionary specification for specific NETs
        ######################

        net = processed_dict_individual_experiment_specification['classifier']['net']
        conv_like_type = processed_dict_individual_experiment_specification['classifier']['conv_like_type']
        if net == 'single_layer' or conv_like_type == 'sm':
            processed_dict_individual_experiment_specification['classifier']['conv_like_type_position'] = 'everywhere'
        pass

        processed_dict_individual_experiment_specification['classifier']['architecture_specific'] = \
            {net: processed_dict_individual_experiment_specification['classifier']['architecture_specific'][net]}

        # If the architecture-specific '<field>_per_conv_layer' is provided then 'overall_<field>_conv_layers' is ignored
        architecture_specific_dict = \
            processed_dict_individual_experiment_specification['classifier']['architecture_specific'][net]
        for field in ['m_kernel_size', 'phi_activation', 'batch_normalization', 'maxpool_reduction']:
            field_per_conv_layer = f"{field}_per_conv_layer"
            overall_field_conv_layers = f"overall_{field}_conv_layers"
            if field_per_conv_layer in architecture_specific_dict \
                    and value_if_bool_false(architecture_specific_dict[field_per_conv_layer], None) is not None:
                processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['traditional'][overall_field_conv_layers] = None
            pass
        pass

        # Other checks regarding the architecture-specific dictionary
        if net == 'multi_layer':
            if architecture_specific_dict.get('num_conv_like_layers', None) == 1:
                processed_dict_individual_experiment_specification['classifier']['conv_like_type_position'] = 'everywhere'
        elif net == 'vggx':
            if architecture_specific_dict['base_channels'] == 1:
                processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias'][
                    'w_independent_channels'] = True
            pass
        elif net == 'alexnet':
            pass
        elif net == 'efficientnetv2s':
            # EfficientNetV2S does not use 'maxpool_reduction': it will always be 1
            processed_dict_individual_experiment_specification['classifier']['maxpool_reduction'] = 1
        pass

        ######################
        # Eliminate unnecessary fields from the dictionary specification for specific CONV_LIKE_TYPEs
        ######################

        if conv_like_type not in ["inrfv1", "inrfv2", "inrfv3", "ibnn_lite", "ibnn_internal", "ibnn"]:
            # In this case, the fields related to the nonlinear bias are not necessary
            processed_dict_individual_experiment_specification['classifier']['conv_like_layer'].pop('nonlinear_bias', None)
        elif conv_like_type not in ["inrfv2", "inrfv3", "ibnn_lite", "ibnn_internal", "ibnn"]:
            list_keys_starting_by_w = [
                key for key in
                processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias'] \
                if key.startswith('w_')
            ]
            for key in list_keys_starting_by_w:
                processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['nonlinear_bias'].pop(
                    key)
            pass
        pass
        #
        ### Regarding the 'fixed_point' field in the conv_like_layer
        #
        if conv_like_type not in ["ibnn_internal", "ibnn"]:
            processed_dict_individual_experiment_specification['classifier']['conv_like_layer'].pop('fixed_point', None)
        else:
            f_solver = processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['fixed_point'].get('f_solver', None)
            if f_solver == "broyden":
                processed_dict_individual_experiment_specification['classifier']['conv_like_layer']['fixed_point']['f_tau'] = None
            pass
        pass

    pass # END 'classifier' block

    #########################################################################################
    # For the 'training' block
    #########################################################################################

    if 'training' in processed_dict_individual_experiment_specification:

        ### Regarding the 'training' part, optimized, scheduler, and adversarial
        if 'training' in processed_dict_individual_experiment_specification:
            if 'scheduler' in processed_dict_individual_experiment_specification['training']:
                scheduler_type = \
                    processed_dict_individual_experiment_specification['training']['scheduler'].get('type', None)
                if scheduler_type is None or scheduler_type==False or scheduler_type=='none':
                    processed_dict_individual_experiment_specification['training']['scheduler']['type'] = None
                    processed_dict_individual_experiment_specification['training']['scheduler']['arguments'] = None
            if 'adversarial' in processed_dict_individual_experiment_specification['training']:
                adversarial_type = \
                    processed_dict_individual_experiment_specification['training']['adversarial'].get('type', None)
                if adversarial_type is None or adversarial_type == False or adversarial_type == 'none':
                    processed_dict_individual_experiment_specification['training']['adversarial']['type'] = None
                    processed_dict_individual_experiment_specification['training']['adversarial']['arguments'] = None
                    processed_dict_individual_experiment_specification['training']['adversarial']['proportion'] = None
                pass

    pass # END 'training' block


    return processed_dict_individual_experiment_specification


####################################################################################
# ENTRY POINT FOR COMMAND-LINE INTERFACING
####################################################################################

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description="argument parser")
    main_entry_point_parser_to_experiment_function(parser, run_classifier_training_experiment_group_from_toml_file)

pass
