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
import sys
import os
from contextlib import contextmanager
import tempfile
import warnings

import numpy as np
import pandas as pd

import argparse
import pathlib
import itertools
import json

import copy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

##############################################################################################################
import pynvml
##############################################################################################################
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
##############################################################################################################
from dotenv import load_dotenv
load_dotenv()
##############################################################################################################

from modified_rf.nn_layers import _dict_conv_like_layers, SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, \
    IBNNInternalLayer, IBNNLayer
from applications import _dict_classifiers, _dict_classifiers_as_in_conf_file
from experimental_evaluation.operations_for_datasets import (_dict_dataset_info_and_constructor,
                                                             LoadedDatasetDict,
                                                             obtain_classification_dataset_loaders,
                                                             obtain_classification_dataset_loaders_from_point_data)

from experimental_evaluation import _dict_optimizer_classes, _dict_scheduler_classes, _dict_loss_functions



####################################################################################
####################################################################################
# CONNECTIVITY FUNCTIONS
####################################################################################
####################################################################################


####################################################################################
# CONNECT TO MLFLOW
####################################################################################

def connect_to_mlflow(mlflow_logging="vmg"):
    if mlflow_logging.lower() == "vmg":
        # Check if the necessary environment variables are found:
        for ev in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
                   'MLFLOW_TRACKING_URI', 'MLFLOW_S3_ENDPOINT_URL']:
            assert os.getenv(ev) is not None, \
                f"Environment variable {ev} is not to be found. " + \
                f"(Tip: in the logging mode 'vmg' the variables are to be stored in .env!)"
        pass
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        mlflow.set_tracking_uri(mlflow_logging)
    pass
    #
    #########################################################################################################
    # NEW (motivated by crashes in Drago when storing GPU statistics, not made available in some GPUs and environments):
    # CHECK IF THE FEATURE IS AVAILABLE AND, IF NOT, DISABLE IT WITH A GOBAL VARIABLE FROM MLFlow
    #########################################################################################################
    supported = True
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Targeted check: This specific call crashes MLflow 2.18.0 if unsupported
        pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
    except (pynvml.NVMLError_NotSupported, pynvml.NVMLError):
        supported = False
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        pass
    pass

    # Set a global variable indicating if the system metrics can be logged, and also issuing a warning about it
    if supported:
        # Enable globally via environment variable
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.system_metrics.enable_system_metrics_logging()
    else:
        # Disable globally to prevent the thread crash
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
        mlflow.system_metrics.disable_system_metrics_logging()
        # Warning
        warnings.warn("System does not support system logging through 'pynvml': SYSTEM METRICS LOGGING DISABLED!")
    pass

pass


####################################################################################
####################################################################################
# READING OF EXPERIMENT RUN URIs AND GENERAL INFO (FOR LATER LOAD) AND LOADING OF METRICS
####################################################################################
####################################################################################


####################################################################################
# Retrieving (the list of) the runs in MLFlow corresponding to a certain experiment
####################################################################################

def list_runs_from_experiment_name(experiment_label, flag_only_finished=True,
                                   params_filter_dict=None, metrics_filter_dict=None):
    """
    Retrieve the list of runs in MLFlow corresponding to a certain experiment, \
    indicated by the experiment name provided by `experiment_label`, optionally filtering them by using respective \
    filtering values for certain parameters (*params*) and metrics (*metrics*) given in the dictionaries \
    `params_filter_dict` and `metrics_filter_dict`. The flag `flag_only_finished` indicates whether only the \
    finished runs are to be considered (which represents the default behavior of the function).

    Regarding the format of the `metrics_filter_dict`: each key corresponds to the name of a metric (without \
    comprising the prefix *'metrics.'*), filtering is performed based on scalar values according to the following \
    convention:

    - If the value is a scalar, then the filtering is done by equality.

    - If the value is a tuple, such tuple MUST have 2 scalar elements, which indicate the lower and upper bounds \
      (included) of a range used for filtering. In such case, in order to indicate 'greater than' or 'less than', \
      the corresponding +/- infinity value (e.g. :py:const:`numpy.inf`) or simply ``None`` in the corresponding \
      element of the tuple must be added: e.g. ``(0.6, np.inf)`` or analogously ``(0.6, None)`` \
      means greater or equal than $0.6$.

    - If the value is a list, then each element of the list would be considered as above (i.e. they must be either \
      scalar or 2D tuple...) and the different list "elements" will be combined as "OR".

    Regarding the format of the `params_filter_dict`: each key corresponds to the name of a parameter (without \
    comprising the prefix *'params.'*) with, OPTIONALLY, the uppercase word ``"NOT "`` appended to the name of the \
    parameter and separated by space(s) and negating the remain`ing comparisons if present, and the convention for \
    filtering (depending on the presence of such negation) is as follows:

    - If the value is a str, then the filtering is done by equality (if negated, inequality).

    - If the value is a list of strings, then the filtering is done as equality for each element of the list and \
      combined as "OR" (that is, the condition is "IN list"; if negated, not in "NOT IN list").

    Parameters
    ----------
    experiment_label : str|int
        Experiment name/id in MLFlow.
    flag_only_finished : bool, optional
        Flag indicating whether only the finished runs are to be considered. \
        Default: ``True``
    params_filter_dict : dict, optional
        Dictionary with the parameters to filter the runs.
    metrics_filter_dict : dict, optional
        Dictionary with the metrics to filter the runs.

    Returns
    -------
    :py:class:`pandas.DataFrame`
        Dataframe wherein each row corresponds to a run in the experiment, \
        and having, among others, columns such as *'run_id'* and *'artifact_uri'* for retrieval of the details \
        of said run, in addition to columns directly containing the params and metrics stored for the run \
        (with the format, respectively, *'params.<name>'* and *'metrics.<name>'*)
    """

    #########################
    # Initial checks (further checks will be performed while filtering)
    #########################

    # Get the experiment ID from the experiment name
    experiment_id = experiment_id_from_experiment_label(experiment_label)
    if experiment_id is None:
        raise Exception(f"Experiment '{experiment_label}' does not exist!")
    pass

    # Check that the 'flag_only_finished' is a boolean
    if not isinstance(flag_only_finished, bool):
        raise ValueError(f"Invalid 'flag_only_finished' type given: {type(flag_only_finished)} given, bool expected.")
    pass

    # Check that the 'params_filter_dict' is a non-empty dictionary (if empty dictionary set to None)
    if params_filter_dict is not None:
        if not isinstance(params_filter_dict, dict):
            raise ValueError(f"Invalid 'params_filter_dict' type given: {type(params_filter_dict)} given, dict expected.")
        elif len(params_filter_dict) == 0:
            params_filter_dict = None
        pass
    pass

    # Check that the 'metrics_filter_dict' is a non-empty dictionary (if empty dictionary set to None)
    if metrics_filter_dict is not None:
        if not isinstance(metrics_filter_dict, dict):
            raise ValueError(f"Invalid 'metrics_filter_dict' type given: {type(metrics_filter_dict)} given, dict expected.")
        elif len(metrics_filter_dict) == 0:
            metrics_filter_dict = None
        pass
    pass

    #########################
    # Load the Pandas DF with the data about all the runs in the experiment
    # (it is that 'raw" DF that we will filter, if indicated, with the filters)
    #########################

    # Get the DF describing the runs in the experiment
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)

    if "drago" not in socket.gethostname():
        # Reemplazar la parte específica en la columna 'artifact_uri'
        df_runs['artifact_uri'] = df_runs['artifact_uri'].str.replace(
            'file:///lustre/home/io/evelasco/',
            'file:///home/erik/',
            regex=False
        )
    #########################
    # Start the filtering in a new DF
    #########################

    df_filtered_runs = df_runs

    #########################
    # Filter only finished
    #########################

    if flag_only_finished:
        df_filtered_runs = df_filtered_runs[df_filtered_runs['status'] == 'FINISHED']
    pass

    #########################
    # Filter the DF using the filter dictionary 'params_filter_dict', if provided
    #########################

    if len(df_filtered_runs) == 0:
        warnings.warn(f"Experiment '{experiment_label}' does not have any finished runs!")
        return df_filtered_runs
    pass

    # WARNING: SO FAR ONLY EQUALITY IS CONSIDERED
    if params_filter_dict is not None:
        for key in params_filter_dict:
            ###
            # Check if the key is negated and extract the full key (with the 'params.' prefix)
            ###
            negated = False
            full_key = 'params.'
            #
            # Divide the key 'key' with spaces and blacks as separators
            key_parts = key.split()
            if len(key_parts) == 1:
                negated = False
                full_key = full_key + str(key_parts[0])
            elif len(key_parts) == 2 and key_parts[0].upper() == 'NOT':
                negated = True
                full_key = full_key + str(key_parts[1])
            else:
                raise Exception(
                    f"The filter 'params_filter_dict' is expected to have single-word keys or negated " +
                    f"(appended by 'NOT ') single-word keys: instead '{key}' found!")
            pass
            #
            # Check if the 'full_key' appears in the columns of the DataFrame 'df_filtered_runs'
            if full_key not in df_filtered_runs.columns:
                raise Exception(f"The key 'params.'+'{key}' requested in 'params_filter_dict' does not appear as " +
                                f"one of the (Pandas DF column) attributes stored for the run!")
            pass
            #
            ###
            # Check the key values and make list if not list
            ###
            list_values = params_filter_dict[key]
            # From 1 string to a list with only 1 string element
            if isinstance(list_values, str):
                list_values = [list_values]
            pass
            # Check the elements of the list
            if not isinstance(list_values, list):
                raise Exception(f"Invalid value type for the key '{full_key}': " +
                                f"{type(list_values)}. Expected str or list.")
            else:
                # Check that all elements are a string: if not, change them to string
                for ind, value in enumerate(list_values):
                    if not isinstance(value, str):
                        # raise Exception(f"Invalid value type for the key '{full_key}': {type(value)}. Expected str.")
                        # Issue a warning and treat the string-ified version of the element
                        warnings.warn(
                            f"Invalid value type for the key '{full_key}': {type(value)}, expected str; " +
                            f"converting to its str version '{str(value)}'.")
                        list_values[ind] = str(value)
                    pass
                pass
            pass
            #
            # Apply the filter: remove the rows where the value of the column 'full_key' is not in 'list_values'
            df_filtered_runs = df_filtered_runs[~df_filtered_runs[full_key].isin(list_values)] if negated \
                else df_filtered_runs[df_filtered_runs[full_key].isin(list_values)]
            #
        pass
    pass

    if len(df_filtered_runs) == 0:
        warnings.warn(f"Experiment '{experiment_label}' does not have any finished runs with the selected 'params'!")
        return df_filtered_runs
    pass

    #########################
    # Filter the DF using the filter dictionary 'metrics_filter_dict', if provided
    #########################

    if metrics_filter_dict is not None:
        for key in metrics_filter_dict:
            ###
            # Extract the full key (with the 'params.' prefix)
            ###
            full_key = 'metrics.' + str(key)
            #
            # Check if the 'full_key' appears in the columns of the DataFrame 'df_filtered_runs'
            if full_key not in df_filtered_runs.columns:
                raise Exception(f"The key 'metrics.'+'{key}' requested in 'metrics_filter_dict' does not appear as " +
                                f"one of the (Pandas DF column) attributes stored for the run!")
            pass
            #
            ###
            # Check the key values and make list if not list
            ###
            list_values = metrics_filter_dict[key]
            # From 1 scalar or 1 tuple to a list with them, if that is the case
            if isinstance(list_values, (float, int, tuple)):
                list_values = [list_values]
            pass
            # Check the elements of the list
            if not isinstance(list_values, list):
                raise Exception(f"Invalid value type for the key '{full_key}': " +
                                f"{type(list_values)}. Expected scalar, tuple or list.")
            else:
                # Use the elements of the list for checking and filtering (using OR)
                column_coincidence = df_filtered_runs[full_key] == 0
                column_coincidence[:] = False
                for value in list_values:
                    if isinstance(value, (float, int)):
                        # In this case we use equality
                        column_coincidence = column_coincidence | (df_filtered_runs[full_key] == value)
                    elif isinstance(value, (list,tuple)) and len(value) == 2:
                        # We fix the case of None values by substituting them by np.inf
                        value = (value[0] if value[0] is not None else -np.inf,
                                 value[1] if value[1] is not None else np.inf)
                        if isinstance(value[0], (float, int)) and isinstance(value[1], (float, int)) and \
                                value[0] <= value[1]:
                            # In this case we use a range
                            column_coincidence = column_coincidence | \
                                                 ((df_filtered_runs[full_key] >= value[0]) & \
                                                  (df_filtered_runs[full_key] <= value[1]))
                        else:
                            raise Exception(f"Invalid range for the key '{full_key}': " +
                                            f"{value}. Expected (a, b) with a <= b (or None in one end).")
                        pass
                    else:
                        raise Exception(f"Invalid value type for the key '{full_key}': {value}. " +
                                        f"Expected scalars or 2D tuples.")
                    pass
                pass
                #
                # Apply the filter, accumulated in the 'column_coincidence' variable
                df_filtered_runs = df_filtered_runs[column_coincidence]
            pass
        pass
    pass

    if len(df_filtered_runs) == 0:
        warnings.warn(f"Experiment '{experiment_label}' does not have any finished runs with the selected 'params' and 'metrics'!")
    pass

    return df_filtered_runs


####################################################################################
# Artifact URI from experiment name/ID and run name/ID
####################################################################################

def artifact_uri_from_experiment_and_run(experiment_label: str|int, run_label:str):
    """
    This function extracts the artifact URI string corresponding to the requested experiment name/ID and run name/ID. \
    If no corresponding run is found ``None`` is returned.

    Parameters
    ----------
    experiment_label : str|int
    run_label : str

    Returns
    -------
    str or None
    """

    #####
    # Check if 'experiment_label' is given as a name or as an ID
    #####

    assert isinstance(experiment_label, (str,int)), \
        f"The provided 'experiment_label', which can be ID or name, must be a string or int: " + \
        f"{experiment_label}, of type {type(experiment_label)}, found!"

    experiment_id = None
    if isinstance(experiment_label, int) or (isinstance(experiment_label, str) and experiment_label.isdigit()):
        # We assume the provided info is the ID
        experiment_id = int(experiment_label)
    else:
        # We assume the provided info is the name
        experiment_structure = mlflow.get_experiment_by_name(experiment_label)
        if experiment_structure is None:
            raise Exception(f"Experiment '{experiment_label}' does not exist!")
        pass
        experiment_id = experiment_structure.experiment_id
    pass

    #####
    # Guess whether 'run_label' is a run name or run ID
    # Our approach: if hexadecimal number, run ID; otherwise, run name.
    #####

    assert isinstance(run_label, str), \
        f"The provided 'run_label', which can be ID or name, must be a string or int: " + \
        f"{run_label}, of type {type(run_label)}, found!"

    run_id = None
    run_name = None
    try:
        run_id = int(run_label, 16) # Convertible into hexadecimal int?
        run_id = hex(run_id)[2:]
    except ValueError:
        run_name = run_label
    except Exception as e:
        raise
    pass

    #####
    # Load the DF corresponding to the experiment and run
    #####

    filter_string = f'run_id = "{run_id}"' if run_id is not None else f'tags.mlflow.runName = "{run_name}"'
    df_run = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)

    #####
    # Extract the artifact URI if the above was not empty
    #####
    artifact_uri = None
    if df_run is not None and len(df_run) > 0:
        artifact_uri = df_run.iloc[0]["artifact_uri"]

    return artifact_uri


####################################################################################
# Artifact URI from experiment name/ID and run name/ID
####################################################################################


def experiment_id_from_experiment_label(experiment_label: str|int):
    """
    This function returns the experiment ID corresponding to the indicated experiment label, \
    which could be either a experiment name or an experiment ID.

    Parameters
    ----------
    experiment_label : str|int

    Returns
    -------
    str
    """

    #####
    # Check if 'experiment_label' is given as a name or as an ID
    #####

    assert isinstance(experiment_label, (str, int)), \
        f"The provided 'experiment_label', which can be ID or name, must be a string or int: " + \
        f"{experiment_label}, of type {type(experiment_label)}, found!"

    experiment_id = None
    if isinstance(experiment_label, int) or (isinstance(experiment_label, str) and experiment_label.isdigit()):
        # We assume the provided info is the ID
        # We test that the experiment can be "connected to"
        experiment_structure = mlflow.get_experiment_by_name(str(experiment_label))
    else:
        # We assume the provided info is the name
        # We test that the experiment can be "connected to"
        experiment_structure = mlflow.get_experiment_by_name(experiment_label)
    pass

    experiment_id = None
    if experiment_structure is None:
        warnings.warn(f"Experiment '{experiment_label}' does not exist!")
    else:
        experiment_id = experiment_structure.experiment_id
    pass

    return experiment_id


def experiment_metrics_for_run(experiment_label: str|int=None, run_label:str=None):
    """
    This function extracts a dictionary with the main figures of merit of a certain experiment, if available: these \
    figures of merit are: 'best_acc' and 'best_loss'.
    The function uses the same `experiment_label`+`run_label` querying as defined by the \
    function py:func:`.artifact_uri_from_experiment_and_run`.

    Parameters
    ----------
    experiment_label : str|int
    run_label : str

    Returns
    -------
    dict
    """

    #####
    # Check if 'experiment_label' is given as a name or as an ID
    #####

    assert isinstance(experiment_label, (str, int)), \
        f"The provided 'experiment_label', which can be ID or name, must be a string or int: " + \
        f"{experiment_label}, of type {type(experiment_label)}, found!"

    experiment_id = experiment_id_from_experiment_label(experiment_label)
    if experiment_id is None:
        raise Exception(f"Experiment '{experiment_label}' does not exist!")
    pass

    #####
    # Guess whether 'run_label' is a run name or run ID
    # Our approach: if hexadecimal number, run ID; otherwise, run name.
    #####

    assert isinstance(run_label, str), \
        f"The provided 'run_label', which can be ID or name, must be a string or int: " + \
        f"{run_label}, of type {type(run_label)}, found!"

    run_id = None
    run_name = None
    try:
        run_id = int(run_label, 16)  # Convertible into hexadecimal int?
        run_id = hex(run_id)[2:]
    except ValueError:
        run_name = run_label
    except Exception as e:
        raise
    pass

    #####
    # Load the DF corresponding to the experiment and run
    #####

    filter_string = f'run_id = "{run_id}"' if run_id is not None else f'tags.mlflow.runName = "{run_name}"'
    df_run = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)

    #####
    # Extract the metrics of interest
    #####
    list_metrics = ['best_acc', 'best_loss', 'acc', 'perturbed_acc', 'perturbed_loss', 'perturbed_acc_sm', 'perturbed_loss_sm']
    dict_metrics = {}
    for metric in list_metrics:
        metric2tag = f"metrics.{metric}"
        if metric2tag in df_run.columns:
            dict_metrics[metric] = float(df_run.iloc[0][metric2tag])
        pass
    pass

    return dict_metrics


def experiment_metrics_history_for_run(experiment_label: str|int=None, run_label:str=None,
                                       averaged_epochs:int|float=None, method:str='median'):
    """
    This function extracts a dictionary with the **history** of the main figures of merit of a certain experiment, \
    if available: these figures of merit are: 'best_acc' and 'best_loss'.
    The function uses the same `experiment_label`+`run_label` querying as defined by the \
    function py:func:`.artifact_uri_from_experiment_and_run`.

    Parameters
    ----------
    experiment_label : str|int
    run_label : str
    averaged_epochs : int|float, optional
        If provided, it indicates that the loaded metrics are averaged over a window of size ``averaged_epochs``.
        Default: ``None``
    method: str, optional
        Value among 'median' and 'mean' indicating the method used for averaging if `averaged_epochs` is provided.
        Default: 'median'

    Returns
    -------
    pandas.DataFrame
    """

    #####
    # Check if 'experiment_label' is given as a name or as an ID
    #####

    assert isinstance(experiment_label, (str, int)), \
        f"The provided 'experiment_label', which can be ID or name, must be a string or int: " + \
        f"{experiment_label}, of type {type(experiment_label)}, found!"

    experiment_id = experiment_id_from_experiment_label(experiment_label)
    if experiment_id is None:
        raise Exception(f"Experiment '{experiment_label}' does not exist!")
    pass

    #####
    # Guess whether 'run_label' is a run name or run ID
    # Our approach: if hexadecimal number, run ID; otherwise, run name.
    #####

    assert isinstance(run_label, str), \
        f"The provided 'run_label', which can be ID or name, must be a string or int: " + \
        f"{run_label}, of type {type(run_label)}, found!"

    run_id = None
    run_name = None
    try:
        run_id = int(run_label, 16)  # Convertible into hexadecimal int?
        run_id = hex(run_id)[2:]
    except ValueError:
        run_name = run_label
    except Exception as e:
        raise
    pass

    #####
    # Assert if 'averaged_epochs' is valid
    #####
    if averaged_epochs is not None:
        assert isinstance(averaged_epochs, (int,float)) and averaged_epochs > 0, \
            f"'averaged_epochs', if provided, must be an integer or float > 0; {averaged_epochs} found!"
        averaged_epochs = float(averaged_epochs)
    pass
    assert method in ['median', 'mean'], \
        f"'method' must be among 'median' and 'mean'; '{method}' found!"

    #####
    # Extract the metrics of interest
    #####

    client = MlflowClient()

    list_metrics = ['val_acc', 'val_loss', 'train_acc', 'train_loss', 'epoch_fraction']
    # Create a Pandas DataFrame where the elements in "list_metrics" are columns and the index is the step
    df_metrics = pd.DataFrame(columns=list_metrics)
    for metric in list_metrics:
        try:
            retrieved_metric_history = client.get_metric_history(run_id, metric)
            for entry in retrieved_metric_history:
                df_metrics.loc[entry.step, metric] = float(entry.value)
            pass
        except Exception as e:
            print(f"EXCEPTION during metric {metric}: {e}")
        pass
    pass

    df_metrics.sort_index(inplace=True)

    #####
    # Average if so indicated
    #####
    if averaged_epochs is not None:
        df_metrics['epoch_fraction_to_time'] = pd.to_timedelta(df_metrics['epoch_fraction'], unit='s')
        df_metrics.set_index('epoch_fraction_to_time', inplace=True)
        window_in_seconds = f"{averaged_epochs:f}s"
        for metric in list_metrics:
            if metric != 'epoch_fraction':
                if method == 'median':
                    df_metrics[metric] = df_metrics[metric].rolling(window_in_seconds, min_periods=1).median()
                else:
                    df_metrics[metric] = df_metrics[metric].rolling(window_in_seconds, min_periods=1).mean()
                pass
        pass
        df_metrics.reset_index(drop=True, inplace=True)
    pass

    #####

    return df_metrics



####################################################################################
####################################################################################
# Functions for retrieving info, or retrieving info+recreating, information about a trained model
# and the training and validation processes from a specific run logged in MLFlow
####################################################################################
####################################################################################


####################################################################################
# Retrieving/recreating a classifier from a run in MLFlow
####################################################################################

def read_state_dict_from_run_artifacts_uri(run_artifacts_uri):
    """
    Retrieve the state dict of a model from the artifacts of a run in MLFlow.

    Parameters
    ----------
    run_artifacts_uri

    Returns
    -------

    """
    #
    artifact_name = "state_dict.pt"
    state_dict_uri = f"{run_artifacts_uri}/{artifact_name}"
    #
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=state_dict_uri, dst_path=temp_dir)
        # Load the state dict
        if torch.cuda.is_available():
            state_dict = torch.load(local_path)
        else:
            state_dict = torch.load(local_path, map_location=torch.device('cpu'))
    pass
    #
    return state_dict


def recreate_classifier_from_run_artifacts_uri(run_artifacts_uri,
                                               net=None, conv_like_type=None, conv_like_type_position=None,
                                               trainable_backbone_layers=None,
                                               trainable_backbone_conv_like_layers=None,
                                               trainable_backbone_batch_norm=None,
                                               trainable_head=None,
                                               verbose='medium',
                                               **requested_kwargs):
    """
    Recreate the (best) classifier (in terms of accuracy) corresponding to a certain run registered in MLFlow \
    from the run artifacts URI given by the argument `run_artifacts_uri`. Internally, the function \
    first obtains the parameters of the classifier from the stored state dict, which contains the parameters \
    used to construct the classifier, and then loads the trainable parameters of the classifier from the state dict \
    using the function :py:meth:`~torch.nn.Module.load_state_dict`.

    The default behavior of the present function is to recreate the classifier as it was originally constructed, \
    including the same type of convolutional layers. In such case no further parameters are required. \
    However, if the argument `conv_like_type` is given and is different from that of the original classifier, then the \
    recreated classifier will have such `conv_like_type` type of convolutional layer instead of the original one and \
    will "fit" the trainable parameters of the original classifier to the new one:

    - For those cases where the original classifier has a type of convolutional-like layer \
          containing fewer parameters than the `conv_like_type` newly requested, then: additional parameters \
          will be required as `**kwargs`; and the trainable parameters of the new classifier present in the \
          original one will be transferred literally whereas the ones not present therein will be \
          initialized as indicated in `**kwargs`.

    - For those cases where the original classifier has a type of convolutional-like layer \
        containing more parameters than the `conv_like_type` newly requested, then: the trainable parameters \
        not necessary in the current classifier will be directly discarded.

    At this regard: our classifiers contain a first part, which we can call the "backbone", composed among others \
    by convolutional-like layers (see :py:class:`.IBNNInternalLayer` and :py:class:`.IBNNLayer`, :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, \
    :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, and :py:class:`.SMLayer`) \
    but including also batch normalization and maxpool layers, and a second part, \
    which we can call the "head", composed by fully connected layers, and normalization and maxpool. \
    The optional flags `trainable_backbone_layers`, and its related alternative flags \
    `trainable_backbone_conv_like_layers` and `trainable_backbone_batch_norm`, and \
    `trainable_head` refer precisely to whether said parts of the new classifier are supposed to be set as \
    trainable, if ``True``, as not trainable, if ``False``, or left as the created+loaded classifier defaults to, \
    if ``None``. Regarding the flag `trainable_backbone_layers`: only when the flag is set to ``None`` \
    the values provided for the alternative (sub-)flags `trainable_backbone_conv_like_layers` and \
    `trainable_backbone_batch_norm` are assessed; when `trainable_backbone_layers` is provided (``True`` or ``False``) \
    the latter (sub-)flags are ignored.

    Parameters
    ----------
    run_artifacts_uri : str
        URI of the run artifacts
    net : str, optional
        Type of classifier to be created, given by a string among 'single_layer', 'double_layer', 'vggx', \
        'alexnet', and 'efficientnetv2s', to be used in the new classifier. If not given, the original type \
        of the classifier of the `run_artifacts_uri` will be used.
        Default: ``None``
    conv_like_type : str, optional
        Type of convolutional-like layer, given by a string among 'sm', 'inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', \
        'ibnn_internal', and 'ibnn', to be used in the new classifier. If not given, the original type \
        will be used.
        Default: ``None``
     conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``). \
        If not given, the original `conv_like_type_position` of the loaded network will be used.
        Default: ``None``
    trainable_backbone_layers, trainable_backbone_conv_like_layers, trainable_backbone_batch_norm, trainable_head : bool, optional
        Flags indicating whether the backbone convolutional-like layers, the backbone batch normalization layers, \
        and the head of the classifier are supposed to be set as trainable (``True``), as not trainable (``False``), \
        or left as the created+loaded classifier defaults to (``None``).
        Default: ``None``
    verbose : str, optional
        Default: ``'medium'``
    requested_kwargs : dict
        Additional parameters required for the new classifier in case the `conv_like_type` is different from the \
        original one. As merely an example: if the original classifier has a type of convolutional-like layer \
        of type 'sm' and the new one is to be of type 'inrfv2', then the new one will require additional parameters \
        to be passed in this argument, namely: 'initial_lambda' and 'w_kernel_size' at least.

    Returns
    -------
    ClassifierBaseModel
    """

    if verbose in ['medium', 'high']:
        print("-----")
        print("(Only) NEW args requested for the classifier:")
        for key in requested_kwargs:
            print(f"\t{key}: {requested_kwargs[key]}")
        pass
        print("-----")
    pass

    classifier_nn = None

    ############################################
    # Initial checks
    ############################################

    # Check that the 'net' is among the valid ones and get
    requested_net_class = None
    if net is not None:
        if not isinstance(net, str):
            raise ValueError(f"Invalid 'net' type given: {type(net)} given, str expected.")
        elif net not in _dict_classifiers_as_in_conf_file:
            raise ValueError(
                f"Invalid 'net': {net}, one of {_dict_classifiers_as_in_conf_file.keys()} expected.")
        else:
            requested_net_class = _dict_classifiers_as_in_conf_file[net]
        pass
    pass

    # Check that the 'conv_like_type' is among the valid ones
    if conv_like_type is not None:
        if not isinstance(conv_like_type, str):
            raise ValueError(f"Invalid 'conv_like_type' type given: {type(conv_like_type)} given, str expected.")
        elif conv_like_type not in _dict_conv_like_layers:
            raise ValueError(
                f"Invalid 'conv_like_type': {conv_like_type}, one of {_dict_conv_like_layers.keys()} expected.")
        pass
    pass
    requested_conv_like_type = conv_like_type

    # Check that the 'conv_like_type_position' is among the valid ones
    if conv_like_type_position is not None:
        if not isinstance(conv_like_type_position, str):
            raise ValueError(f"Invalid 'conv_like_type_position' type given: {type(conv_like_type_position)} given, str expected.")
        elif conv_like_type_position not in ['everywhere', 'first', 'last']:
            raise ValueError(
                f"Invalid 'conv_like_type_position': {conv_like_type_position}, one of ['everywhere', 'first', 'last'] expected.")
        pass
    pass
    requested_conv_like_type_position = conv_like_type_position

    # Check that the 'trainable_...' is among the valid ones
    aux_dict = {'trainable_backbone_conv_like_layers': trainable_backbone_conv_like_layers,
                'trainable_backbone_batch_norm': trainable_backbone_batch_norm,
                'trainable_head': trainable_head}
    for key in aux_dict:
        if aux_dict[key] is not None and not isinstance(aux_dict[key], bool):
            raise ValueError(f"Invalid '{key}' type given: {type(aux_dict[key])} given, bool (or None) expected.")
        pass
    pass

    # Check the "trainable" flags related to the backbone layers
    if trainable_backbone_layers is not None:
        if not isinstance(trainable_backbone_layers, bool):
            raise ValueError(f"Invalid 'trainable_backbone_layers' type given: {type(trainable_backbone_layers)} given, bool expected.")
        else:
            warnings.warn(
                f"Flag 'trainable_backbone_layers' provided: the flags 'trainable_backbone_conv_like_layers' and " +
                f"'trainable_backbone_batch_norm', also provided (respectively as " +
                f"{trainable_backbone_conv_like_layers} and {trainable_backbone_batch_norm}), will be ignored and " +
                f"the value of 'trainable_backbone_layers' ({trainable_backbone_layers}) will be used instead."
            )
            trainable_backbone_conv_like_layers = trainable_backbone_layers
            trainable_backbone_batch_norm = trainable_backbone_layers
        pass
    pass

    ############################################
    # Load the state dict of the run indicated by the 'run_artifacts_uri'
    ############################################

    state_dict = read_state_dict_from_run_artifacts_uri(run_artifacts_uri)

    # WARNING: THE STATE DICT STORES THE NET USING THE NAME OF THE CLASS (_dict_classifiers),
    #          NOT THE NAME OF THE CLASS AS IN THE CONFIG FILE (_dict_classifiers_as_in_conf_file)

    net_class = _dict_classifiers[state_dict['_extra_state']['net']]
    old_conv_like_type = state_dict['_extra_state']['conv_like_type']
    old_conv_like_type_position = state_dict['_extra_state'].get('conv_like_type_position', 'everywhere')
    #
    constructor_kwargs = state_dict['_extra_state']['constructor_kwargs']

    ############################################
    # Analyse the state dict to decide what to create (empty) and what to fill. The idea:
    # - we take the constructor of the retrieved classifier...
    # - and we substitute the parameters provided in the arguments
    #   `net`, `conv_like_type`, and `conv_like_type_position`, and `requested_kwargs` (if any)
    ############################################

    # Create a new dictionary with the constructor kwargs
    new_constructor_kwargs = copy.deepcopy(constructor_kwargs)

    # Check if the net is compatible!
    if requested_net_class is not None:
        if requested_net_class != net_class:
            # raise ValueError(f"The requested net class '{requested_net_class.__name__}' is not compatible with " +
            #                  f"the net class '{net_class.__name__}' of the run artifacts URI '{run_artifacts_uri}'.")
            print(f"The requested net class '{requested_net_class.__name__}' is not compatible with " +
                  f"the net class '{net_class.__name__}' of the run artifacts URI '{run_artifacts_uri}'.")
            classifier_nn = None
            return classifier_nn
        pass
    pass

    # Fill the "new" conv_like_type and the conv_like_type_position
    new_constructor_kwargs['conv_like_type'] = \
        requested_conv_like_type if requested_conv_like_type is not None \
            else old_conv_like_type
    new_constructor_kwargs['conv_like_type_position'] = \
        requested_conv_like_type_position if requested_conv_like_type_position is not None \
            else old_conv_like_type_position

    # Fill the rest!
    for key in requested_kwargs:
        new_constructor_kwargs[key] = requested_kwargs[key]
    pass

    # Print all the args of the constructor (before and after) if so indicated by the 'verbose' argument
    if verbose in ['high']:
        print("-----")
        print("(All) args used to create the loaded classifier:")
        for key in constructor_kwargs:
            print(f"\t{key}: {constructor_kwargs[key]}")
        pass
        print("-----")
        print("(All) args used for the (re)creation of the classifier:")
        for key in new_constructor_kwargs:
            print(f"\t{key}: {new_constructor_kwargs[key]}")
        pass
        print("-----")
    pass

    # Create the classifier
    classifier_nn = net_class(**new_constructor_kwargs)

    # Try to load the loaded state dict into the classifier!
    # classifier_nn.load_state_dict(state_dict, strict=False)

    # Build an exclusion list to make sure that some parameters, if requested, are kept:
    list_exclusions = []
    if 'initial_lambda' in requested_kwargs:
        list_exclusions.append('lambda')
    for elem in ['sigma_x_compress', 'sigma_y_stretch', 'sigma_x_offset', 'sigma_y_offset', 'w_padding', 'm_padding']:
        if elem in requested_kwargs:
            list_exclusions.append(elem)
        pass
    pass

    # Load with the exclusion_list
    classifier_nn.load_state_dict_but_exceptions(state_dict, list_exclusions, strict=False)

    ############################################
    # And we set the trainable/non-trainable parameters of the classifier as indicated in the arguments
    ############################################
    # Since there is no explicit separation between backbone and head in the classifiers, we will differentiate \
    # between the trainable parameters of the backbone and the head by the type of the layers. So:
    # - The backbone will be composed by the convolutional-like layers listed as the values within \
    #   the dictionary '_dict_conv_like_layers' and 2D batch normalization layers
    # - The head will be composed by the fully connected layers and 1D batch normalization layers
    ############################################

    if trainable_backbone_conv_like_layers is not None:
        for layer in classifier_nn._nn.backbone.modules():
            if isinstance(layer, tuple(_dict_conv_like_layers.values())):
                for name, param in layer.named_parameters():
                    param.requires_grad = trainable_backbone_conv_like_layers
                pass
            pass
        pass
    pass
    #
    if trainable_backbone_batch_norm is not None:
        for layer in classifier_nn._nn.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                for name, param in layer.named_parameters():
                    param.requires_grad = trainable_backbone_batch_norm
                pass
            pass
        pass
    pass
    #
    if trainable_head is not None:
        for layer in classifier_nn._nn.head.modules():
            for name, param in layer.named_parameters():
                param.requires_grad = trainable_head
            pass
        pass
    pass
    #
    return classifier_nn


####################################################################################
# Retrieving/recreating a dataset and dataset loader corresponding to a run in MLFlow
####################################################################################

def read_kwargs_dataset_from_run_artifacts_uri(run_artifacts_uri):
    """
    Obtain the parameters of the dataset from the run artifact URI, which are in fact to be extracted from the \
    pandas DataFrame that is stored in the artifact.

    CURRENT VERSION (2025-10-09): it uses the function :py:func:`.read_dataset_dict_from_run_artifacts_uri` instead of \
    reading from the logged pandas DataFrame, which is deprecated.

    Parameters
    ----------
    run_artifacts_uri : str

    Returns
    -------
    dict
    """

    kwargs_dataset = {}

    dataset_dict_wo_dataloaders = read_dataset_dict_from_run_artifacts_uri(run_artifacts_uri)
    # Get the value of the multi-level index 'dataset_name' of the data frame 'df_run'
    kwargs_dataset['dataset_name'] = dataset_dict_wo_dataloaders['dataset_name']
    kwargs_dataset['desired_im_size'] = (int(dataset_dict_wo_dataloaders['im_width']),
                                         int(dataset_dict_wo_dataloaders['im_height']))
    kwargs_dataset['loaded_im_colorspace'] = dataset_dict_wo_dataloaders['colorspace']
    kwargs_dataset['batch_size'] = int(dataset_dict_wo_dataloaders['batch_size'])
    kwargs_dataset['train_proportion'] = float(dataset_dict_wo_dataloaders['proportion_train'])
    kwargs_dataset['val_proportion'] = float(dataset_dict_wo_dataloaders['proportion_val'])
    kwargs_dataset['generator_seed'] = int(dataset_dict_wo_dataloaders['generator_seed']) \
        if isinstance(dataset_dict_wo_dataloaders['generator_seed'], (int, np.int64)) \
        else None

    return kwargs_dataset

def read_dataset_dict_from_run_artifacts_uri(run_artifacts_uri):
    """
    Obtain the parameters of the dataset from the run artifact URI, which are in fact to be extracted from the \
    pandas DataFrame that is stored in the artifact.

    Parameters
    ----------
    run_artifacts_uri : str

    Returns
    -------
    dict
    """
    #
    artifact_name = "dataset_dict.json"
    dataset_dict_uri = os.path.join(run_artifacts_uri, artifact_name)
    #
    dataset_dict_wo_dataloaders = mlflow.artifacts.load_dict(dataset_dict_uri)
    #
    # The field 'dataset_dict_wo_dataloaders['tuple_of_pairs_other_kwargs']' is read as a list of 2D lists, when \
    # it should be a tuple of tuples: must be corrected!
    list_of_tuple_pairs_other_kwargs = []
    for pair in dataset_dict_wo_dataloaders['tuple_of_pairs_other_kwargs']:
        list_of_tuple_pairs_other_kwargs.append(tuple(pair))
    dataset_dict_wo_dataloaders['tuple_of_pairs_other_kwargs'] = tuple(list_of_tuple_pairs_other_kwargs)
    #
    return dataset_dict_wo_dataloaders


def load_same_dataset_of_run_artifacts_uri(run_artifacts_uri):
    """
    Recreate (literally, create again) the :class:`.LoadedDatasetDict` from the run artifacts URI by first \
    obtaining the parameters of the dataset from the pandas DataFrame that is stored in the artifact, using \
    the function :func:`._retrieve_kwargs_dataset_from_run_artifacts_uri`, and then using these parameters to \
    create the dataset.

    CURRENT VERSION (2025-10-09): valid for both 'real' image datasets and for the custom dataset ('moons', etc.).

    Parameters
    ----------
    run_artifacts_uri : str

    Returns
    -------
    :class:`.LoadedDatasetDict`
    """

    # First, we load the 'dataset_dict_wo_dataloaders' to see what is the type of dataset
    dataset_dict_wo_dataloaders = read_dataset_dict_from_run_artifacts_uri(run_artifacts_uri)

    dataset_loader_dict = None
    if dataset_dict_wo_dataloaders['dataset_name'] in _dict_dataset_info_and_constructor:
        kwargs_dataset = read_kwargs_dataset_from_run_artifacts_uri(run_artifacts_uri)
        dataset_loader_dict = obtain_classification_dataset_loaders(**kwargs_dataset)
    elif dataset_dict_wo_dataloaders['dataset_name'] in ['moons', 'circles', 'blobs', 'classification']:
        # Use directly 'dataset_dict_wo_dataloaders' to load the dataset
        # Make the 'tuple_of_pairs_other_kwargs', tuple of 2D tuples, into a dictionary
        kwargs_for_point_generation = {}
        for pair in dataset_dict_wo_dataloaders['tuple_of_pairs_other_kwargs']:
            kwargs_for_point_generation[pair[0]] = pair[1]
        pass
        dataset_loader_dict = obtain_classification_dataset_loaders_from_point_data(
            dataset_name=dataset_dict_wo_dataloaders['dataset_name'],
            num_points_training=kwargs_for_point_generation.pop('num_points_training'),
            num_points_test=kwargs_for_point_generation.pop('num_points_test'),
            normalization=None if dataset_dict_wo_dataloaders['normalized_dataset']=='no' \
                else dataset_dict_wo_dataloaders['normalized_dataset'],
            batch_size=dataset_dict_wo_dataloaders['batch_size'],
            train_proportion=dataset_dict_wo_dataloaders['proportion_train'],
            val_proportion=dataset_dict_wo_dataloaders['proportion_val'],
            shuffle=True,
            generator_seed=dataset_dict_wo_dataloaders['generator_seed'],
            **kwargs_for_point_generation
        )
    else: # Probably a custom dataset
        # Use directly 'dataset_dict_wo_dataloaders' to load the dataset
        # Make the 'tuple_of_pairs_other_kwargs', tuple of 2D tuples, into a dictionary
        kwargs_for_point_generation = {}
        for pair in dataset_dict_wo_dataloaders['tuple_of_pairs_other_kwargs']:
            kwargs_for_point_generation[pair[0]] = pair[1]
        pass
        dataset_loader_dict = obtain_classification_dataset_loaders_from_point_data(
            dataset_name=dataset_dict_wo_dataloaders['dataset_name'],
            num_points_training=kwargs_for_point_generation.pop('num_points_training'),
            num_points_test=kwargs_for_point_generation.pop('num_points_test'),
            normalization=None if dataset_dict_wo_dataloaders['normalized_dataset'] == 'no' \
                else dataset_dict_wo_dataloaders['normalized_dataset'],
            batch_size=dataset_dict_wo_dataloaders['batch_size'],
            train_proportion=dataset_dict_wo_dataloaders['proportion_train'],
            val_proportion=dataset_dict_wo_dataloaders['proportion_val'],
            shuffle=True,
            generator_seed=dataset_dict_wo_dataloaders['generator_seed'],
            **kwargs_for_point_generation
        )
    pass

    print(f"Dataset loaded successfully!")
    # print(f"dataset_loader_dict: \n{dataset_loader_dict}")

    return dataset_loader_dict


def mlflow_log_dataset(loaded_dataset_dict:LoadedDatasetDict, mlflow_run_id:str,
                       flag_artifact:bool=True, flag_metrics:bool=True, flag_params:bool=True):
    """
    Log the dataset into the current MLFlow run, given by its run ID `mlflow_run_id`.
    The logging comprises 3 different aspects, which can be optionally selected by the flags \
    `flag_artifact`, `flag_metrics`, and `flag_params` but which are by default all set to ``True``:
    - logging the dataset dictionary as an artifact (``flag_artifact``) using :py:meth:`mlflow.artifacts.log_dict`;
    - logging certain metrics relative to the dataset (``flag_metrics``) using :py:meth:`mlflow.log_metrics`, \
      consisting in the fields of the dataset dictionary that are numerical, \
      and logged with names (metrics.)dataset.<field>;
    - logging certain parameters relative to the dataset (``flag_params``) using :py:meth:`mlflow.log_params`, \
      consisting in the string-ified fields of the dataset dictionary, \
      and logged with names (params.)dataset.<field>.

    Parameters
    ----------
    loaded_dataset_dict : :class:`.LoadedDatasetDict`
        Dataset loader structure
    mlflow_run_id : str
        Exactly so
    flag_artifact, flag_metrics, flag_params: bool, optional
        Default: ``True``

    Returns
    -------
    """

    assert isinstance(loaded_dataset_dict, LoadedDatasetDict), \
        f"Invalid 'loaded_dataset_dict' argument: {type(loaded_dataset_dict)}, LoadedDatasetDict expected."
    assert isinstance(mlflow_run_id, str), \
        f"Invalid 'mlflow_run_id' argument: {type(mlflow_run_id)}, str expected."
    for flag in [flag_artifact, flag_metrics, flag_params]:
        assert isinstance(flag, bool), \
            f"Invalid flag argument: {type(flag)}, bool expected."
    pass

    # Log as metrics
    if flag_metrics:
        for key, value in loaded_dataset_dict.dict_to_metrics_fields().items():
            mlflow.log_metrics({"dataset."+key: value}, run_id=mlflow_run_id)

    # Log as params
    if flag_metrics:
        for key, value in loaded_dataset_dict.dict_wo_dataloaders().items():
            mlflow.log_params({"dataset." + key: str(value)}, run_id=mlflow_run_id)

    # Log as metrics
    if flag_artifact:
        mlflow.log_dict(loaded_dataset_dict.dict_wo_dataloaders(),
                        'dataset_dict.json', run_id=mlflow_run_id)

####################################################################################
# Retrieving training and validation data (e.g. optimizer and scheduler options...) for classifier from a run in MLFlow
####################################################################################

def read_training_parameters_from_run_artifacts_uri(run_artifacts_uri):
    """
    Retrieve the training parameters from the run artifacts URI, which are in fact to be extracted from the \
    pandas DataFrame that is stored in the artifact.

    The training parameters comprise:

    - maximum_epochs (int), loss_function (Callable), validations_per_epoch (int), \
      validation_on_test_subset (bool), epochs_sm_based_warmup (int), early_stop_epochs (int)
    - optimizer_type, optimizer_class, optimizer_arguments
    - scheduler_type, scheduler_class, scheduler_arguments.

    Parameters
    ----------
    run_artifacts_uri

    Returns
    -------
    dict
    """
    #
    artifact_name = "pandas.pkl"
    pandas_df_uri = f"{run_artifacts_uri}/{artifact_name}"
    #
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=pandas_df_uri, dst_path=temp_dir)
        # Load the state dict
        df_run = pd.read_pickle(local_path)
    pass
    #
    ##########################
    # Create a (kwargs) dictionary to fill with the relevant data for training. It might have more or less data \
    # than the function 'experimental_evaluation.classifier_training' expects, but that will be addressed later \
    # if necessary.
    ##########################
    #
    kwargs_training = {}
    #
    ######
    # Data relative to the number of epochs
    ######
    #
    # Get the values relative to training from the multi-level index 'dataset_name' of the data frame 'df_run'
    #check if exists in the index
    if 'max_epochs' in df_run.index.names:
        kwargs_training['maximum_epochs'] = int(df_run.index.get_level_values('max_epochs')[0])
    else:
        kwargs_training['maximum_epochs'] = int(df_run.index.get_level_values('maximum_epochs')[0])
    kwargs_training['loss_function'] = _dict_loss_functions[df_run.index.get_level_values('loss_function')[0]]
    kwargs_training['validations_per_epoch'] = len(df_run.xs(1, level="epoch"))
    kwargs_training['validation_on_test_subset'] = df_run.index.get_level_values('subset_for_validation')[0] == 'test'
    # There is, probably, no stored data about the 'epochs_sm_based_warmup'. In such case, just ignore it
    if 'epochs_sm_based_warmup' in df_run.index:
        print(f"'epochs_sm_based_warmup' is in fact in df_run.index")
        if not pd.isnull(df_run.index.get_level_values('epochs_sm_based_warmup')[0]) and \
                not pd.isna(df_run.index.get_level_values('epochs_sm_based_warmup')[0]):
            kwargs_training['epochs_sm_based_warmup'] = int(df_run.index.get_level_values('epochs_sm_based_warmup')[0])
        pass
    else:
        print(f"'epochs_sm_based_warmup' is NOT in fact in df_run.index")
    pass
    # There is, probably, no stored data about the 'early_stop_epochs'. In such case, just ignore it
    if 'early_stop_epochs' in df_run.index:
        print(f"Early stop epochs is in fact in df_run.index")
        if not pd.isnull(df_run.index.get_level_values('early_stop_epochs')[0]) and \
                not pd.isna(df_run.index.get_level_values('early_stop_epochs')[0]):
            kwargs_training['early_stop_epochs'] = int(df_run.index.get_level_values('early_stop_epochs')[0])
        pass
    else:
        print(f"Early stop epochs is NOT in fact in df_run.index")
    pass
    #
    ######
    # Data relative to the optimizer
    ######
    #
    kwargs_training['optimizer_type'] = df_run.index.get_level_values('optimizer')[0].lower()
    kwargs_training['optimizer_class'] = _dict_optimizer_classes[kwargs_training['optimizer_type']]
    kwargs_training['optimizer_arguments'] = \
        json.loads(df_run.index.get_level_values('optimizer_args')[0].replace("'", "\"")) \
            if isinstance(df_run.index.get_level_values('optimizer_args')[0], str) \
            else None
    if isinstance(kwargs_training['optimizer_arguments'], dict) and len(kwargs_training['optimizer_arguments']) == 0:
        kwargs_training['optimizer_arguments'] = None
    pass
    #
    ######
    # Data relative to the scheduler
    ######
    #
    kwargs_training['scheduler_type'] = df_run.index.get_level_values('scheduler')[0] \
        if df_run.index.get_level_values('scheduler')[0] != 'None' else None

    if kwargs_training['scheduler_type'] is not None:
        kwargs_training['scheduler_class'] = _dict_scheduler_classes[kwargs_training['scheduler_type']]
        kwargs_training['scheduler_arguments'] = \
            json.loads(df_run.index.get_level_values('scheduler_args')[0].replace("'", "\"")) \
                if isinstance(df_run.index.get_level_values('scheduler_args')[0], str) \
                else None
    else:
        kwargs_training['scheduler_class'] = None
        kwargs_training['scheduler_arguments'] = None
    if isinstance(kwargs_training['scheduler_arguments'], dict) and len(kwargs_training['scheduler_arguments']) == 0:
        kwargs_training['scheduler_arguments'] = None
    pass
    #
    # print(f"OBTAINED TRAINING PARAMETERS:")
    # for key in kwargs_training:
    #     print(f"\t{key} = {kwargs_training[key]}")
    # pass
    #
    return kwargs_training



####################################################################################
####################################################################################
# Loading runs in MLFlow corresponding to a certain experiment
####################################################################################
####################################################################################


def recreate_complete_classifier_run_from_run_artifacts_uri(run_artifacts_uri,
                                                            conv_like_type=None, conv_like_type_position=None,
                                                            trainable_backbone_layers=None,
                                                            trainable_backbone_conv_like_layers=None,
                                                            trainable_backbone_batch_norm=None,
                                                            trainable_head=None,
                                                            **kwargs):
    """
    Recreate/retrieve all the artifacts corresponding to a single run in MLFlow, given as the `run_artifacts_uri` \
    for the artifacts of said experiment, ready for evaluation or retraining of the classifier. The artifacts \
    involved are returned as a dictionary with the following keys and content:

    - key ``'classifier_nn'``: the classifier NN, in the form of a :class:`.ClassifierBaseModel`, read and recreated \
      by the function :py:func:`~.recreate_classifier_from_run_artifacts_uri`;

    - key ``'dataset_loader_dict'``: the dataset loaders in the form of a :class:`.LoadedDatasetDict`, read and \
      by the function :py:func:`~.load_same_dataset_than_in_run_artifacts_uri`;

    - key ``'kwargs_training'``: a dictionary with the *kwargs* related to the training, such as optimizer \
      and scheduler info, read by the function :py:func:`~.read_training_parameters_from_run_artifacts_uri`.

    Regarding all the arguments of this current function beyond the compulsory `run_artifacts_uri`: they correspond \
    to the arguments of the function :py:func:`~.recreate_classifier_from_run_artifacts_uri` and are consumed \
    exclusively by it, so refer to the documentation of said function for further details.

    Parameters
    ----------
    run_artifacts_uri : str
        URI of the run artifacts
    conv_like_type : str, optional
        Type of convolutional-like layer, given by a string among 'sm', 'inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', and \
        'ibnn_internal', to be used in the new classifier. If not given, the original type \
        will be used.
        Default: ``None``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``). \
        If not given, the original `conv_like_type_position` of the loaded network will be used.
        Default: ``None``
    trainable_backbone_layers, trainable_backbone_conv_like_layers, trainable_backbone_batch_norm, trainable_head : bool, optional
        Flags indicating whether the backbone convolutional-like layers, the backbone batch normalization layers, \
        and the head of the classifier are supposed to be set as trainable (``True``), as not trainable (``False``), \
        or left as the created+loaded classifier defaults to (``None``).
        Default: ``None``
    kwargs : dict
        Additional parameters required for the new classifier in case the `conv_like_type` is different from the \
        original one. As merely an example: if the original classifier has a type of convolutional-like layer \
        of type 'sm' and the new one is to be of type 'inrfv2', then the new one will require additional parameters \
        to be passed in this argument, namely: 'initial_lambda' and 'w_kernel_size' at least.

    Returns
    -------
    dict
        Dictionary with the artifacts of the run, listed using the keys ``'classifier_nn'``, \
        ``'dataset_loader_dict'``, and ``'kwargs_training'``
    """

    #######################################
    # Recreate the classifier
    #######################################

    classifier_nn = recreate_classifier_from_run_artifacts_uri(
        run_artifacts_uri,
        conv_like_type=conv_like_type, conv_like_type_position=conv_like_type_position,
        trainable_backbone_layers=trainable_backbone_layers,
        trainable_backbone_conv_like_layers=trainable_backbone_conv_like_layers,
        trainable_backbone_batch_norm=trainable_backbone_batch_norm,
        trainable_head=trainable_head, **kwargs
    )

    #######################################
    # Recreate the dataset loader
    #######################################

    dataset_loader_dict = load_same_dataset_of_run_artifacts_uri(run_artifacts_uri)

    #######################################
    # Retrieve the training parameters
    #######################################

    kwargs_training = read_training_parameters_from_run_artifacts_uri(run_artifacts_uri)

    #######################################
    # Return as a dictionary
    #######################################

    return {'classifier_nn': classifier_nn,
            'dataset_loader_dict': dataset_loader_dict,
            'kwargs_training': kwargs_training}


####################################################################################
####################################################################################
# AUXILIARY FUNCTIONS
####################################################################################
####################################################################################


@contextmanager
def suppress_mlflow_stdout_stderr():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# Usage:
# with suppress_mlflow_stdout_stderr():
#     # Place your MLflow code here
#     mlflow.artifacts.download_artifacts(...)
