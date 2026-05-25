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
import re

import numpy as np
import pandas as pd
import math
from random import randint
from ast import literal_eval as make_tuple

import torch
import torch.nn.functional as F


import mlflow
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

import seaborn as sns
from seaborn import objects as so

from experimental_evaluation.interaction_with_mlflow import (connect_to_mlflow,
                                                             artifact_uri_from_experiment_and_run,
                                                             recreate_classifier_from_run_artifacts_uri,
                                                             experiment_id_from_experiment_label,
                                                             experiment_metrics_history_for_run,
                                                             list_runs_from_experiment_name,
                                                             experiment_metrics_for_run,
                                                             load_same_dataset_of_run_artifacts_uri)
from experimental_evaluation.operations_for_datasets import obtain_classification_dataset_loaders_from_point_data
from experimental_evaluation.experiment_utils import classifier_training, formatted_log_base_name
from modified_rf import SMLayer
from applications import _dict_classifiers_as_in_conf_file


#############################################################################################
#############################################################################################
#############################################################################################
#
# LOADING, FILTERING AND PROCESSING THE RESULTS CORRESPONDING TO A GIVEN EXPERIMENTAL CONDITION
#
#############################################################################################
#############################################################################################
#############################################################################################


#############################################################################################
# LOADING THE DF, FILTERED, IN ITS ORIGINAL FORMAT
#############################################################################################


def load_df_for_experiment_and_parameters(experiment_label,
                                          dataset_name=None, train_proportion=None, conv_like_type=None,
                                          num_conv_layers=None, m_padding=None, neurons_per_layer=None,
                                          channels=None, single_lambda=None, initial_lambda=None,
                                          p=None, **other_paras_to_filter):
    """
    This function loads the complete DF corresponding to the experiment with name/id `experiment_label` and \
    to the optional filters indicated by the rest of kw arguments.

    Parameters
    ----------
    experiment_label : str|int
    dataset_name : str, optional
        Default: ``None``
    train_proportion : float, optional
        Default: ``None``
    conv_like_type : str, optional
        Default: ``None``
    m_padding : str, optional
        Default: ``None``
    num_conv_layers : int, optional
        Default: ``None``
    neurons_per_layer : int, optional
        *This function is specifically intended for hidden layers in the FC mode, that is, with * \
        *m_padding=fc*.
        Default: ``None``
    channels : int|tuple[int], optional
        *This function is specifically intended for hidden layers NOT in the FC mode, that is, WITHOUT * \
        *m_padding=fc*. If (provided and) scalar, the number of layers MUST be provided and the \
        same number of channels for all layers is assumed; \
        if tuple, its length must be equal to the number of layers, otherwise an error is raised.
        Default: ``None``
    single_lambda : float|int, optional
        Only one of `single_lambda` and `initial_lambda` can be provided: one or the other, not both. \
        For `single_lambda` the number of layers must be given too.
        Default: ``None``
    initial_lambda : tuple[float|int], optional
        Only one of `single_lambda` and `initial_lambda` can be provided: one or the other, not both.
        Default: ``None``
    p: float|int, optional
        This argument refers to the parameter 'sigma_x_compress' of ibnn_internal, ibnn, and ibnn_lite nets.
        Default: ``None``
    other_paras_to_filter : dict, optional
        If provided, all the provided keys will be understood as 'params.', \
        their values transformed into strings, and used as params filters.

    Returns
    -------
    pandas.DataFrame
    """

    ##########################################
    # Experiment id
    ##########################################
    #
    experiment_id = experiment_id_from_experiment_label(experiment_label)
    if experiment_id is None:
        raise Exception(f"Experiment '{experiment_label}' does not exist!")
    pass
    #
    ##########################################
    # Fill the params and metrics to filter
    ##########################################
    params_filter_dict = {}
    metrics_filter_dict = {}

    ##########################################
    # Argument checks, and
    # fill the params and metrics to filter
    ##########################################
    #
    if dataset_name is not None:
        assert isinstance(dataset_name, str), f"'dataset_name' must be a string: {type(dataset_name)} provided!"
        params_filter_dict['dataset_name'] = dataset_name
    #
    if train_proportion is not None:
        assert isinstance(train_proportion, (int,float)) and 0<train_proportion<=1.0, \
            f"'train_proportion' must be a float between 0 and 1: {train_proportion} provided!"
        metrics_filter_dict['train_proportion'] = float(train_proportion)
    #
    if conv_like_type is not None:
        assert isinstance(conv_like_type, str), f"'conv_like_type' must be a string: {type(conv_like_type)} provided!"
        params_filter_dict['conv_like_type'] = conv_like_type
    #
    if num_conv_layers is not None:
        assert isinstance(num_conv_layers, int) and num_conv_layers >= 1, \
            f"'num_conv_layers' must be an integer>0: {num_conv_layers} provided!"
        params_filter_dict['num_conv_layers'] = f"{num_conv_layers:d}"
    #
    if m_padding is not None:
        assert isinstance(m_padding, str) and m_padding in ['fc', 'same', 'valid'], \
            f"'m_padding', if provided, must be a string among 'fc', 'same', 'valid': {m_padding} provided!"
        params_filter_dict['m_padding'] = m_padding
    if neurons_per_layer is not None:
        assert isinstance(neurons_per_layer, int) and neurons_per_layer > 1, \
            f"'neurons_per_layer' must be an integer>1: {neurons_per_layer} provided!"
        assert m_padding is None or m_padding == "fc", \
            f"When 'neurons_per_layer' provided 'm_padding' must be None or 'fc': {m_padding} provided!"
        params_filter_dict['m_padding'] = "fc"
        params_filter_dict['m_kernel_size_per_conv_layer'] = str(tuple([(neurons_per_layer, 1)] * num_conv_layers))
    #
    if channels is not None:
        assert m_padding is None or m_padding != "fc", \
            f"When 'channels' provided 'm_padding' must be None or NOT 'fc': {m_padding} provided!"
        if isinstance(channels, int):
            assert channels > 0, f"'channels' must be an integer>1: {channels} provided!"
            assert num_conv_layers is not None, f"'num_conv_layers' must be provided when 'channels' is an integer!"
            channels = tuple([channels] * num_conv_layers)
        pass
        if isinstance(channels, tuple):
            assert all([isinstance(elem, int) and elem > 0 for elem in channels]), \
                f"'channels' must be an integer>1 or a tuple of integers>1: {channels} provided!"
            if num_conv_layers is not None:
                assert len(channels) == num_conv_layers, \
                    f"Length of 'channels' and 'num_conv_layers' must coincide; " + \
                    f"however, {len(channels)} and {num_conv_layers} provided!"
            pass
        pass
        params_filter_dict['channels_per_conv_layer'] = str(channels)
    pass
    #
    if single_lambda is not None:
        assert initial_lambda is None, \
            f"Only one of 'single_lambda' and 'initial_lambda' must be provided: " + \
            f"both present ({single_lambda} and {initial_lambda})."
        assert num_conv_layers is not None, f"'num_conv_layers' must be provided when 'single_lambda' is used!"
        assert isinstance(single_lambda, (float,int)), \
            f"'single_lambda' must be a float or integer: {type(single_lambda)} provided!"
        initial_lambda = tuple([single_lambda]*num_conv_layers)
    elif initial_lambda is not None:
        assert isinstance(initial_lambda, tuple), f"'initial_lambda' must be a tuple: {type(initial_lambda)}"
        assert all([isinstance(elem, (int,float)) for elem in initial_lambda]), \
            f"'initial_lambda' must be a tuple of int|floats: {initial_lambda} found!"
        initial_lambda = tuple([float(elem) for elem in initial_lambda])
    pass
    if initial_lambda is not None:
        params_filter_dict['initial_lambda'] = str(initial_lambda)
    pass
    #
    if p is not None:
        assert isinstance(p, (float,int)), f"'p' must be a float/integer: {type(p)} provided!"
        params_filter_dict['sigma_x_compress'] = f"{p}"
    #
    if other_paras_to_filter is not None:
        assert isinstance(other_paras_to_filter, dict), \
            f"'other_paras_to_filter' must be a dict: {type(other_paras_to_filter)} found!"
        for key in other_paras_to_filter:
            params_filter_dict[key] = str(other_paras_to_filter[key])

    ##########################################
    # Load the DF using the filters
    ##########################################

    if len(params_filter_dict) == 0:
        params_filter_dict = None
    if len(metrics_filter_dict) == 0:
        metrics_filter_dict = None

    df = list_runs_from_experiment_name(experiment_label,
                                        params_filter_dict=params_filter_dict, metrics_filter_dict=metrics_filter_dict)
    df.reset_index(drop=True, inplace=True)

    return df


#############################################################################################
# LOADING THE DF WITH THE METRIC HISTORY OF THE RUNS OF AN EXPERIMENT FULFILLING CERTAIN FILTERS
#############################################################################################

def load_metric_history_for_experiment_and_parameters(experiment_label,
                                                      flag_sm_to_lambdas=True, flag_acc_100=False,
                                                      averaged_epochs:int|float=None, method:str='median',
                                                      **kwargs):
    """

    Parameters
    ----------
    experiment_label
    flag_sm_to_lambdas
    flag_acc_100
    averaged_epochs : int|float, optional
        If provided, it indicates that the loaded metrics are averaged over a window of size ``averaged_epochs``.
        Default: ``None``
    method: str, optional
        Value among 'median' and 'mean' indicating the method used for averaging if `averaged_epochs` is provided.
        Default: 'median'
    kwargs

    Returns
    -------
    pandas.DataFrame
    """

    ####################
    # Load the experiment runs filtered DF
    ####################

    df_experiment_runs = load_df_for_experiment_and_parameters(experiment_label, **kwargs)
    df_experiment_runs.reset_index(drop=True, inplace=True)

    ####################
    # Extract the DF with the metrics of each run, and put all runs in a list
    ####################

    # Make the column 'lambda' host tuples (otherwise it would host other objects but not understand tuples)
    num_conv_layers = kwargs.get('num_conv_layers', None)

    list_df_metrics = []

    for i in range(len(df_experiment_runs)):
        #
        df_i = experiment_metrics_history_for_run(experiment_label, df_experiment_runs.loc[i, 'run_id'],
                                                  averaged_epochs=averaged_epochs, method=method)
        #
        df_i.reset_index(drop=True, inplace=True)
        #
        run_id = df_experiment_runs.loc[i, 'run_id']
        df_i['run_id'] = run_id
        #
        model = df_experiment_runs.loc[i, 'params.conv_like_type']
        df_i['m'] = model
        #
        df_i['d'] = df_experiment_runs.loc[i, 'params.dataset_name'] \
            if 'params.dataset_name' in df_experiment_runs else None
        #
        df_i['t'] = df_experiment_runs.loc[i, 'metrics.train_proportion'] \
            if 'metrics.train_proportion' in df_experiment_runs else 1.0
        df_i['w'] = df_experiment_runs.loc[i, 'metrics.mislabeled_proportion'] \
            if 'metrics.mislabeled_proportion' in df_experiment_runs else 0.0
        df_i['seed'] = df_experiment_runs.loc[i, 'metrics.generator_seed'] \
            if 'metrics.generator_seed' in df_experiment_runs else None
        #
        channels_per_conv_layer = make_tuple(df_experiment_runs.loc[i, 'params.channels_per_conv_layer'])
        col_for_tuples = pd.Series([channels_per_conv_layer] * len(df_i))
        df_i['c'] = col_for_tuples
        df_i['c0'] = channels_per_conv_layer[0]
        #
        # Estimate the number of parameters in the net
        in_size = make_tuple(df_experiment_runs.loc[i, 'params.in_size'])
        m_kernel_size_per_conv_layer = make_tuple(df_experiment_runs.loc[i, 'params.m_kernel_size_per_conv_layer'])
        abs_kernel_size_per_conv_layer = tuple(
            [tuple([round(m_kernel_size_layer_i[i]*in_size[i]) for i in range(len(m_kernel_size_layer_i))]) \
                 if m_kernel_size_layer_i[0]<1.0 \
                 else m_kernel_size_layer_i \
             for m_kernel_size_layer_i in m_kernel_size_per_conv_layer]
        )
        df_i['num_params'] = sum(
            [ch_layer_i + (ch_layer_i*ch_layer_i)*math.prod(list(abs_m_kernel_size_layer_i)) \
             for abs_m_kernel_size_layer_i, ch_layer_i in zip(abs_kernel_size_per_conv_layer, channels_per_conv_layer)]
        )
        #
        if model == 'sm' and flag_sm_to_lambdas:
            this_tuple = tuple([0.0] * num_conv_layers) if num_conv_layers is not None else (0.0,)
        else:
            this_tuple = make_tuple(df_experiment_runs.loc[i, 'params.initial_lambda'])
        pass
        col_for_tuples = pd.Series([this_tuple] * len(df_i))
        df_i['lambda'] = col_for_tuples
        df_i['lambda0'] = this_tuple[0]
        #
        # Extract also 'lambda_trainable'.
        if model == 'sm':
            df_i['lambda_trainable'] = False
        else:
            df_i['lambda_trainable'] = df_experiment_runs.loc[i, 'params.lambda_trainable'] == 'True'
        pass

        list_df_metrics.append(df_i)

    ####################
    # Concatenate all them in one single DF
    ####################

    concatenated_df = pd.concat(list_df_metrics)

    ####################
    # Include errors (apart from acc) and make everything over 100 (%) if so requested
    ####################

    for key in concatenated_df.columns:
        if 'acc' in key:
            new_key = key.replace('acc', 'err')
            concatenated_df[new_key] = 1.0 - concatenated_df[key]
        pass
    pass

    if flag_acc_100:
        for key in concatenated_df.columns:
            if 'acc' in key or 'err' in key:
                concatenated_df[key] = concatenated_df[key] * 100
        pass
    pass

    concatenated_df.reset_index(drop=True, inplace=True)
    return concatenated_df



def load_summarized_metric_history_for_experiment_and_parameters(experiment_label, **kwargs):
    """
    The kwargs parameter are used as arguments for the function \
    :py:func:`.load_metric_history_for_experiment_and_parameters`, called internally by the present function.

    Parameters
    ----------
    experiment_label
    kwargs

    Returns
    -------
    pandas.DataFrame
    """

    ####################
    # We load the complete experiment history using "load_metric_history_for_experiment_and_parameters"
    # and then, run by run, summarize the last epochs
    ####################

    df_metric_history = load_metric_history_for_experiment_and_parameters(experiment_label, **kwargs)

    # Process individually each run id and accumulate the result for later concatenation
    list_df_summarized_metrics = []

    for run_id in df_metric_history.run_id.unique():
        #
        # Filter the rows for the current 'run_id'
        df_i = df_metric_history[df_metric_history['run_id'] == run_id].copy()
        #
        if len(df_i.index) > 0:  # If it is empty do not process at all
            #
            # Fields to keep
            fields_to_transfer = ["val_acc", "val_loss", "train_acc", "train_loss", "epoch_fraction"]

            #############################################
            # We get the results AT THE LAST RECORDED EPOCH
            # (and we use it as the "base", since it has all the rest of fields that we would keep
            #############################################
            # Order df_i based on the column 'epoch_fraction'
            df_i.sort_values(by='epoch_fraction', inplace=True)
            # Take the epoch fraction of the last row (i.e. last calculated epoch)
            df_row_summarized_metrics_last_i = df_i.iloc[-1].to_frame().T.copy()
            # Rename the metrics to keep, appending the prefix "last_"
            for key in fields_to_transfer:
                df_row_summarized_metrics_last_i.rename(columns={key: f"last_{key}"}, inplace=True)
            pass

            #############################################
            # We get the results AT THE EPOCH OF THE BEST RECORDED VAL ACC and add them
            #############################################
            # Order df_i based on the column 'val_acc'
            df_i.sort_values(by='val_acc', inplace=True)
            # Take the epoch fraction of the last row, which in this case is the best
            series_row_summarized_metrics_best_i = df_i.iloc[-1].copy()
            # Copy only the (metric) fields in 'fields_to_transfer' renaming them appending "best_"
            for key in fields_to_transfer:
                df_row_summarized_metrics_last_i[f"best_{key}"] = series_row_summarized_metrics_best_i[key]
            pass

            # Accumulate in the list 'list_df_metrics' for later concatenation
            list_df_summarized_metrics.append(df_row_summarized_metrics_last_i)
        pass
    pass

    # Concatenate all the summarized dfs in one single df
    summarized_df = pd.concat(list_df_summarized_metrics)
    summarized_df.reset_index(drop=True, inplace=True)

    return summarized_df




#############################################################################################
# LOADING THE DF OF A 2D EXPERIMENT, FILTERED, SO THE EXPERIMENTS ARE ORGANIZED HAVING THE COLUMNS 'acc', 'n', 'p', 'lambda'
#############################################################################################


def load_2D_experiment_acc_for_n_lambda_p(experiment_label, method=None, flag_acc_100=False, **kwargs):
    """
    *This function is specifically intended for hidden layers in their FC mode, that is, with m_padding=fc*. \
    *Additinally, it is set for these cases: (i) one single hidden layer, or * \
    *(ii) all layers having the same number of units and the same $\\lambda$.*

    It loads the runs in the experiment indicated by `experiment_label` and according to the filters in `kwargs` \
    using the function :py:func:`.load_df_for_experiment_and_parameters` \
    (`kwargs` refers to the filters of the latter), and generates (through pivoting) a :py:class:`pandas.DataFrame` \
    having, as columns, the number of hidden units ('n'), $\\lambda$ ('lambda') and $p$ ('p') of the run/runs \
    and the (validation) accuracy ('acc').
    If `method` is ``None`` the different seeds of the same experiment will not be grouped! Otherwise \
    they will be summarized/collapsed using the indicated method.

    Parameters
    -------
    experiment_label : str|int
        Experiment name/id
    method : str, optional
        Method among ['mean', 'max', 'min', 'median'].
        Default: ``None``
    flag_acc_100 : bool, optional
        Default: ``False``
    kwargs : dict
        Keyword arguments to be passed to :py:func:`.load_df_for_experiment_and_parameters`.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe having, per row, the 'n', 'p', 'lambda', and 'acc' of an experiment.
    """

    #
    ################################################
    # Check the summarization method and the columns_to_leave
    ################################################
    #
    if method is not None:
        assert isinstance(method, str) and method in ['mean', 'max', 'min', 'median'], \
            f"Unknown (summarization) method '{method}'!"
    pass
    print(f"Summarization method: {str(method)}")
    #
    ################################################
    # Load the dataset
    ################################################
    #
    df = load_df_for_experiment_and_parameters(experiment_label, **kwargs)

    ################################################
    # Process the dataset
    ################################################
    table = None
    if df is None or len(df) == 0:
        return df
    else:
        #
        df['lambda'] = df.apply(
            lambda x: (make_tuple(x['params.initial_lambda'])[0] if x['params.sigma_x_compress'] is not None else 0.0),
            axis=1
        )
        df['n'] = df.apply(lambda x: make_tuple(x['params.m_kernel_size_per_conv_layer'])[0][0], axis=1)
        df['p'] = df.apply(lambda x:
                           (float(x['params.sigma_x_compress']) if x['params.sigma_x_compress'] is not None else 0.0),
                           axis=1)
        df['seed'] = df.apply(lambda x:
                           (int(x['params.generator_seed']) if x['params.generator_seed'] is not None else -1),
                           axis=1)
        df.rename(columns={'metrics.best_acc': 'acc'}, inplace=True)
        #
        ################################################
        # Pivot the table and summarize using agg function, if desired
        ################################################
        extra_args = {'aggfunc': method} if method is not None else {}
        index=['n', 'lambda', 'p'] if method is not None else ['n', 'lambda', 'p', 'seed']

        table = pd.pivot_table(
            df, values='acc',
            index=index,
            **extra_args
        ).reset_index()

        ################################################
        # Add the complementary of the acc (err) and scale if desired
        ################################################
        table['err'] = 1.0 - table['acc']
        if flag_acc_100:
            table['acc'] = table['acc']*100
            table['err'] = table['err'] * 100
        pass
    pass

    return table



#############################################################################################
# LOADING THE DF OF AN EXPRESSIVITY EXPERIMENT
#############################################################################################


def _calculate_num_params_backbone_net(df_row):
    """
    The DF row (a Series) comes from the DF of an experiment. This function is meant to be used inside "apply" to \
    calculate a col./series with the number of parameters of a net.

    Parameters
    ----------
    df_row : pandas.Series

    Returns
    -------
    int : number of parameters
    """

    ######
    # Params related to the backbone
    ######

    in_size = make_tuple(df_row['params.in_size'])
    in_channels = int(df_row['params.in_channels'])
    m_kernel_size_per_conv_layer = make_tuple(df_row['params.m_kernel_size_per_conv_layer'])
    out_channels_per_conv_layer = make_tuple(df_row['params.channels_per_conv_layer'])
    in_channels_per_conv_layer = tuple([in_channels] + list(out_channels_per_conv_layer[0:-1]))
    abs_kernel_size_per_conv_layer = tuple(
        [tuple([round(m_kernel_size_layer_i[i] * in_size[i]) for i in range(len(m_kernel_size_layer_i))]) \
             if m_kernel_size_layer_i[0] < 1.0 \
             else m_kernel_size_layer_i \
         for m_kernel_size_layer_i in m_kernel_size_per_conv_layer]
    )

    ######
    # Calculation
    ######

    num_params_backbone = sum(
        [out_ch_layer_i + out_ch_layer_i * in_ch_layer_i * math.prod(list(abs_m_kernel_size_layer_i)) \
         for abs_m_kernel_size_layer_i, out_ch_layer_i, in_ch_layer_i \
         in zip(abs_kernel_size_per_conv_layer, out_channels_per_conv_layer, in_channels_per_conv_layer)]
    )

    return num_params_backbone


######################################################


def _calculate_num_params_complete_net(df_row):
    """
    The DF row (a Series) comes from the DF of an experiment. This function is meant to be used inside "apply" to \
    calculate a col./series with the number of parameters of a net.

    Parameters
    ----------
    df_row : pandas.Series

    Returns
    -------
    int : number of parameters
    """

    ######
    # Params related to the head
    ######

    in_size = make_tuple(df_row['params.in_size'])
    out_channels_per_conv_layer = make_tuple(df_row['params.channels_per_conv_layer'])

    out_classes = int(df_row['params.out_classes'])
    fc_num_layers = int(df_row['params.fc_num_layers'])
    fc_num_units_intermediate_layers = int(df_row['params.fc_num_units_intermediate_layers'])

    ######
    # Calculation
    ######

    num_params_backbone = _calculate_num_params_backbone_net(df_row)

    list_elements_fc = [out_channels_per_conv_layer[-1] * in_size[0] * in_size[1]] + \
                       [fc_num_units_intermediate_layers]*(fc_num_layers-1) + \
                       [out_classes]
    num_params_fc = math.prod([list_elements_fc[i] * (list_elements_fc[i-1] + 1) \
                               for i in range(1, len(list_elements_fc))])

    num_params = num_params_backbone + num_params_fc

    return num_params


######################################################


def _set_errorbar_for_plot(with_ci:bool|int=False):

    #################
    # Set the CI (with 90 as default value if True provided)
    #################
    ci = None
    if isinstance(with_ci, bool):
        if with_ci:
            with_ci = 90  # The default value
        else:
            with_ci = None
        pass
    pass
    if with_ci is not None:
        assert isinstance(with_ci, int) and 0 < with_ci <= 100, \
            f"Arg 'with_ci', if int, must be between 0 and 100 (included); instead, {with_ci} found."
        ci = ('ci', with_ci)
    pass

    return ci


######################################################


_dict_estimators_lineplot = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min,
    'median': np.median,
}


######################################################


def load_expressivity_experiment_acc_for_t_lambda_p(experiment_label, method=None, flag_acc_100=False,
                                                    **kwargs):
    """
    It loads the runs in the experiment indicated by `experiment_label` and according to the filters in `kwargs` \
    using the function :py:func:`.load_df_for_experiment_and_parameters` \
    (`kwargs` refers to the filters of the latter), and generates (through pivoting) a :py:class:`pandas.DataFrame` \
    having, as columns, $\\lambda$ ('lambda') and $p$ ('p') of the run/runs, the percentage ('t') of the dataset, \
    and the (validation) accuracy ('acc').
    If `method` is ``None`` the different seeds of the same experiment will not be grouped! Otherwise \
    they will be summarized/collapsed using the indicated method.

    Parameters
    -------
    experiment_label : str|int
        Experiment name/id
    method : str, optional
        Method among ['mean', 'max', 'min', 'median'].
        Default: ``None``
    flag_acc_100 : bool, optional
        Default: ``False``
    kwargs : dict
        Keyword arguments to be passed to :py:func:`.load_df_for_experiment_and_parameters`.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe having, per row, the 'n', 'p', 'lambda', and 'acc' of an experiment.
    """

    #
    ################################################
    # Check the summarization method and the columns_to_leave
    ################################################
    #
    if method is not None:
        assert isinstance(method, str) and method in ['mean', 'max', 'min', 'median'], \
            f"Unknown (summarization) method '{method}'!"
    pass
    print(f"Summarization method: {str(method)}")
    #
    ################################################
    # Load the experiment
    ################################################
    #
    df = load_df_for_experiment_and_parameters(experiment_label, **kwargs)

    ################################################
    # Process the dataset
    ################################################

    table = None
    if df is None or len(df) == 0:
        return df
    else:
        #######################################################
        # Format columns
        #######################################################
        #
        # Make the column 'lambda' host tuples (otherwise it would host other objects but not understand tuples)
        num_conv_layers = kwargs.get('num_conv_layers', None)
        #
        if 'metrics.train_proportion' in df.columns:
            df['t'] = df.apply(
                lambda x: (float(x['metrics.train_proportion']) if x['metrics.train_proportion'] is not None else 0.0),
                axis=1
            )
        else:
            df['t'] = 1.0
        pass
        #
        #########################
        # Channels per layer
        #########################
        value_for_c = (1,) if num_conv_layers is None else tuple([1] * num_conv_layers)
        new_col_c = pd.Series([value_for_c] * len(df))
        df['c'] = new_col_c
        if 'params.channels_per_conv_layer' in df.columns:
            df['c'] = df.apply(
                lambda x: (tuple(make_tuple(x['params.channels_per_conv_layer']))
                           if x['params.channels_per_conv_layer'] is not None else value_for_c),
                axis=1
            )
        pass
        #
        #########################
        # Kernel size (m) per layer
        #########################
        value_for_k = ((1, 1),) if num_conv_layers is None else tuple([(1,1)] * num_conv_layers)
        new_col_k = pd.Series([value_for_k] * len(df))
        df['k'] = new_col_k
        if 'params.m_kernel_size_per_conv_layer' in df.columns:
            df['k'] = df.apply(
                lambda x: (tuple(make_tuple(x['params.m_kernel_size_per_conv_layer']))
                           if x['params.m_kernel_size_per_conv_layer'] is not None else new_col_k),
                axis=1
            )
        pass

        ################################
        # 'c0': Num channels of the first layer and
        # 'k0': Side of the kernel m of the first layer
        # 'num_params': Num of estimated parameters of the network
        ################################
        df['c0'] = df.apply(lambda x: x['c'][0], axis=1)
        df['k0'] = df.apply(lambda x: x['k'][0][0], axis=1)
        #
        df['num_params_backbone'] = df.apply(_calculate_num_params_backbone_net, axis=1)
        df['num_params'] = df.apply(_calculate_num_params_complete_net, axis=1)
        #
        value_for_lambda = 0.0 if num_conv_layers is None else tuple([0.0] * num_conv_layers)
        new_col_lambda = pd.Series([value_for_lambda] * len(df))
        df['lambda'] = new_col_lambda
        if 'params.initial_lambda' in df.columns:
            df['lambda'] = df.apply(
                lambda x: (tuple(make_tuple(x['params.initial_lambda'])) if x['params.initial_lambda'] is not None else value_for_lambda),
                axis=1
            )
        pass
        #
        if 'params.sigma_x_compress' in df.columns:
            df['p'] = df.apply(lambda x:
                               (float(x['params.sigma_x_compress']) if x['params.sigma_x_compress'] is not None else 0.0),
                               axis=1)
        else:
            df['p'] = 0.0
        pass
        #
        if 'params.generator_seed' in df.columns:
            df['seed'] = df.apply(lambda x:
                               (int(x['params.generator_seed']) if x['params.generator_seed'] is not None else -1),
                               axis=1)
        else:
            df['seed'] = -1
        pass
        #
        df.rename(columns={'metrics.best_acc': 'acc'}, inplace=True)
        #
        ################################################
        # Pivot the table and summarize using agg function, if desired
        ################################################
        extra_args = {'aggfunc': method} if method is not None else {}
        basic_index = ['t', 'c', 'c0', 'k', 'k0', 'lambda', 'p', 'num_params', 'num_params_backbone']
        index = basic_index if method is not None else basic_index + ['seed']

        table = pd.pivot_table(
            df, values='acc',
            index=index,
            **extra_args
        ).reset_index()

        ################################################
        # Add the complementary of the acc (err) and scale if desired
        ################################################
        table['err'] = 1.0 - table['acc']
        if flag_acc_100:
            table['acc'] = table['acc']*100
            table['err'] = table['err'] * 100
        pass
    pass

    return table


#############################################################################################


def plot_comparison_expressivity_results(list_tuples_total_df, dataset_name,
                                         display_mode='num_params',
                                         flag_err=False, flag_log_x=False,
                                         agg_method="median", with_ci:bool|int=False,
                                         ax=None, savefig=False, folder_figs=".", prefix=None):
    """

    Parameters
    ----------
    list_tuples_total_df : list[tuple]
        List of 3D tuples, where: \
        the 1st element of each tuple is a DF with results; \
        the 2nd element is the color for the display of such tuple; and\
        the 3rd element is the label for the tuple.
    dataset_name : str
    display_mode : str, optional
        String among 'num_params', 'num_params_backbone', 'c0', and 'k0'. \
        'c0' indicates the number for channels of the first hidden layer, and \
        'k0', kernel size of the first layer.
        Default: 'num_params'
    flag_err
    flag_log_x
    flag_params_backbone_only : bool, optional
        If ``True``, the x-axis will be the number of parameters of the backbone net. Otherwise, \\
        the x-axis will be the number of parameters of the complete net (backbone + head).
        Default: ``False``.
    agg_method : str, optional
        Method among ['mean', 'max', 'min', 'median'].
        Default: "median"
    with_ci
    ax
    savefig
    folder_figs
    prefix

    Returns
    -------

    """
    #
    dict_x_labels = {
        'num_params': 'Num. params',
        'num_params_backbone': 'Num. params backbone',
        'c0': 'Num. channels/layer',
        'k0': 'm, kernel side'
    }
    valid_display_modes = list(dict_x_labels.keys())
    #
    assert display_mode in valid_display_modes, \
        f"The argument 'display_mode' must be one of {valid_display_modes}; instead, {display_mode} found."
    assert all([display_mode in entry[0].columns for entry in list_tuples_total_df]), \
        f"A DF of 'list_tuples_total_df' does not contain the column '{display_mode}' selected as 'display_mode'."
    x_value = display_mode
    x_label = dict_x_labels[display_mode]
    #
    field_value = 'err' if flag_err else 'acc'
    #
    min_unit = +np.inf
    max_unit = -np.inf
    #
    ax_err = ax
    if ax_err is None:
        fig_err, ax_err = plt.subplots(figsize=(8,6))
    pass
    #
    list_legend = []
    list_names_df = []
    #
    ci = _set_errorbar_for_plot(with_ci)
    #
    assert isinstance(agg_method, str) and agg_method in _dict_estimators_lineplot, \
        f"Unknown 'agg_method' '{agg_method}'! It must be a string among 'mean', 'max', 'min', 'median'."
    #
    # Get the xticks to be plotted
    xticks = []
    for total_df, color_df, name_df in list_tuples_total_df:
        # total_df[x_value] = total_df['c'].apply(lambda x: x[0])
        estimator_lineplot = _dict_estimators_lineplot[agg_method]
        sns.lineplot(x=total_df[x_value], y=total_df[field_value], color=color_df,
                     estimator=estimator_lineplot, errorbar=ci, ax=ax_err)
        list_legend.extend([name_df, None] if with_ci else [name_df])
        list_names_df.append(name_df)
        #
        min_unit = min(min_unit, total_df[x_value].min())
        max_unit = max(max_unit, total_df[x_value].max())
        #
        # xticks.extend(list(total_df[x_value].unique()))
        xticks = list(total_df[x_value].unique())
    pass
    xticks = list(set(xticks))
    #
    ax_err.set_xlabel(x_label + (" (log scale)" if flag_log_x else ""))
    ax_err.set_ylabel(f'{field_value.capitalize()} (%)')
    ax_err.grid(True)
    ax_err.legend(list_legend)
    ax_err.set_title(f"Mean {field_value} achieved at each network complexity on '{dataset_name}'" +
                     (f" ({ci[0].upper()} {ci[1]})" if with_ci else ""))
    if flag_log_x:
        ax_err.set_xscale('log')
        ax_err.minorticks_off()
        ax_err.set_xticks(xticks, labels=[f"{x:d}" for x in xticks])
    else:
        ax_err.set_xticks(xticks)
    pass
    #
    if "num_params" in display_mode:
        # If "num_params" or "num_params_backbone", the xticks are long: rotate them
        # ax.tick_params(axis='x', rotation=45)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    pass
    #
    if savefig:
        if not os.path.exists(folder_figs):
            os.makedirs(folder_figs)
        pass
        figure_name = \
            (f"{prefix}_" if prefix is not None else "") + \
            f"{dataset_name}_comparison_sm_vs_ibnnx" + f"_{field_value}" + \
            f"_{display_mode}" + \
            ("_logx" if flag_log_x else "") + \
            ("_CI" if with_ci else "")
        for ext in ["eps", "png"]:
            complete_abs_name = os.path.join(folder_figs, figure_name + f".{ext}")
            plt.savefig(complete_abs_name, dpi=300, pad_inches=0.0)
        pass
    pass



#############################################################################################
# LOADING THE DF OF A PARTIAL DATASET EXPERIMENT, FILTERED, SO THE EXPERIMENTS ARE ORGANIZED HAVING THE COLUMNS 'acc', 'lambda', 'p' 't'
#############################################################################################


def load_partial_dataset_experiment_acc_for_t_lambda_p(experiment_label, method=None, flag_acc_100=False, **kwargs):
    """
    It loads the runs in the experiment indicated by `experiment_label` and according to the filters in `kwargs` \
    using the function :py:func:`.load_df_for_experiment_and_parameters` \
    (`kwargs` refers to the filters of the latter), and generates (through pivoting) a :py:class:`pandas.DataFrame` \
    having, as columns, $\\lambda$ ('lambda') and $p$ ('p') of the run/runs, the percentage ('t') of the dataset, \
    and the (validation) accuracy ('acc').
    If `method` is ``None`` the different seeds of the same experiment will not be grouped! Otherwise \
    they will be summarized/collapsed using the indicated method.

    Parameters
    -------
    experiment_label : str|int
        Experiment name/id
    method : str, optional
        Method among ['mean', 'max', 'min', 'median'].
        Default: ``None``
    flag_acc_100 : bool, optional
        Default: ``False``
    kwargs : dict
        Keyword arguments to be passed to :py:func:`.load_df_for_experiment_and_parameters`.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe having, per row, the 'n', 'p', 'lambda', and 'acc' of an experiment.
    """

    #
    ################################################
    # Check the summarization method and the columns_to_leave
    ################################################
    #
    if method is not None:
        assert isinstance(method, str) and method in ['mean', 'max', 'min', 'median'], \
            f"Unknown (summarization) method '{method}'!"
    pass
    print(f"Summarization method: {str(method)}")
    #
    ################################################
    # Load the experiment
    ################################################
    #
    df = load_df_for_experiment_and_parameters(experiment_label, **kwargs)

    ################################################
    # Process the dataset
    ################################################

    table = None
    if df is None or len(df) == 0:
        return df
    else:
        # Format columns
        #
        if 'metrics.train_proportion' in df.columns:
            df['t'] = df.apply(
                lambda x: (float(x['metrics.train_proportion']) if x['metrics.train_proportion'] is not None else 0.0),
                axis=1
            )
        else:
            df['t'] = 1.0
        pass
        #
        # Make the column 'lambda' host tuples (otherwise it would host other objects but not understand tuples)
        num_conv_layers = kwargs.get('num_conv_layers', None)
        value_for_lambda = (0.0,) if num_conv_layers is None else tuple([0.0] * num_conv_layers)
        new_col_lambda = pd.Series([value_for_lambda] * len(df))
        df['lambda'] = new_col_lambda
        if 'params.initial_lambda' in df.columns:
            df['lambda'] = df.apply(
                lambda x: (tuple(make_tuple(x['params.initial_lambda'])) if x['params.initial_lambda'] is not None else value_for_lambda),
                axis=1
            )
        pass
        #
        if 'params.sigma_x_compress' in df.columns:
            df['p'] = df.apply(lambda x:
                               (float(x['params.sigma_x_compress']) if x['params.sigma_x_compress'] is not None else 0.0),
                               axis=1)
        else:
            df['p'] = 0.0
        pass
        #
        if 'params.generator_seed' in df.columns:
            df['seed'] = df.apply(lambda x:
                               (int(x['params.generator_seed']) if x['params.generator_seed'] is not None else -1),
                               axis=1)
        else:
            df['seed'] = -1
        pass
        df.rename(columns={'metrics.best_acc': 'acc'}, inplace=True)
        #
        ################################################
        # Pivot the table and summarize using agg function, if desired
        ################################################
        extra_args = {'aggfunc': method} if method is not None else {}
        index=['t', 'lambda', 'p'] if method is not None else ['t', 'lambda', 'p', 'seed']

        table = pd.pivot_table(
            df, values='acc',
            index=index,
            **extra_args
        ).reset_index()

        ################################################
        # Add the complementary of the acc (err) and scale if desired
        ################################################
        table['err'] = 1.0 - table['acc']
        if flag_acc_100:
            table['acc'] = table['acc']*100
            table['err'] = table['err'] * 100
        pass
    pass

    return table


#############################################################################################


def plot_comparison_partial_dataset_results(list_tuples_total_df, dataset_name, value='acc',
                                            agg_method="median", with_ci:bool|int=False,
                                            ax=None, figsize=(6, 4),
                                            savefig=False, folder_figs=".", prefix=None):
    #
    assert value in ['acc', 'err'], f"Value must be 'acc' or 'err', {value} found!"
    #
    # fig_value, ax_value = plt.subplots(figsize=figsize)
    list_legend = []
    list_names_df = []
    #
    ci = _set_errorbar_for_plot(with_ci)
    #
    assert isinstance(agg_method,
                      str) and agg_method in _dict_estimators_lineplot, \
        f"Unknown 'agg_method' '{agg_method}'! It must be a string among 'mean', 'max', 'min', 'median'."
    #
    ax_value = ax
    if ax_value is None:
        fig_value, ax_value = plt.subplots(figsize=figsize)
    pass
    #
    for total_df, color_df, name_df in list_tuples_total_df:
        sns.lineplot(x=100 * total_df['t'], y=total_df[value], color=color_df,
                     estimator=_dict_estimators_lineplot[agg_method], errorbar=ci, ax=ax_value)
        list_legend.extend([name_df, None] if with_ci else [name_df])
        list_names_df.append(name_df)
    pass
    #
    #
    ax_value.set_xlabel('training dataset proportion (%)')
    ax_value.set_ylabel(f'{value} (%)')
    ax_value.grid(True)
    ax_value.legend(list_legend)
    ax_value.set_title(f"Mean {value} achieved per % of training dataset '{dataset_name}'" + \
                       (f" ({ci[0].upper()} {ci[1]})" if with_ci else ""))
    #
    if savefig:
        if not os.path.exists(folder_figs):
            os.makedirs(folder_figs)
        pass
        figure_name = f"{dataset_name}_comparison_sm_vs_ibnnx_{value}" + ("_CI" if with_ci else "")
        for ext in ["eps", "png"]:
            complete_abs_name = os.path.join(folder_figs, figure_name+f".{ext}")
            plt.savefig(complete_abs_name, dpi=300, pad_inches=0.0)
        pass
    pass


#############################################################################################
# LOADING THE DF OF A FGSM/PGD ATTACK
#############################################################################################


def load_aa_experiment_acc_for_lambda_p(experiment_label, attack_type,
                                                       num_conv_layers=None,
                                                       method=None, roi_lambdas_1st_layer:tuple=None,
                                                       flag_sm_to_lambdas=True, flag_acc_100=False, **kwargs):
    """
    Parameters
    ----------
    experiment_label
    attack_type
    num_conv_layers : int, optional
    method : str, optional
        Aggregation method among ['mean', 'max', 'min', 'median'] to be applied when summarizing.
        Default: ``None``, that is, no summarization, all seeds are kept.
    roi_lambdas_1st_layer : tuple[int|float], optional
        If provided, 2D tuple of int/floats indicating the range of interest for the lambdas of the first layer. \
        Default: ``None``, that is, no range of interest, all lambdas are kept
    flag_sm_to_lambdas : bool, optional
        Default: ``True``
    flag_acc_100 : bool, optional
        Default: ``False``
    kwargs

    Returns
    -------

    """
    #
    #
    ################################################
    # Check the attack to load
    ################################################
    #
    dict_lists_allowable_attacks = {'with_epsilon': ['pgd', 'fgsm'],
                                    'without_epsilon': ['pixle']}
    #
    assert isinstance(attack_type,
                      str), f"'attack_type' must be a string: {attack_type} of type {type(attack_type)} given."
    attack_type = attack_type.lower()
    #
    assert any([attack_type in dict_lists_allowable_attacks[key] for key in dict_lists_allowable_attacks]), \
        f"'attack_type' must an element in {dict_lists_allowable_attacks}: {attack_type} of type {type(attack_type)} given."
    #
    if num_conv_layers is not None:
        assert isinstance(num_conv_layers, int) and num_conv_layers > 0, \
            f"'num_conv_layers' must be, if provided, an int > 0; {num_conv_layers} provided!"
    #
    ################################################
    # Check the summarization method and the columns_to_leave
    ################################################
    #
    if method is not None:
        assert isinstance(method, str) and method in ['mean', 'max', 'min', 'median'], \
            f"Unknown (summarization) method '{method}'!"
    pass
    print(f"Summarization method: {str(method)}")
    #
    ################################################
    # Check the 'roi_lambdas_1st_layer'
    ################################################
    #
    if roi_lambdas_1st_layer is not None:
        assert isinstance(roi_lambdas_1st_layer, tuple) and len(roi_lambdas_1st_layer) == 2, \
            f"'roi_lambdas_1st_layer' must be a tuple of length 2: {roi_lambdas_1st_layer} provided!"
        assert all([isinstance(elem, (int,float)) for elem in roi_lambdas_1st_layer]), \
            f"'roi_lambdas_1st_layer' must be a tuple of int/floats: {roi_lambdas_1st_layer} provided!"
        assert roi_lambdas_1st_layer[0] < roi_lambdas_1st_layer[1], \
            f"The first element of 'roi_lambdas_1st_layer' must be smaller than the second one: {roi_lambdas_1st_layer} provided!"
        roi_lambdas_1st_layer = tuple([float(elem) for elem in roi_lambdas_1st_layer])
    pass
    #
    ################################################
    # Load the experiment and filter for the requested attack
    ################################################
    #
    df = load_df_for_experiment_and_parameters(experiment_label, num_conv_layers=num_conv_layers, **kwargs)
    if num_conv_layers is not None:
        df = df[df['metrics.num_conv_layers'] == num_conv_layers]
    df = df[df['params.attack_type'] == attack_type]

    ################################################
    # Process the dataset
    ################################################
    table = None
    if df is None or len(df) == 0:
        return df
    else:
        #
        # Extract also 'lambda_trainable'.
        # And for those corresponding to trainable lambdas, change 'lambda' for 'trainable'!
        df['lambda'] = df.apply(
            lambda x: \
                tuple(make_tuple(x['params.initial_lambda'])) if x['params.sigma_x_compress'] is not None else 0.0,
            axis=1
        )
        df['lambda'] = df['lambda'].astype('object')
        df['lambda_trainable'] = df.apply(lambda x: x['params.lambda_trainable'] == 'True', axis=1)
        df.loc[df['lambda_trainable'], 'lambda'] = 'trainable'
        #
        df['p'] = df.apply(lambda x:
                           (float(x['params.sigma_x_compress']) if x['params.sigma_x_compress'] is not None else 0.0),
                           axis=1)
        df['seed'] = df.apply(lambda x:
                              (int(x['params.generator_seed']) if x['params.generator_seed'] is not None else -1),
                              axis=1)
        #
        df.rename(columns={'params.conv_like_type': 'conv_like_type'}, inplace=True)
        df.rename(columns={'metrics.acc': 'clean_acc'}, inplace=True)
        df.rename(columns={'metrics.perturbed_acc': 'acc'}, inplace=True)
        #
        if attack_type in dict_lists_allowable_attacks['with_epsilon']:
            df.rename(columns={'metrics.attack_parameter_epsilon': 'epsilon'}, inplace=True)
        pass
        #
        ################################################
        # If indicated so, make the entries marked as 'sm' have zero lambda: if 'num_conv_layers', in fact, with that exact number of layers
        ################################################
        #
        if flag_sm_to_lambdas:
            if len(df[df['conv_like_type'] == 'sm']) > 0:
                ######################################################################
                # Usual problems: store "new" tuples in DF cells. Best: create a new Series where everything is a tuple
                ######################################################################
                value_to_enter = tuple([0.0] * num_conv_layers) if num_conv_layers is not None else 0.0
                list_values = [
                    value_to_enter if row['conv_like_type'] == 'sm' else tuple(row['conv_like_type']) \
                    for i, row in df.iterrows()
                ]
                new_col_lambda = pd.Series(list_values)
                df['lambda'] = new_col_lambda
            pass
        pass
        #
        ################################################
        # Now, for the 'with_epsilon' cases, all epsilons for the same rest of parameters have 'clean_acc' repeated
        # and, additionally, is listed as a column but not as entries with 'epsilon'=0:
        # we create new entries having it so
        ################################################
        #
        if attack_type in dict_lists_allowable_attacks['with_epsilon'] or \
                attack_type in dict_lists_allowable_attacks['without_epsilon']:
            df['flag_clean'] = False
            #
            clean_df = df.copy()
            clean_df['acc'] = clean_df['clean_acc']
            clean_df['flag_clean'] = True
            clean_df['epsilon'] = 0.0
            #
            df = pd.concat([df, clean_df])
            df.drop(columns=['clean_acc'], inplace=True)
        pass

        # if attack_type in dict_lists_allowable_attacks['with_epsilon']:
        #     clean_df = df.copy()
        #     clean_df['perturbed_acc'] = clean_df['clean_acc']
        #     clean_df['epsilon'] = 0.0
        #     #
        #     df = pd.concat([df, clean_df])
        # pass
        #
        ################################################
        # Pivot the table and summarize using agg function, if desired
        ################################################
        extra_args = {'aggfunc': method} if method is not None else {}
        if attack_type in dict_lists_allowable_attacks['with_epsilon']:
            index = ['conv_like_type', 'lambda', 'lambda_trainable', 'p', 'flag_clean', 'epsilon']
            values = ['acc']
        elif attack_type in dict_lists_allowable_attacks['without_epsilon']:
            index = ['conv_like_type', 'lambda', 'lambda_trainable', 'p', 'flag_clean']
            values = ['acc']
        pass
        if method is None:
            index.append('seed')
        pass
        # values = ['clean_acc', 'perturbed_acc']

        table = pd.pivot_table(
            df, values=values,
            index=index,
            **extra_args
        ).reset_index()

        ################################################
        # Add the complementary of the acc (err) and scale if desired
        ################################################
        for key in table.columns:
            if 'acc' in key:
                new_name = key.replace('acc', 'err')
                table[new_name] = 1.0 - table[key]
            pass
        pass
        if flag_acc_100:
            for key in table.columns:
                if 'acc' in key or 'err' in key:
                    table[key] = table[key] * 100
            pass
        pass
    pass

    ################################################
    # Filter using 'roi_lambdas_1st_layer' if provided
    # Careful: the trainable ones have its 'lambda' col to 'trainable'!
    ################################################
    if roi_lambdas_1st_layer is not None:
        series_1st_lambda = table.apply(
            lambda x: x['lambda'][0] if ~x['lambda_trainable'] else x['lambda'],
            axis=1
        )
        table = table[table['lambda_trainable'] | \
                      (series_1st_lambda >= roi_lambdas_1st_layer[0]) & (series_1st_lambda <= roi_lambdas_1st_layer[1])]
    pass

    return table


#############################################################################################


def plot_aa_experiment_acc_for_lambda(
        results_table, attack_type, flag_display_mode_only_first, dodge_bars:bool=True,
        agg_method="median", with_ci:bool|int=False,
        acc_lims:tuple|list=None, epsilon_lims:tuple|list=None,
        ax=None, title:str=None, savefig=False, folder_figures=None, base_name_figures=None):
    """
    Parameters
    ----------
    results_table : pandas.DataFrame
    attack_type : str
        'fgsm' or 'pgd'
    flag_display_mode_only_first : bool
        If ``True``, it only display the values corresponding to `results_table['lambda']` where only the first element\
        is different from 0.0, and takes the first element of the column `results_table['lambda']` \
        (if it is a tuple; otherwise complete) to order and display; \
        if ``False``, it takes the complete tuple/value of the column `results_table['lambda']`.
    dodge_bars : bool, optional
        (It applies only when results are a bar plot, like for Pixle attacks.) \
        If ``True``, bars of the two classes are placed by each other. \
        If ``False``, bars of the two classes are overlapped.
        Default: ``True``
    agg_method : str, optional
        Aggregation method.
        Default: 'median'
    with_ci : bool or int, optional
        Whether to include confidence intervals in the plot; if ``True``, \
        the confidence intervals will be set as ci = ('ci', 90). If an int (between 0 and 100), that is the percentile.
        Default: ``False`` or ``None`` (which means ci = None)
    acc_lims : tuple, optional
        (If not ``None``) A 2D tuple of floats indicating the lower/upper limits of the accs (y-axis) to display.
        Default: ``None``
    epsilon_lims : tuple, optional
        (If not ``None``) A 2D tuple of floats indicating the lower/upper limits of the epsilons (x-axis) to display.
        Default: ``None``
    ax : matplotlib.axes.Axes, optional
    title : str, optional
        If ``None``, a default title is provided.
        Default: ``None``
    savefig : bool, optional
        Default: ``False``
    folder_figures
    base_name_figures : str, optional
        ``None`` means "<attack_type>_first_layer.jpg/eps" if `flag_display_mode_only_first` is ``True`` \
        and "<attack_type>_successive.jpg/eps" if ``False``.
        Default: ``None``

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    #####
    # Checks
    #####
    dict_check = {'acc_lims': acc_lims, 'epsilon_lims': epsilon_lims}
    for key in dict_check:
        if dict_check[key] is not None:
            assert isinstance(dict_check[key], (tuple, list)) and len(dict_check[key]) == 2 and \
                   all([isinstance(elem, (int,float)) for elem in dict_check[key]]), \
                f"Arg {key} must be a 2D tuple of floats/ints; instead, {dict_check[key]} found!"
            assert dict_check[key][0] < dict_check[key][1], \
                f"Arg {key} must be contain a first element lower than its second; instead, {dict_check[key]} found!"
        pass
    pass
    #
    results_table_for_plot = results_table.copy()
    #
    var_lambdas = None
    label_lambdas = None
    #
    if flag_display_mode_only_first:
        # Remove the trainable options
        results_table_for_plot = results_table_for_plot[ ~results_table_for_plot['lambda_trainable'] ]
        print(f"Results corresponding to trainable lambda, if any, have been removed!")

        # Extract 'lambda_1'
        results_table_for_plot['lambda_1'] = results_table_for_plot.apply(
            lambda x: x['lambda'][0] if isinstance(x['lambda'], (tuple, list)) else x['lambda'], axis=1
        )

        # Leave only those indices where only the 2nd... and subsequent lambdas are 0.0
        indices_removal = results_table_for_plot.apply(
            lambda x: isinstance(x['lambda'], (tuple, list)) and any([elem != 0.0 for elem in x['lambda'][1:]]),
            axis=1
        )
        results_table_for_plot = results_table_for_plot[~indices_removal]
        #
        var_lambdas = 'lambda_1'
        label_lambdas = "$\\lambda_1$"
    else:
        # indices_true = results_table_for_plot.apply(lambda x : x['lambda'][0] in [0.0, -0.1], axis=1)
        # results_table_for_plot = results_table_for_plot[indices_true]
        #
        var_lambdas = 'lambda'
        label_lambdas = "$\\lambda$"
    pass
    #
    ci = _set_errorbar_for_plot(with_ci)
    #
    assert isinstance(agg_method, str) and agg_method in _dict_estimators_lineplot, \
        f"Unknown 'agg_method' '{agg_method}'! It must be a string among 'mean', 'max', 'min', 'median'."
    #
    # Transform the elements in the 'lambda' column to str for plotting
    # str_version_of_lambda = results_table_for_plot['lambda'].apply(lambda x: str(x))
    # results_table_for_plot = results_table_for_plot.copy()
    # results_table_for_plot['lambda'] = str_version_of_lambda
    # results_table_for_plot['lambda'] = results_table_for_plot['lambda'].apply(lambda x: str(x))
    #
    sns.set_style('whitegrid')
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    pass
    #
    # Force the order of the lambdas for better display
    order_lambda_axis = sorted(
        results_table_for_plot.loc[~results_table_for_plot['lambda_trainable'], var_lambdas].unique(), reverse=False)
    if not flag_display_mode_only_first:
        order_lambda_axis.append('trainable')
    #
    #
    if not attack_type.lower() in ['fgsm', 'pgd']:
        palette = "mako"
        attack_label = attack_type.capitalize()
        #
        x_axis = var_lambdas
        x_label = label_lambdas
        #
        results_table_for_plot['flag_clean_text'] = results_table_for_plot.apply(
            lambda x: 'Clean' if x['flag_clean'] else 'Perturbed', axis=1
        )
        order_hue_axis = ['Clean', 'Perturbed']
        #
        # sns.barplot(results_table_for_plot, x=x_axis, y='clean_acc', order=order_x_axis, errorbar=ci, ax=ax)
        # sns.barplot(results_table_for_plot, x=x_axis, y='acc', order=order_x_axis, errorbar=ci, ax=ax)
        sns.barplot(results_table_for_plot,
                    x=x_axis, order=order_lambda_axis, y='acc',
                    hue='flag_clean_text', hue_order=order_hue_axis, dodge=True,
                    estimator=_dict_estimators_lineplot[agg_method], errorbar=ci, palette=palette, ax=ax)
        #
        ax.set_xlabel(x_label)
        ax.set_ylabel("Acc (%)")
        #
        # If complete 'lambdas' and not only the first element, rotate the x ticks
        if not flag_display_mode_only_first:
            # ax.tick_params(axis='x', labelrotation=45)
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        pass
        #
        # Remove the title of the legend
        plt.legend(title="")
        # handles0, labels0 = ax.get_legend_handles_labels()
        # ax.legend(handles=handles0[0:], labels=labels0[0:])
    else:
        results_table_for_plot[var_lambdas] = results_table_for_plot.apply(lambda x: str(x[var_lambdas]), axis=1)
        ordered_lambda_strings = [str(elem) for elem in order_lambda_axis]

        #########################################
        # Build a custom palette and thickness
        #########################################
        sm_keys = ['0.0']
        if not flag_display_mode_only_first:
            for c in [1, 2, 3, 4, 5, 6]:
                sm_keys.append(str(tuple([0.0]*c)))
            pass
        pass
        sm_value_color = 'b'
        sm_value_style = (1, 0)
        #
        trainable_key = 'trainable'
        trainable_value_color = 'limegreen'
        trainable_value_style = (1, 0)
        #
        palette = "rocket_r"
        custom_palette = dict(zip(ordered_lambda_strings, sns.color_palette(palette, len(ordered_lambda_strings))))
        list_styles = [(1, 1), (3, 3), (3, 1, 1, 1), (1, 3)]
        custom_styles = dict(
            zip(ordered_lambda_strings,
                [list_styles[elem%len(list_styles)] for elem in range(len(ordered_lambda_strings))]
            )
        )
        custom_widths = {key: 2.0 for key in ordered_lambda_strings}
        for sm_key in sm_keys:
            custom_palette[sm_key] = sm_value_color
            custom_styles[sm_key] = sm_value_style
            custom_widths[sm_key] = 4.0
        pass
        custom_palette[trainable_key] = trainable_value_color
        custom_styles[trainable_key] = trainable_value_style
        custom_widths[trainable_key] = 4.0
        #########################################
        sns.lineplot(results_table_for_plot, x='epsilon', y='acc',
                     hue=var_lambdas, hue_order=ordered_lambda_strings, palette=custom_palette,
                     style=var_lambdas, dashes=custom_styles,
                     size=var_lambdas, sizes=custom_widths,
                     estimator=_dict_estimators_lineplot[agg_method], errorbar=ci, ax=ax)
        if epsilon_lims is not None:
            ax.set_xlim(epsilon_lims[0], epsilon_lims[1])
        pass
        #
        ax.set_xlabel("$\\epsilon$ (attack power)")
        ax.set_ylabel("Acc (%)")
        #
        ax.legend(title=label_lambdas)
    pass
    #
    if acc_lims is not None:
        ax.set_ylim(acc_lims[0], acc_lims[1])
    pass
    #
    if title is not None:
        assert isinstance(title, str), f"`title' must be a string; instead, {title}  of type {type(title)} provided!"
    else:
        title = f"Safety curves for attack {attack_type.upper()}" if attack_type.lower() in ['fgsm', 'pgd'] else \
            f"Clean acc. VS acc. under attack {attack_type.upper()}"
    pass
    # We always add the CI and print it
    title += (f" ({ci[0].upper()} {ci[1]})" if with_ci else "")
    ax.set_title(title)
    #
    if savefig:
        folder_figures = folder_figures if folder_figures is not None else "."
        os.makedirs(folder_figures, exist_ok=True)
        #
        base_name_figures = base_name_figures if base_name_figures is not None else \
            (attack_type + ("_first_layer" if flag_display_mode_only_first else attack_type + "_successive"))
        #
        for ext in ['jpg', 'eps']:
            filename = os.path.join(folder_figures, base_name_figures + f".{ext}")
            ax.get_figure().savefig(filename)
            print(f"Saved fig in {filename}!")
        pass
    pass
    #
    return ax