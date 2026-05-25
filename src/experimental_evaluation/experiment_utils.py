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

import copy
import datetime
import time
import errno
import os
import random

import traceback
import json
import socket
import time
import io
import shutil
import argparse
import typing
import warnings

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from PIL import Image

from collections import namedtuple

import torch
import torch.profiler

import torchattacks

from torchvision.transforms import v2 as v2

from torch import optim

import mlflow


from applications.classifiers import _dict_classifiers, _dict_classifiers_as_in_conf_file


from experimental_evaluation.configuration_file_reading_utils import get_multilevel_dict_element

from experimental_evaluation.configuration_file_reading_utils import (
    read_experiment_group_specification_file,
    duplicated_experiment_specification_removal,
)

from experimental_evaluation.interaction_with_mlflow import mlflow_log_dataset

from modified_rf.nn_layers import ModifiedRFLayer

##############################################################################################################
# load_dotenv()
##############################################################################################################



##############################################################################################################
##############################################################################################################
##############################################################################################################



#############################################################################################
# DATA STRUCTURES AND NAME GENERATING FUNCTIONS FOR LOGGING EXPERIMENTS
#############################################################################################

_training_progress_fields = {
    'epoch': int, 'batch': int, 'loss': float, 'acc': float
}

#############################################################################################

_datetime_str_format = "%Y%m%d-%H%M%S"


def formatted_log_base_name(current_date: datetime.datetime,
                             host=None, dataset_name=None, net_name=None,
                             extra_field=None, flag_random_id=False) -> str:
    #
    log_file_name = f"{current_date:{_datetime_str_format}}"
    import random
    random_id_str = f"{random.randint(0000, 9999):04d}" if flag_random_id else ""
    #

    for element in [host, dataset_name, net_name, random_id_str, extra_field]:
        if (element is not None) and (element != ''):
            log_file_name += f"_{element}"
    #
    return log_file_name



#############################################################################################
# EXCEPTION HANDLING FUNCTION
#############################################################################################


def exception_display_and_log(err, dict_individual_experiment_specification=None, mlflow_logging=False):
    """
    This function traces the origin of the error in the code, displaying it, and logs the error and the \
    experiment specification in MLFlow if requested and available (for later analysis).

    Parameters
    ----------
    err : Exception
    dict_individual_experiment_specification : dict, optional
        If provided, it is the dictionary with the individual experiment specification
        Default: ``None``
    mlflow_logging : bool, optional
        If ``True``, the experiment specification and the error are logged in MLFlow (in the active run, if any).
        Default: ``False``
    """

    print()
    print(f"\t!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"\t!!! ERROR in experiment!")
    print(f"\t!!! Error type: {type(err)}")
    err_frame_summary = traceback.extract_tb(err.__traceback__)[-1]
    print(f"\t!!! File: {err_frame_summary.filename};")
    print(f"\t!!! Function: {err_frame_summary.name}; line {err_frame_summary.lineno}")
    print(f"\t!!! Error: {err}")
    print(f"\t!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()
    error_msg = (
        f"Error in experiment! \tError type: {type(err)}; \tFile: {err_frame_summary.filename}; "
        f"\tFunction: {err_frame_summary.name}; line {err_frame_summary.lineno}; \tError: {err}")

    if mlflow_logging is not None and mlflow_logging == True and mlflow.active_run() is not None and \
            dict_individual_experiment_specification:
        mlflow.log_params(dict_individual_experiment_specification)
        json_string = json.dumps(dict_individual_experiment_specification, indent=4, default=lambda o: str(o))
        with open("experiment_specification.json", "w") as file:
            file.write(json_string)
        mlflow.log_artifact("experiment_specification.json")
        os.remove("experiment_specification.json")
        mlflow.log_param("error_message", error_msg)
    pass
pass



#############################################################################################
# TRAINING CORE
#############################################################################################

def log_param_histogram_to_mlflow(param_tensor, name, epoch=None):
    """
    Registra un histograma de un tensor de parámetros en MLflow como imagen.
    """
    import matplotlib.pyplot as plt
    import mlflow

    fig, ax = plt.subplots()
    ax.hist(param_tensor.detach().cpu().numpy().flatten(), bins=50)
    ax.set_title(f"Histograma de {name} (epoch {epoch})" if epoch is not None else f"Histograma de {name}")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    # Convertir a imagen PIL
    img = Image.open(buf)

    # Registrar la imagen PIL en MLflow
    mlflow.log_image(img, f"{name}_hist_epoch_{epoch}.png" if epoch is not None else f"{name}_hist.png")

    buf.close()


class ConvParamStatsTracker:
    """
    Tracker para estadísticas de parámetros m y b de capas convolucionales.
    """

    def __init__(self):
        self.stats = {}

    def update(self, layer_name, param_name, tensor):
        arr = tensor.detach().cpu().numpy().flatten()
        stats = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'tensor': tensor.detach().cpu().clone()
        }
        self.stats[(layer_name, param_name)] = stats

    def log_to_mlflow(self, epoch=None):
        import mlflow
        for (layer_name, param_name), stat in self.stats.items():
            prefix = f"epoch_{epoch}_" if epoch is not None else ""
            mlflow.log_metric(f"{prefix}{layer_name}_{param_name}_mean", stat['mean'])
            mlflow.log_metric(f"{prefix}{layer_name}_{param_name}_std", stat['std'])
            mlflow.log_metric(f"{prefix}{layer_name}_{param_name}_min", stat['min'])
            mlflow.log_metric(f"{prefix}{layer_name}_{param_name}_max", stat['max'])
            log_param_histogram_to_mlflow(stat['tensor'], f"{layer_name}_{param_name}", epoch=epoch)

    def reset(self):
        self.stats = {}


def collect_conv_param_stats(model, tracker):
    """
    Recorre el modelo y actualiza el tracker con estadísticas de m y b de capas convolucionales.
    """
    for name, module in model.named_modules():
        # Considera solo capas convolucionales personalizadas con atributos m y b
        if hasattr(module, 'm') and isinstance(module.m, torch.nn.Parameter):
            tracker.update(name, 'm', module.m)
        if hasattr(module, 'b') and isinstance(module.b, torch.nn.Parameter):
            tracker.update(name, 'b', module.b)


def classifier_training(classifier_network, dataset_dict, maximum_epochs,
                        loss_function, optimizer_class, optimizer_arg_dict=None,
                        scheduler_class=None, scheduler_arg_dict=None, epochs_sm_based_warmup=None,
                        adversarial_type=None, adversarial_proportion=1.0, adversarial_arg_dict=None,
                        early_stop_epochs=None,
                        validations_per_epoch=10, validation_on_test_subset=False, pre_validation=False,
                        random_initialization=True, random_initialization_gain:float=None,
                        mlflow_run_id=None, run_name=None,
                        local_log_folder=None,
                        verbose='medium'):
    """
    This function trains the classifier network `classifier_network` using the dataset `dataset_dict` for `maximum_epochs` \
    returning the best result of the training and the log information of the training and additionally storing such \
    information if indicated.

    The best model/instance/version of the input model is measured as the instance with the lowest validation loss.

    Parameters
    ----------
    classifier_network : :py:class:`torch.nn.Module`
        **Important**: It the argument network is already in GPU ('cuda'), the training will be performed  in GPU; \
        if in CPU ('cpu'), in CPU; this function does not internally move the model to a different device, it keeps \
        the model and the involved calculations in the same device of the `classifier_network` as provided
    dataset_dict : LoadedDatasetDict
    maximum_epochs : int
        Number of epochs if the training is not stopped before
    loss_function : ~collections.abc.Callable
        Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
        user-defined functions of *(1)* the output of the network and (2) the target can be used too
    optimizer_class : :py:class:`torch.optim.Optimizer`
        Optimizer linked to the trainable parameters of `classifier_network``
    optimizer_arg_dict : dict, optional
        Dictionary containing the key word arguments which, beyond the params of the network to optimize, \
        the optimizer `optimizer` to create would accept. Default: ``None``
    scheduler_class : :py:class:`torch.optim.lr_scheduler.LRScheduler` \
        or :py:class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, optional
        Scheduler object/function associated to `optimizer`. Regarding the list of schedulers,
        see the list in the point \
        `"How to adjust learning rate" <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ \
        of PyTorch's package ``torch.optim``. \
        Default: ``None`` (no scheduler)
    scheduler_arg_dict : dict, optional
        Dictionary containing the key word arguments which, beyond the optimizer, \
        the scheduler `scheduler` to create would accept. Default: `None`
    epochs_sm_based_warmup: int, optional
        For non-SM models, when not ``None``, this function first creates a network equivalent to the network to train \
        but with SM-layers in all its steps and performs `epochs_sm_based_warmup` epochs on it; after said warm-up is \
        finished all the "usable" weights are transferred to the network to train and the training resumes.
        Default: ``None`` (no SM-based warmup)
    adversarial_type : str, optional
        Type of adversarial attack to be applied to the training images. \
        If ``None`` no adversarial attack is applied. \
        If provided, it must be one of the types defined in \
        :py:func:`experimental_evaluation.adversarial_attacks.get_adversarial_attack` \
        (e.g. ``'fgsm'``, ``'pgd'``). \
        Default: ``None`` (no adversarial attack)
    adversarial_proportion: float, optional
        Proportion of images in each training batch to which the adversarial attack is applied. \
        If ``None`` or ``1.0`` all images in the training batch are adversarially perturbed. \
        If ``0.0`` no image in the training batch is adversarially perturbed. \
        Default: ``1.0``
    adversarial_arg_dict : dict, optional
        Dictionary containing the key word arguments which, beyond the adversarial type, \
        the adversarial attack `adversarial` to create would accept. \
        Default: ``None`` (no adversarial attack)
        If `adversarial_type` is provided, this argument must be provided too, \
        otherwise an exception is raised.
        If `adversarial_type` is provided, the adversarial attack is applied to the training images \
        before the training starts, and the adversarial attack is applied to the training images.
    early_stop_epochs : int, optional
        The training is early-stopped before the indicated ``maximum_epochs`` if the average validation loss \
        during an epoch has not improved for ``early_stop_epochs``. If ``None`` or ``0`` provided, no early stop. \
        Default: ``None``
    validations_per_epoch : int, optional
        Number of times, equally spaced, that the loss/accuracy is compared to the *val(idation)* subset per epoch. \
        **Warning**: the validation is calculated for the complete *val(idation)* subset: if big it can represent \
        are significant overload in the training process.
        Default: ``10``
    validation_on_test_subset : bool, optional
        Whether the validations are to be performed on the test subset of the dataset instead of on the validation \
        subdataset used by default for the periodic validations.
        Default: ``False`` (i.e. they are performed on the val subset)
    pre_validation : bool, optional
        Whether a validation run prior to any training is to be performed. This pre_validation is simply run and \
        printed but not logged.
    random_initialization: bool, optional
        Whether the network is to be randomly initialized before the training starts. Additionally, if ``True`` and \
        the network has, included, a prenormalization layer integrated in it, the prenormalization layer is adjusted \
        to the statistics of the dataset.
        Default: ``True``
    random_initialization_gain: float, optional
        If `random_initialization` is ``True``, this parameter is the gain (std, not variance) of the initialization \
        noise of the weights of the network. When `random_initialization_gain` is not provided but \
        `random_initialization` is ``True`` the default gain of the classifier is used.
        Default: ``None``
    mlflow_run_id : str, optional
        If provided, it is the ID of the MLFlow run, already started, to which the current experiment will be logged. \
        Said run must exist and be active. If ``None`` no experiment where the run is to be stored. \
        If ``None`` the new run is a top-level run and will not refer to any previous run. \
        Default: ``None``
    run_name : str, optional
        Name of the run, to be used in the local logging functions if necessary, in case `mlflow_run_id` is not \
        provided and therefore the run name cannot be obtained from it; if a valid MLFlow \
        run is provided, and `run_name` is also provided, an exception is raised. \
        In the case local logging is requested and no run name is provided, neither through `mlflow_run_id` nor \
        in the form of a directly provided `run_name`, a name for the experiment related to the host machine, \
        start time of the training, and certain attributes of the classifier, will be automatically generated.
        Default: ``None``
    local_log_folder : str, optional
        Local folder where the experiment-specific folder (whose name corresponds to the run name either \
        obtained through `mlflow_run_id` or directly provided `run_name` or automatically generated) \
        where the best model and Pandas-based logs are to be stored. (If no ``local_log_folder`` \
        is provided a default temporal folder is used for practical reasons and deleted after the execution.)
        Default: ``None``
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of printing the training progress. \
        Default: ``'medium'``

    Returns
    -------
    tuple

        **Named tuple** (see :py:func:`collections.namedtuple`) with the fields:

        - '''experiment_folder''': :py:class:`str`

        - ``'best_model'`` : :py:class:`torch.nn.Module`

        - ``'best_loss'`` : :py:class:`float`

        - ``'best_acc'`` : :py:class:`float`

        - ``'pandas_series_epoch_for_acc'`` : :py:class:`pandas.series` where the index is the acc %

        - ``'pandas_df_log'`` : :py:class:`pandas.DataFrame`
    """

    #########
    # Initial setting for original 'classifier_network' versus SM-based surrogate
    #########
    dict_classifier_networks = {
        'original': classifier_network,
        'sm_surrogate': None
    }
    dict_optimizers = {}
    dict_schedulers = {}
    dict_attacks = {}
    #########
    total_epochs = maximum_epochs if epochs_sm_based_warmup is None else epochs_sm_based_warmup+maximum_epochs
    epoch_offset = 0 if epochs_sm_based_warmup is None else epochs_sm_based_warmup
    if epochs_sm_based_warmup is not None:
        assert isinstance(epochs_sm_based_warmup, int) and 0<=epochs_sm_based_warmup<maximum_epochs, \
            f"'epochs_sm_based_warmup' must be either 'None', or an int between 1 and {maximum_epochs}; " + \
            f"{epochs_sm_based_warmup} found!"
        epochs_sm_based_warmup = epochs_sm_based_warmup if epochs_sm_based_warmup > 0 else None
        
        # If the network is already SM-based... set to None
        # if classifier_network.get_extra_state()['conv_like_type']=='sm' and epochs_sm_based_warmup is not None:
        #     warnings.warn((
        #         f"Although 'epochs_sm_based_warmup' different from 'None' ({epochs_sm_based_warmup}) provided, " +
        #         f"the network is SM-based and therefore the training will be effectively done without " +
        #         f"specific SM-surrogate-based warm-up."
        #     ))
        #     epochs_sm_based_warmup = None
        # pass
    pass

    if verbose in ['medium', 'high']:
        if verbose== 'high':
            print(f"----------------------------------------------------------------------------------")
            print(f"Training of the classifier network:")
            print(f"----------------------------------------------------------------------------------")
            print(str(classifier_network))
        else:
            list_of_modules_to_print = []
            for name, module in classifier_network.named_modules():
                if isinstance(module, ModifiedRFLayer):
                    list_of_modules_to_print.append(type(module).__name__)
                pass
            pass
            print(f"----------------------------------------------------------------------------------")
            print("Training of the classifier network containing hidden layers of type " +
                  " + ".join(list_of_modules_to_print))
            print(f"----------------------------------------------------------------------------------")
        pass
    pass

    if verbose == 'high':
        attributes_to_display = classifier_network.fields_to_log
        print(f"\n\tFields to log:\n")
        for key in attributes_to_display:
            print(f"\t\t{key}: {attributes_to_display[key]}")
        pass
    pass

    ###########################
    # Setting the logging intervals for experiment logging
    ###########################

    subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'

    # Batch indices where evaluations are to be calculated
    tensor_num_plus_1 = torch.linspace(
        0, len(dataset_dict['dataloader_train']) - 1, validations_per_epoch + 1, dtype=torch.int32
    )
    list_ind_batch_for_validation = tensor_num_plus_1[1:].flatten().tolist()

    list_final_validation_loss_across_epochs = []
    list_final_validation_acc_across_epochs = []

    tic_start_training = time.time()

    ###########################
    # Parameters of potential special interest for logging
    ###########################

    start_training = datetime.datetime.now()
    end_training = None

    host = socket.gethostname()
    net = classifier_network.fields_to_log['net']
    conv_like_type = classifier_network.fields_to_log['conv_like_type']

    ###########################
    # Definition, or load, of the name defining the experiment/run
    ###########################

    if mlflow_run_id is not None:
        if run_name is not None:
            raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
                            "only one of them can be provided at a time!")
        else:
            # Check if the run with the provided ID is the active run
            if mlflow.active_run() is not None and mlflow.active_run().info.run_id == mlflow_run_id:
                # Great, the active run is the provided run
                pass
            else:
                # Re-start the MLFlow run corresponding to the provided ID
                mlflow.start_run(run_id=mlflow_run_id)
            pass
            # And obtain the name of the run name
            run_name = mlflow.active_run().info.run_name
        pass
    elif run_name is None:
        # If no MLFlow run is provided, and no name is provided, we generate one
        run_name = formatted_log_base_name(start_training, host=host, dataset_name=dataset_dict['dataset_name'],
                                            net_name=net, extra_field=f"{conv_like_type}")
    pass

    ###########################
    # Pre-setting the folders, filenames, classes, or Pandas structures for local experiment logging
    ###########################
    # The idea is: we will always store locally the best model and the Pandas, independently of the "local_log_folder".
    # if the user did not provide any folder, the folder will be a temporal one and will be deleted;
    # if the folder was provided, it remains.
    ###########################

    # When no log folder is provided, we create a temporal one (indicated with a flag) and delete it at the end
    flag_temp_log_folder = False
    if local_log_folder is None:  # NOWHERE TO STORE LOGS: ERROR IF ASKED OTHERWISE
        flag_temp_log_folder = True
        local_log_folder = "./tmp/tmp_" + run_name
    pass

    # We get ready structures and folder and file names for storage, even for those not bound to be used in the end

    rel_log_file_name = run_name
    abs_experiment_folder = os.path.join(local_log_folder, rel_log_file_name)
    for _ in range(10):
        if not os.path.exists(abs_experiment_folder):
            # Sometimes, after checking that the dir exists (negatively)... then the folder ALREADY exists
            # (race condition). Check and recheck if so.
            try:
                os.makedirs(abs_experiment_folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                else:
                    # This means that, although existence check for the folder was negative... now it is true!
                    # Recheck... for a number of times
                    time.sleep(10)
                    continue
                pass
            pass
        else:
            break
        pass
    pass
    if not os.path.exists(abs_experiment_folder):
        raise Exception(f"Experiment folder '{abs_experiment_folder}' could not be created!!!")
    pass
    #
    # For Pandas
    abs_log_file_name_pandas_pkl = os.path.join(local_log_folder, rel_log_file_name, "pandas.pkl")
    abs_log_file_name_pandas_csv = os.path.join(local_log_folder, rel_log_file_name, "pandas.csv")
    #
    # For the model
    abs_log_file_name_state_dict = os.path.join(local_log_folder, rel_log_file_name, "state_dict.pt")
    #
    # For a text summary of the stored model
    abs_log_file_name_summary = os.path.join(local_log_folder, rel_log_file_name, "summary.txt")

    # def get_model_size(model):
    #     param_size = 0
    #     for param in model.parameters():
    #         param_size += param.nelement() * param.element_size()
    #     buffer_size = 0
    #     for buffer in model.buffers():
    #         buffer_size += buffer.nelement() * buffer.element_size()
    #     size_all_mb = (param_size + buffer_size) / 1024 ** 2
    #     return size_all_mb

    # Plot the model size
    # classifier_network = torch.compile(classifier_network)
    # model_size_mb = get_model_size(classifier_network)
    # print(f"Model size: {model_size_mb:.2f} MB")

    ###########################
    # Creation of the Pandas DF for logging of the parameters of the model+training and the progress
    ###########################

    # First, we create the bare DataFrame with the figures of merit per epoch-batch
    #
    # The rows will be a multi-index with the epoch number, the batch number and the absolute batch number.
    row_multi_index = pd.MultiIndex.from_product(
        [list(range(1, total_epochs + 1)), [batch + 1 for batch in list_ind_batch_for_validation]],
        names=["epoch", "batch"]
    )
    df_for_row_multi_index = row_multi_index.to_frame().reset_index(drop=True)
    df_for_row_multi_index['abs_batch'] = df_for_row_multi_index['batch'] + \
                                          (df_for_row_multi_index['epoch'] - 1) * dataset_dict['num_batches_train']
    df_for_row_multi_index['epoch_fraction'] = df_for_row_multi_index['abs_batch'] / dataset_dict['num_batches_train']
    row_multi_index = pd.MultiIndex.from_frame(df_for_row_multi_index,
                                               names=["epoch", "batch", "abs_batch", "epoch_fraction"])

    # The columns will be a multi-index with the subsets and the figures of merit;
    col_multi_index = pd.MultiIndex.from_product([['train', 'val'], ['loss', 'acc']],
                                                 names=["subset", "figure_of_merit"])
    # The DF, from them
    df_training_progress = pd.DataFrame(index=row_multi_index, columns=col_multi_index, dtype=float)

    # Separate the 'abs_batch' and 'epoch_fraction' from the index and a new column 'elapsed_time'
    df_training_progress.reset_index(level=['abs_batch', 'epoch_fraction'], inplace=True)
    # And add the column "elapsed_time"
    df_training_progress['elapsed_time'] = np.nan
    df_training_progress['elapsed_time'] = df_training_progress['elapsed_time'].astype("timedelta64[s]")

    # The rest of fields regarding the general settings of the test will be pushed in the form of multi-index
    # by the end of the training

    # df_val_loss_within_epoch = []
    # df_val_acc_within_epoch = []
    # df_epoch_summary = []

    ###########################
    ###########################
    # Preparation of the optimization-related aspects of the training, INCLUDING SURROGATE IF NECESSARY
    ###########################
    ###########################

    #####
    # If SM-based surrogate is required!
    #####
    sm_based_surrogate_classifier_network = None
    if epochs_sm_based_warmup is not None:
        # Constructor of the original model model
        constructor_kwargs = dict_classifier_networks['original'].constructor_kwargs
        constructor_kwargs['conv_like_type'] = 'sm'
        print(f"----------------------------------------------------------------------------------")
        print(f"| CONSTRUCTING A SM-BASED SURROGATE FOR A {epochs_sm_based_warmup}-epoch WARM-UP...")
        dict_classifier_networks['sm_surrogate'] = \
            type(dict_classifier_networks['original'])(**constructor_kwargs)
        print(f"| ... SM-BASED SURROGATE FOR A {epochs_sm_based_warmup}-epoch WARM-UP: CONSTRUCTED!")
        print(f"| TRANSFERRING THE INITIAL WEIGHTS OF THE ORIGINAL...")
        dict_classifier_networks['sm_surrogate'].load_state_dict(
            dict_classifier_networks['original'].state_dict(),
            strict=False
        )
        print(f"| ... INITIAL WEIGHTS OF THE ORIGINAL: TRANSFERRED!")
        print(f"----------------------------------------------------------------------------------")
        if verbose in ['high']:
            print(f"Training of the surrogate network:\n")
            print(str(dict_classifier_networks['sm_surrogate']))
        pass
    pass

    #####
    # Optimizer
    #####
    for key in dict_classifier_networks:
        dict_optimizers[key] = None
        if dict_classifier_networks[key] is not None:
            if optimizer_arg_dict is not None:
                dict_optimizers[key] = optimizer_class(dict_classifier_networks[key].parameters(), **optimizer_arg_dict)
            else:
                dict_optimizers[key] = optimizer_class(dict_classifier_networks[key].parameters())
            pass
        pass
    pass

    #####
    # Scheduler (if any provided)
    #####
    for key in dict_classifier_networks:
        dict_schedulers[key] = None
        if dict_classifier_networks[key] is not None:
            if scheduler_class is not None:
                if scheduler_arg_dict is not None:
                    dict_schedulers[key] = scheduler_class(dict_optimizers[key], **scheduler_arg_dict)
                else:
                    dict_schedulers[key] = scheduler_class(dict_optimizers[key])
                pass
            pass
        pass
    pass

    #####
    # Adversarial attack (if any provided)
    #####
    for key in dict_classifier_networks:
        dict_attacks[key] = None
        if dict_classifier_networks[key] is not None:
            if adversarial_type is not None:
                if adversarial_arg_dict is None:
                    raise Exception("Adversarial attack type provided but no arguments provided for it!")
                pass
                # Check that the adversarial type is valid
                if adversarial_type.lower() == 'fgsm':  # We use RFGSM to make sure it has random start
                    dict_attacks[key] = torchattacks.RFGSM(dict_classifier_networks[key], **adversarial_arg_dict)
                elif adversarial_type.lower() == 'pgd':
                    adversarial_arg_dict['random_start'] = True
                    dict_attacks[key] = torchattacks.PGD(dict_classifier_networks[key], **adversarial_arg_dict)
                pass
                # Check that the proportion is valid
                if not isinstance(adversarial_proportion, (int, float)) or (adversarial_proportion < 0.0) or (
                        adversarial_proportion > 1.0):
                    raise Exception("Adversarial proportion must be an int/float between 0.0 and 1.0!")
                pass
            pass
        pass
    pass

    ###########################
    # Pack the information about the optimization, the optimizer, and scheduler for logging
    ###########################

    dict_optimizer_and_scheduler_params = {}
    dict_optimizer_and_scheduler_params['subset_for_validation'] = 'test' if validation_on_test_subset else 'val'
    dict_optimizer_and_scheduler_params['maximum_epochs'] = maximum_epochs
    dict_optimizer_and_scheduler_params['loss_function'] = loss_function.__name__
    dict_optimizer_and_scheduler_params['epochs_sm_based_warmup'] = epochs_sm_based_warmup \
        if epochs_sm_based_warmup is not None else 'None'
    dict_optimizer_and_scheduler_params['early_stop_epochs'] = early_stop_epochs \
        if early_stop_epochs is not None else 'None'
    dict_optimizer_and_scheduler_params['optimizer'] = optimizer_class.__name__ \
        if optimizer_class is not None else 'None'
    dict_optimizer_and_scheduler_params['optimizer_args'] = str(optimizer_arg_dict) \
        if optimizer_arg_dict is not None else 'None'
    dict_optimizer_and_scheduler_params['scheduler'] = scheduler_class.__name__ \
        if scheduler_class is not None else 'None'
    dict_optimizer_and_scheduler_params['scheduler_args'] = str(scheduler_arg_dict) \
        if scheduler_arg_dict is not None else 'None'

    ###########################
    # Auxiliary variables for acc and loss assessment in training and validation
    ###########################

    train_loss = None
    train_acc = None
    val_loss = None
    val_acc = None

    best_loss = np.inf
    best_acc = 0.0
    best_loss_adv = None
    best_acc_adv = None
    best_model = copy.deepcopy(classifier_network.state_dict())
    ###########################
    proportion_non_convergent_best_model = None
    ###########################
    # Value of max norm for gradient clipping
    max_norm = 1.0
    ###########################

    ###########################
    # Inicializar trackers de estadísticas
    ###########################
    stats_tracker = LayerStatsTracker()
    conv_param_tracker = ConvParamStatsTracker()
    result_classifier_training = None

    ###########################
    # Mark the beginning of the training to log the abs time (elapsed) to each validation point
    ###########################
    tic_beginning_training = time.time()

    ###########################
    # Apply the dataset statistics to the prenormalization layer, if existing, and if "random_initialization" is True.
    # Although random initialization is a different thing, it indicates when initialization of the trainable layers
    # is needed: when no "random_initialization" is needed, no prenormalization initialization is needed either.
    ###########################

    # Prenormalization of the network
    if random_initialization:
        for key in dict_classifier_networks:
            if dict_classifier_networks[key] is not None:
                flag_achieved = \
                    dict_classifier_networks[key].prenormalization_initialization_to_dataset_statistics(dataset_dict)
                if verbose in ['high', 'medium', 'low']:
                    print(f"Prenormalization layer initialized with the dataset statistics " +
                          f"{'WAS' if flag_achieved else 'NOT'} performed for the {key} network.")
                pass
            pass
        pass
    else:
        if verbose in ['high', 'medium', 'low']:
            print(f"Dataset-based initialization of the (potential) prenormalization layer has NOT been requested.")
        pass
    pass

    ###############
    # Randomly initialize the network, if so requested and epoch is 0!
    ###############

    if random_initialization:
        #
        dict_random_initialization_gain = {}
        if random_initialization_gain is None:
            pass
        elif isinstance(random_initialization_gain, (float, int)) and (random_initialization_gain > 0):
            dict_random_initialization_gain = {'gain': float(random_initialization_gain)}
        else:
            raise Exception(
                (f"'random_initialization_gain', if provided (not None), must be an int/float > 0: " +
                 f"{random_initialization_gain} (type {type(random_initialization_gain)}) found!"))
        pass
        #
        if verbose in ['high', 'medium', 'low']:
            print(
                (f"Random initialization of the trainable layers HAS BEEN requested" +
                 (
                     "." if random_initialization_gain is None else f" with gain {float(random_initialization_gain)}")
                 )
            )
        pass
        #
        # Randomly initialize the network(s) weights
        for key in dict_classifier_networks:
            if dict_classifier_networks[key] is not None:
                dict_classifier_networks[key].random_initialization(**dict_random_initialization_gain)
            pass
        pass
    else:
        if verbose in ['high', 'medium', 'low']:
            print(f"Random initialization of the trainable layers has NOT been requested.")
        pass
    pass

    ###########################
    # PERFORM THE LOGS WHICH CAN BE PERFORMED AT THIS TIME: e.g. dataset info, adversarial training info
    ###########################

    # Log the dataset info
    if mlflow_run_id is not None:
        mlflow_log_dataset(dataset_dict, mlflow_run_id)
    pass

    # Log the adversarial training info
    mlflow.log_param("adversarial_type", str(adversarial_type))

    if adversarial_type is not None:
        # "adversarial_proportion", both as metric and param
        mlflow.log_param("adversarial_proportion", str(adversarial_proportion))
        mlflow.log_metric("adversarial_proportion", adversarial_proportion)
        # adversarial_arg_dict, as params
        if isinstance(adversarial_arg_dict, dict):
            for key in adversarial_arg_dict.keys():
                mlflow.log_param("adversarial_training_"+str(key), str(adversarial_arg_dict[key]))
                if isinstance(adversarial_arg_dict[key], (int, float)):
                    mlflow.log_metric("adversarial_training_"+str(key), adversarial_arg_dict[key])
                pass
            pass
        pass
    pass

    ###########################
    # If pre-validation: before randomizing or training the network, calculate validation!
    ###########################

    # HERE: 'classifier_network' IS THE ORIGINAL NETWORK!
    classifier_network = dict_classifier_networks['original']

    if pre_validation:
        #
        epoch = -1
        #
        if verbose in ['high', 'medium', 'low']:
            print(f"EPOCH [{(epoch + 1):3d}/{total_epochs:3d}] --- PRE-VALIDATION")
        pass
        #
        # TIC block_between_validations: reset
        tic_block_between_validations = time.time()
        #
        # Set the network in evaluation mode and start a .no_grad() scope!
        classifier_network.eval()
        # Registrar hooks para todas las capas
        register_stats_hooks(classifier_network, stats_tracker)
        # Registrar estadísticas de parámetros m y b
        collect_conv_param_stats(classifier_network, conv_param_tracker)

        with (torch.no_grad()):
            #
            cum_val_images = 0
            cum_non_convergent_images = 0
            cum_val_loss_times_num_images = 0.0
            cum_val_right_classifications = 0
            #
            for val_batch, (val_images_batch, val_labels_batch) in enumerate(dataset_dict[subset_for_validation]):
                #
                # TIC val_batch
                tic_val_batch = time.time()
                #
                base_string_justifying_skipped_val_batch = \
                    f"\t(Validation batch {val_batch + 1} skipped. Reason: %s)"
                #
                flag_any_problem_encountered_in_validation = True
                #
                output_val_images_batch = classifier_network(val_images_batch)
                #
                # Statistics regarding convergence
                _, warning_lack_of_convergence, _, _ = classifier_network.get_last_forward_convergence_info()
                cum_non_convergent_images += warning_lack_of_convergence.sum(dim=None).item()

                if torch.any(torch.isnan(output_val_images_batch)):
                    print(base_string_justifying_skipped_val_batch % 'forward pass result contains NaN')
                elif torch.any(torch.isinf(output_val_images_batch)):
                    print(base_string_justifying_skipped_val_batch % 'forward pass result contains Inf')
                else:
                    # ... and calculate the loss (in 'cuda', if justified)
                    device_for_comparison = 'cuda' \
                        if (output_val_images_batch.device.type == 'cuda') \
                           or (val_labels_batch.device.type == 'cuda') \
                        else 'cpu'
                    # Validation LOSS
                    val_loss_i = loss_function(output_val_images_batch.to(device_for_comparison),
                                               val_labels_batch.to(device_for_comparison))

                    if torch.any(torch.isnan(val_loss_i)):
                        print(base_string_justifying_skipped_val_batch % "calculated loss is/contains NaN")
                    elif torch.any(torch.isinf(val_loss_i)):
                        print(base_string_justifying_skipped_val_batch % "calculated loss is/contains Inf")
                    else:
                        # Since 'val_loss_i' is probably averaged, we un-average it and accumulate it
                        cum_val_loss_times_num_images += val_loss_i.item() * val_images_batch.size(0)
                        cum_val_images += val_images_batch.size(0)
                        #
                        # Validation ACCURACY (multiclass classification)
                        output_val_labels_batch = torch.max(output_val_images_batch, dim=-1)[1]
                        cum_val_right_classifications += (
                                output_val_labels_batch.to(device_for_comparison) == val_labels_batch.to(
                            device_for_comparison)
                        ).sum().item()
                        #
                        # TOC val_batch
                        elapsed_val_batch = time.time() - tic_val_batch
                        if verbose == 'high':
                            print((f"    " +
                                   f"Val batch   [{(val_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]:   " +
                                   f"batch loss = {val_loss_i:11.4f},   " +
                                   f"batch time = {elapsed_val_batch:.4f} s " +
                                   f"(h:m:s " + str(
                                        datetime.timedelta(seconds=round(elapsed_val_batch))
                                    ) + ", " +
                                   f"avg {elapsed_val_batch / val_images_batch.size(-4):.3f} s/im)"),
                                  end='\r')
                        pass
                        #
                        flag_any_problem_encountered_in_validation = False
                        #
                    pass
                pass
                #
            pass  # for val_batch, (val_images_batch, val_labels_batch) in enumerate(dataset_dict[subset_for_validation]):

        pass  # with (torch.no_grad()):

        # Set the network back in training mode
        classifier_network.train()

        val_loss = cum_val_loss_times_num_images / cum_val_images if cum_val_images > 0 else np.nan
        val_acc = cum_val_right_classifications / cum_val_images if cum_val_images > 0 else np.nan

        proportion_non_convergent = cum_non_convergent_images / cum_val_images

        # TOC train_batch
        elapsed_block_between_validations = time.time() - tic_block_between_validations

        num_images_between_validations = dataset_dict['batch_size'] * (
                list_ind_batch_for_validation[1] - list_ind_batch_for_validation[0]
        ) if len(list_ind_batch_for_validation) > 1 else list_ind_batch_for_validation[0]  # If only 1 validation

        if verbose in ['high', 'medium', 'low']:
            print((f"    " +
                   f"Val @ epoch [{(epoch + 1):3d}/{total_epochs:3d}]:   " +
                   f"loss = {val_loss:11.4f},   " +
                   f"acc = {100 * val_acc:7.3f} %,   " +
                   f"non-converged images = {100 * proportion_non_convergent:7.3f} %   " +
                   f"(block between validations, train+val, h:m:s " + str(
                        datetime.timedelta(seconds=round(elapsed_block_between_validations))
                    ) + ")"
                   ), end='\n')
        pass
        #
        # MLFlow logging
        if mlflow_run_id is not None:
            mlflow.log_metrics({'val_loss': val_loss, 'val_acc': val_acc}, step=epoch)
        pass
        #
        # MLFlow logging
        if mlflow_run_id is not None:
            # TOC beginning_training
            elapsed_since_beginning_training = time.time() - tic_beginning_training
            mlflow.log_metrics({
                'val_loss': val_loss, 'val_acc': val_acc, 'proportion_non_convergent': proportion_non_convergent,
                'elapsed_time': elapsed_since_beginning_training, 'epoch_fraction': 0.0
            }, step=0, run_id=mlflow_run_id)
            conv_param_tracker.log_to_mlflow(epoch=0)
        pass
        conv_param_tracker.reset()
    pass  # if pre_validation:

    ###########################
    # Start the training!
    ###########################

    if maximum_epochs > 0:

        global_batch = 0

        #####
        # Set the classifier, optimizer, scheduler, and attack to be used at the beginning
        #####
        current_model_key = 'sm_surrogate' if dict_classifier_networks['sm_surrogate'] is not None else 'original'
        if current_model_key=='sm_surrogate':
            print(f"----------------------------------------------------------------------------------")
            print(f"| START THE TRAINING USING THE SM-BASED SURROGATE!")
            print(f"----------------------------------------------------------------------------------")
        pass
        classifier_network = dict_classifier_networks[current_model_key]
        optimizer = dict_optimizers[current_model_key]
        scheduler = dict_schedulers[current_model_key]
        attack = dict_attacks[current_model_key]

        # TIC block_between_validations
        tic_block_between_validations = time.time()

        epoch = 0
        while epoch < total_epochs: # INTENDED: for epoch in range(total_epochs)

            cum_train_images = 0
            #
            cum_train_loss_times_num_images = 0.0
            cum_train_right_classifications = 0

            ###################
            # CHANGE THE CURRENT classifier_network, optimizer, scheduler, attack IF SURROGATE AND DUE TIME
            #####
            # AND RESET EPOCH
            ###################

            if dict_classifier_networks['sm_surrogate'] is not None and epochs_sm_based_warmup == epoch:
                #
                #####
                # Change the "target" classifier network!
                #####
                print(f"----------------------------------------------------------------------------------")
                print(f"| TRANSFERRING THE WEIGHTS FROM SURROGATE TO ORIGINAL MODEL...")
                dict_classifier_networks['original'].load_state_dict(
                    copy.deepcopy(dict_classifier_networks['sm_surrogate'].state_dict()),
                    strict=False
                )
                print(f"| WEIGHTS FROM SURROGATE TO ORIGINAL MODEL: TRANSFERRED!")
                print(f"----------------------------------------------------------------------------------")
                print(f"| CONTINUE WITH THE WEIGHT-TRANSFERRED ORIGINAL NETWORK!")
                print(f"----------------------------------------------------------------------------------")
                current_model_key = 'original'
                classifier_network = dict_classifier_networks[current_model_key]
                optimizer = dict_optimizers[current_model_key]
                scheduler = dict_schedulers[current_model_key]
                attack = dict_attacks[current_model_key]
                #
                old_model_key = 'sm_surrogate'
                dict_optimizers[old_model_key].zero_grad()
                dict_classifier_networks[old_model_key] = None
                dict_optimizers[old_model_key] = None
                dict_schedulers[old_model_key] = None
                dict_attacks[old_model_key] = None
            pass
            #
            classifier_network.train()
            #
            # TIC train_epoch
            tic_train_epoch = time.time()
            
            if verbose in ['high', 'medium']:
                lr_current = optimizer.state_dict()['param_groups'][0]['lr']
                print(f"EPOCH [{(epoch + 1):3d}/{total_epochs:3d}] --- Current LR = {lr_current:.4f}")
            pass
            #
            list_validation_loss_within_epoch = []
            list_validation_acc_within_epoch = []
            #
            for train_batch, (train_images_batch, train_labels_batch) in enumerate(dataset_dict['dataloader_train']):
                # Print the current batch size in MB and the shape of the batch
                # print(f"\nTrain batch [{(train_batch + 1):5d}/{dataset_dict['num_batches_train']:5d}] "
                #       f"with {train_images_batch.size(-4)} images, "
                #       f"shape {train_images_batch.shape}, "
                #       f"labels shape {train_labels_batch.shape}")
                # batch_size_mb = train_images_batch.element_size() * train_images_batch.nelement() / (1024 ** 2)
                # print(f"Batch size: {batch_size_mb:.2f} MB")
                # TIC train_batch
                tic_train_batch = time.time()

                ##############################################################################
                # USE A TRAIN BATCH TO UPDATE THE PARAMETERS OF THE NETWORK
                ##############################################################################

                # Clear gradients for this training step
                optimizer.zero_grad()

                # Calculate the output for the batch...
                # train_images_batch.requires_grad = True
                # Print the memory usage of train_images_batch

                # with torch.profiler.profile(
                #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                #         profile_memory=True,
                #         record_shapes=True,
                #         with_stack=True
                # ) as prof:

                if adversarial_type is None: # or epoch < 5:
                    output_train_images_batch = classifier_network(train_images_batch)
                else:
                    device_for_operation = train_images_batch.device.type
                    # Apply the adversarial attack to the training images batch
                    adv_images_batch = train_images_batch.clone()
                    adv_images_batch = attack(adv_images_batch, train_labels_batch).to(device_for_operation)
                    # Create a mixed batch with original and adversarial images, according to the indicated proportion
                    size_tensors = train_images_batch.size()
                    selector_tensor = (
                            torch.rand(
                                (size_tensors[0],) + (1,) * (len(size_tensors) - 1), device=device_for_operation
                            ) < adversarial_proportion
                    ).expand(size_tensors)
                    mixed_images_batch = torch.where(selector_tensor, adv_images_batch, train_images_batch)
                    output_train_images_batch = classifier_network(mixed_images_batch)
                pass

                # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                base_string_justifying_skipped_train_batch = f"\t(Train batch {train_batch + 1} skipped. Reason: %s       )"

                min_num_images_to_accept_training_batch = 5
                if train_images_batch.size(-4) < min_num_images_to_accept_training_batch:
                    print(base_string_justifying_skipped_train_batch %
                          (f"batch contains {train_images_batch.size(-4)} images, " +
                           f"less than the min {min_num_images_to_accept_training_batch} required.")
                          )
                elif torch.any(torch.isnan(output_train_images_batch)):
                    print(base_string_justifying_skipped_train_batch % 'forward pass result contains NaN')
                else:
                    # ... and calculate the loss (in 'cuda', if justified)
                    device_for_comparison = 'cuda' \
                        if (output_train_images_batch.device.type) == 'cuda' or (
                                train_labels_batch.device.type == 'cuda') \
                        else 'cpu'
                    train_loss_i = loss_function(output_train_images_batch.to(device_for_comparison),
                                                 train_labels_batch.to(device_for_comparison))
                    #
                    if torch.any(torch.isnan(train_loss_i)):
                        print(base_string_justifying_skipped_train_batch % "calculated loss is/contains NaN")
                    elif torch.any(torch.isinf(train_loss_i)):
                        print(base_string_justifying_skipped_train_batch % "calculated loss is/contains Inf")
                    else:
                        # Backpropagate (compute gradients) and update parameters with them
                        train_loss_i.backward()
                        # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(classifier_network.parameters(), max_norm)
                    # Check that all the gradients have been properly calculated
                    grad_problem_detected = False
                    for param_name, param_tensor in classifier_network.named_parameters():
                        if param_tensor.requires_grad:
                            if param_tensor.grad is None:
                                grad_problem_detected = True
                                print(base_string_justifying_skipped_train_batch % f"grad of '{param_name}' is None")
                                break
                            elif torch.any(torch.isnan(param_tensor.grad)):
                                grad_problem_detected = True
                                print(
                                    base_string_justifying_skipped_train_batch % f"grad of '{param_name}' contains NaN")
                                break
                            elif torch.any(torch.isinf(param_tensor.grad)):
                                grad_problem_detected = True
                                print(
                                    base_string_justifying_skipped_train_batch % f"grad of '{param_name}' contains Inf")
                                break
                            pass
                        pass
                    pass
                    if not grad_problem_detected:
                        optimizer.step()
                        # TOC train_batch
                        elapsed_train_batch = time.time() - tic_train_batch
                        if verbose == 'high':
                            print((f"    " +
                                   f"Train batch [{(train_batch + 1):5d}/{len(dataset_dict['dataloader_train']):5d}]:   " +
                                   f"batch loss = {train_loss_i:11.4f},   " +
                                   f"batch time = {elapsed_train_batch:.4f} s " +
                                   f"(h:m:s " + str(datetime.timedelta(seconds=round(elapsed_train_batch))) + "), " +
                                   f"avg {elapsed_train_batch / train_images_batch.size(-4):.2f} s/im"),
                                  end='\r')
                        pass
                        #
                        #################################
                        # Accumulate the info of the batch for the (average) train loss and acc
                        #################################
                        #
                        # Since 'train_loss_i' is probably averaged, we un-average it and accumulate it
                        cum_train_loss_times_num_images += \
                            train_loss_i.clone().detach().item() * train_images_batch.size(0)
                        cum_train_images += train_images_batch.size(0)
                        #
                        # Validation ACCURACY (multiclass classification)
                        output_train_labels_batch = torch.max(output_train_images_batch.clone().detach(), dim=-1)[1]
                        cum_train_right_classifications += (
                                output_train_labels_batch.to(device_for_comparison) ==
                                train_labels_batch.to(device_for_comparison)
                        ).sum().item()
                        #
                    pass
                pass

                for param_name, param_tensor in classifier_network.named_parameters():
                    if torch.any(torch.isnan(param_tensor)):
                        grad_problem_detected = True
                        print(base_string_justifying_skipped_train_batch % f"'{param_name}' contains NaN")
                    pass
                    if torch.any(torch.isinf(param_tensor)):
                        grad_problem_detected = True
                        print(base_string_justifying_skipped_train_batch % f"'{param_name}' contains Inf")
                    pass
                pass

                ##############################################################################
                # CALCULATE THE VALIDATION SCORE (IF INTENDED AT THIS NUMBER OF PROCESSED TRAIN BATCHES,
                # AND STORE THE TRAIN AND VALIDATION LOSS AND THE VALIDATION ACCURACY.
                ##############################################################################

                # If in a batch (index) where validation is required: validate and store!!!
                if train_batch in list_ind_batch_for_validation:
                    #
                    # Set the network in evaluation mode and start a .no_grad() scope!
                    classifier_network.eval()
                    #
                    cum_val_images = 0
                    #
                    cum_val_loss_times_num_images = 0.0
                    cum_val_right_classifications = 0
                    #
                    cum_non_convergent_images = 0
                    #
                    if adversarial_type is not None:
                        cum_val_loss_adv_times_num_images = 0.0
                        cum_right_adv_classifications = 0
                    pass
                    #
                    for val_batch, (val_images_batch, val_labels_batch) in enumerate(
                            dataset_dict[subset_for_validation]):
                        #
                        if adversarial_type is not None:
                            if adversarial_arg_dict is None:
                                raise Exception("Adversarial attack type provided but no arguments provided for it!")
                            pass
                            # Apply the adversarial attack to the validation images batch
                            adv_images_batch = val_images_batch.clone()
                            adv_images_batch = attack(adv_images_batch, val_labels_batch)
                        pass
                        with (torch.no_grad()):
                            # TIC val_batch
                            tic_val_batch = time.time()
                            #
                            base_string_justifying_skipped_val_batch = \
                                f"\t(Validation batch {val_batch + 1} skipped. Reason: %s)"
                            #
                            flag_any_problem_encountered_in_validation = True
                            #
                            output_val_images_batch = classifier_network(val_images_batch)

                            if adversarial_type is not None:
                                output_adv_val_images_batch = classifier_network(adv_images_batch)
                            pass
                            #
                            # Statistics regarding convergence
                            _, warning_lack_of_convergence, _, _ = classifier_network.get_last_forward_convergence_info()
                            cum_non_convergent_images += warning_lack_of_convergence.sum(dim=None).item()
                            #
                            if torch.any(torch.isnan(output_val_images_batch)):
                                print(base_string_justifying_skipped_val_batch % 'forward pass result contains NaN')
                            elif torch.any(torch.isinf(output_val_images_batch)):
                                print(base_string_justifying_skipped_val_batch % 'forward pass result contains Inf')
                            else:
                                # ... and calculate the loss (in 'cuda', if justified)
                                device_for_comparison = 'cuda' \
                                    if (output_val_images_batch.device.type == 'cuda') \
                                       or (val_labels_batch.device.type == 'cuda') \
                                    else 'cpu'

                                # Validation LOSS
                                val_loss_i = loss_function(output_val_images_batch.to(device_for_comparison),
                                                           val_labels_batch.to(device_for_comparison))

                                if torch.any(torch.isnan(val_loss_i)):
                                    print(base_string_justifying_skipped_val_batch % "calculated loss is/contains NaN")
                                elif torch.any(torch.isinf(val_loss_i)):
                                    print(base_string_justifying_skipped_val_batch % "calculated loss is/contains Inf")
                                else:
                                    # Since 'val_loss_i' is probably averaged, we un-average it and accumulate it
                                    cum_val_loss_times_num_images += val_loss_i.item() * val_images_batch.size(0)
                                    cum_val_images += val_images_batch.size(0)
                                    #
                                    # Validation ACCURACY (multiclass classification)
                                    output_val_labels_batch = torch.max(output_val_images_batch, dim=-1)[1]
                                    cum_val_right_classifications += (
                                            output_val_labels_batch.to(device_for_comparison) ==
                                            val_labels_batch.to(device_for_comparison)
                                    ).sum().item()
                                    #
                                    if adversarial_type is not None:
                                        # Validation LOSS
                                        val_loss_adv_i = loss_function(output_adv_val_images_batch.to(device_for_comparison),
                                                                   val_labels_batch.to(device_for_comparison))
                                        if torch.any(torch.isnan(val_loss_i)):
                                            print(
                                                base_string_justifying_skipped_val_batch % "calculated loss is/contains NaN")
                                        elif torch.any(torch.isinf(val_loss_i)):
                                            print(
                                                base_string_justifying_skipped_val_batch % "calculated loss is/contains Inf")
                                        else:
                                            # Since 'val_loss_i' is probably averaged, we un-average it and accumulate it
                                            cum_val_loss_adv_times_num_images += val_loss_adv_i.item() * val_images_batch.size(
                                                0)
                                            #
                                            # Validation ACCURACY (multiclass classification) for adversarial images
                                            output_adv_val_labels_batch = torch.max(output_adv_val_images_batch, dim=-1)[1]
                                            cum_right_adv_classifications += (
                                                    output_adv_val_labels_batch.to(device_for_comparison) ==
                                                    val_labels_batch.to(device_for_comparison)
                                            ).sum().item()
                                    # TOC val_batch
                                    elapsed_val_batch = time.time() - tic_val_batch
                                    if verbose == 'high' and adversarial_type is None:
                                        print((f"    " +
                                               f"Val batch   [{(val_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]:   " +
                                               f"batch loss = {val_loss_i:11.4f},   " +
                                               f"batch time = {elapsed_val_batch:.4f} s " +
                                               f"(h:m:s " + str(
                                                    datetime.timedelta(seconds=round(elapsed_val_batch))
                                                ) + "), " +
                                               f"avg {elapsed_val_batch / val_images_batch.size(-4):.3f} s/im"),
                                              end='\r')
                                    elif verbose == 'high' and adversarial_type is not None:
                                        print((f"    " +
                                               f"Val batch   [{(val_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]:   " +
                                               f"batch loss = {val_loss_i:11.4f},   " +
                                               f"batch loss_adv = {val_loss_adv_i:11.4f},   " +
                                               f"batch time = {elapsed_val_batch:.4f} s " +
                                               f"(h:m:s " + str(
                                                    datetime.timedelta(seconds=round(elapsed_val_batch))
                                                ) + "), " +
                                               f"avg {elapsed_val_batch / val_images_batch.size(-4):.3f} s/im"),
                                              end='\r')
                                    pass
                                    #
                                    flag_any_problem_encountered_in_validation = False
                                    #
                                pass
                            pass
                            #
                        pass  # for val_batch, (val_images_batch, val_labels_batch) in enumerate(dataset_dict[subset_for_validation]):


                    if adversarial_type is not None:
                        # If adversarial attack, we need to calculate the accuracy for the adversarial images
                        output_adv_val_labels_batch = torch.max(output_adv_val_images_batch, dim=-1)[1]
                        cum_right_adv_classifications += (
                                output_adv_val_labels_batch.to(device_for_comparison) == val_labels_batch.to(
                            device_for_comparison)
                        ).sum().item()
                    pass  # with (torch.no_grad()):

                    # Set the network back in training mode
                    classifier_network.train()

                    val_loss = cum_val_loss_times_num_images / cum_val_images if cum_val_images > 0 else np.nan
                    val_acc = cum_val_right_classifications / cum_val_images if cum_val_images > 0 else np.nan

                    if adversarial_type is not None:
                        val_loss_adv = cum_val_loss_adv_times_num_images / cum_val_images if cum_val_images > 0 else np.nan
                        val_acc_adv = cum_right_adv_classifications / cum_val_images if cum_val_images > 0 else np.nan
                    pass

                    proportion_non_convergent = cum_non_convergent_images / cum_val_images

                    ############################################################
                    # Calculate the averaged train loss and acc up to this (reporting) batch
                    ############################################################
                    train_loss = cum_train_loss_times_num_images / cum_train_images if cum_train_images > 0 else np.nan
                    train_acc = cum_train_right_classifications / cum_train_images if cum_train_images > 0 else np.nan
                    ############################################################

                    # TOC train_batch
                    elapsed_block_between_validations = time.time() - tic_block_between_validations
                    num_images_between_validations = dataset_dict['batch_size'] * (
                            list_ind_batch_for_validation[1] - list_ind_batch_for_validation[0]
                    ) if len(list_ind_batch_for_validation) > 1 else list_ind_batch_for_validation[
                        0]  # If only 1 validation
                    # TIC block_between_validations: reset
                    tic_block_between_validations = time.time()

                    if verbose in ['high', 'medium']:
                        print((f"    " +
                               f"Val @ epoch [{(epoch + 1):3d}/{total_epochs:3d}]:  " +
                               f"loss={val_loss:10.3f},   " +
                               f"acc={100 * val_acc:7.2f} %,   " +
                               f"non-converged images={100 * proportion_non_convergent:6.2f}%"),
                              end='')
                        if adversarial_type is not None:
                            print((f",   " +
                                   f"loss_adv={val_loss_adv:10.3f},  " +
                                   f"acc_adv={100 * val_acc_adv:7.2f} %   "),
                                  end='')
                        pass
                        print((f"(train+val between validations: h:m:s " + str(
                                    datetime.timedelta(seconds=round(elapsed_block_between_validations))
                                ) + ")"
                               ), end='\n')
                    pass

                    ##############################################################################
                    # Perform logging at each validation step
                    ##############################################################################

                    # Add to the list of all values stored within the epoch
                    list_validation_loss_within_epoch.append(val_loss)
                    list_validation_acc_within_epoch.append(val_acc)

                    # Pandas logging
                    # TOC beginning_training
                    elapsed_since_beginning_training = time.time() - tic_beginning_training

                    df_training_progress.loc[(epoch + 1, train_batch + 1), ('val', 'loss')] = val_loss
                    df_training_progress.loc[(epoch + 1, train_batch + 1), ('train', 'loss')] = train_loss
                    df_training_progress.loc[(epoch + 1, train_batch + 1), ('val', 'acc')] = val_acc
                    df_training_progress.loc[(epoch + 1, train_batch + 1), ('train', 'acc')] = train_acc
                    df_training_progress.loc[(epoch + 1, train_batch + 1), 'elapsed_time'] = \
                        datetime.timedelta(seconds=round(elapsed_since_beginning_training))
                    #
                    # MLFlow logging
                    if mlflow_run_id is not None and current_model_key != 'sm_surrogate':
                        metrics_dict = {
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'proportion_non_convergent': proportion_non_convergent,
                            'current_lr': optimizer.state_dict()['param_groups'][0]['lr'],
                            'elapsed_time': elapsed_since_beginning_training,
                            'epoch_fraction': df_training_progress.loc[(epoch + 1, train_batch + 1), 'epoch_fraction']
                        }
                        if isinstance(optimizer, torch.optim.SGD): # Parameters only present in SGD
                            metrics_dict.update({
                                'momentum': optimizer.param_groups[0]['momentum']
                            })
                        pass

                        # Add adversarial metrics if adversarial attack is used
                        if adversarial_type is not None:
                            metrics_dict.update({
                                'val_loss_adv': val_loss_adv,
                                'val_acc_adv': val_acc_adv
                            })
                        mlflow.log_metrics(metrics_dict, step=global_batch + 1, run_id=mlflow_run_id)
                        conv_param_tracker.log_to_mlflow(epoch=epoch)
                    pass
                    conv_param_tracker.reset()

                    # Record as best if it is the best (according to validation loss):
                    # WARNING: THIS APPLIES ONLY TO THE ORIGINAL, NOT TO THE SURROGATE!
                    if current_model_key == 'original':
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_acc = val_acc
                            best_model = copy.deepcopy(classifier_network.state_dict())
                            proportion_non_convergent_best_model = proportion_non_convergent
                            if adversarial_type is not None:
                                best_loss_adv = val_loss_adv
                                best_acc_adv = val_acc_adv
                            pass
                        pass
                    pass

                pass  # if (train_batch in list_ind_batch_for_validation:
                #
                # # Delete some variables
                # for var_name in ['output_train_images_batch', 'train_loss']:
                #     if var_name in locals(): del locals()[var_name]
                # gc.collect()
                # torch.cuda.empty_cache()
                #
                # Increase the global batch pointer
                global_batch += 1
                #
                # And reset the counters for assessing the training acc and loss
                cum_train_images = 0
                cum_train_loss_times_num_images = 0.0
                cum_train_right_classifications = 0
                #
            pass  # for train_batch, (train_images_batch, train_labels_batch) in enumerate(dataset_dict['dataloader_train']):


            ##############################################################################
            # Perform logging at the end of the epoch
            ##############################################################################

            # Add to the list of the final validation loss+acc at the end of the epochs
            list_final_validation_loss_across_epochs.append(list_validation_loss_within_epoch[-1])
            list_final_validation_acc_across_epochs.append(list_validation_acc_within_epoch[-1])

            ##############################################################################
            # CHECK EARLY STOP CONDITIONS
            ##############################################################################

            # Stop training before 'maximum_epochs' if no mean validation loss improvement in the last `early_stop_epochs`
            if (early_stop_epochs is not None) and (early_stop_epochs != 0) and \
                    (len(list_final_validation_loss_across_epochs) > early_stop_epochs):
                previous_min, epoch_min = torch.Tensor(list_final_validation_loss_across_epochs[:-1]).min(dim=0)
                if (previous_min <= list_final_validation_loss_across_epochs[-1]) and \
                        ((epoch - epoch_min) >= early_stop_epochs):
                    print(f"Early stop: no loss progress in the last {early_stop_epochs} epochs!")
                    break
            pass

            ##############################################################################
            # USE THE SCHEDULER (IF SO SET)
            ##############################################################################

            # If scheduler, "evolve" the learning rate. Warning: if 'ReduceLROnPlateau' slightly different operation
            if scheduler is not None:
                lr_before_update = optimizer.state_dict()['param_groups'][0]['lr']
                #
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # The step depends on the validation loss of the epoch
                    scheduler.step(list_final_validation_loss_across_epochs[-1])
                else:
                    # Update the scheduler
                    scheduler.step()
                pass
                lr_after_update = optimizer.state_dict()['param_groups'][0]['lr']
                if lr_after_update != lr_before_update and verbose in ['high', 'medium']:
                    print(f"\tNew learning rate! {lr_before_update:.4f} ----> {lr_after_update:.4f}")
                pass
            pass

            ############
            # @DISPLAY: epoch summary
            ############

            # TOC train_epoch
            elapsed_train_epoch = time.time() - tic_train_epoch
            num_images_epoch = dataset_dict['batch_size'] * len(dataset_dict['dataloader_train'])

            if verbose in ['high', 'medium', 'low'] and adversarial_type is None:
                print((f"    " +
                       f"EPOCH [{(epoch + 1):3d}/{total_epochs:3d}]:   " +
                       f"loss = {val_loss:10.3f},   " +
                       f"acc = {100 * val_acc:7.2f} %,    " +
                       ("" if proportion_non_convergent_best_model is None else \
                            f"non-converged images = {100 * proportion_non_convergent_best_model:7.3f} %   ") +
                       f"(h:m:s " + str(
                            datetime.timedelta(seconds=round(elapsed_train_epoch))
                        ) + ", " +
                       f"avg {elapsed_train_epoch / num_images_epoch:.3f} s/im)"),
                      end='\n')
            elif verbose in ['high', 'medium', 'low'] and adversarial_type is not None:
                print((f"    " +
                       f"EPOCH [{(epoch + 1):3d}/{total_epochs:3d}]:   " +
                       f"loss = {val_loss:10.3f},   " +
                       f"acc = {100 * val_acc:7.2f} %,   " +
                       f"loss_adv = {val_loss_adv:10.3f},   " +
                       f"acc_adv = {100 * val_acc_adv:7.2f} %,   " +
                       ("" if proportion_non_convergent_best_model is None else \
                            f"non-converged images = {100 * proportion_non_convergent_best_model:7.3f} %   ") +
                       f"(h:m:s " + str(
                            datetime.timedelta(seconds=round(elapsed_train_epoch))
                        ) + ", " +
                       f"avg {elapsed_train_epoch / num_images_epoch:.3f} s/im)"),
                      end='\n')
            pass
            #
            #
            epoch += 1
        pass
        # INTENDED: END OF THE...  # for epoch in range(maximum_epochs):

        #
        end_training = datetime.datetime.now()  # to mirror 'start_training'

        elapsed_total_training = time.time() - tic_start_training
        if verbose in ['high', 'medium', 'low'] and adversarial_type is None:
            print(f"TRAINING FINISHED: [{total_epochs:3d} epochs]   " +
                  f"best_loss = {best_loss:11.4f},   " +
                  f"best_acc = {100 * best_acc:7.3f} % " +
                  f"(h:m:s " + str(datetime.timedelta(seconds=round(elapsed_total_training))) + ")",
                  end='\n\n')
        elif verbose in ['high', 'medium', 'low'] and adversarial_type is not None:
            print(f"TRAINING FINISHED: [{total_epochs:3d} epochs]   " +
                  f"best_loss = {best_loss:11.4f},   " +
                  f"best_acc = {100 * best_acc:7.3f} %,   " +
                  f"best_loss_adv = {best_loss_adv:11.4f},   " +
                  f"best_acc_adv = {100 * best_acc_adv:7.3f} % " +
                  f"(h:m:s " + str(datetime.timedelta(seconds=round(elapsed_total_training))) + ")",
                  end='\n\n')
        pass

        # Once training has been finished, set the network in evaluation mode
        classifier_network.eval()

        ###############################################################################################
        # We finish the logging: closing the SummaryWriter if open, saving the required data, and preparing the "return"
        ###############################################################################################

        ###############################################################################################
        # Finish the preparation of the Pandas DF with the progress of the training and the info of the net and test

        # The progress of the training will be increased with the information of the dataset and of the network. \
        # Such added info will be pushed simply as "front columns" in the multiindex that the \
        # "df_training_progress" already has

        df_params_and_progress = None

        # Create DFs with parameters of different aspects of the training, and save the resulting data if required

        list_of_df_with_additional_parameters = []

        # Create a multiindex, and a DF from it, with the date info
        multiindex_date = pd.MultiIndex.from_product(
            [[f"{datetime_i:{_datetime_str_format}}"] for datetime_i in [start_training, end_training]] + [[host]],
            names=["start_training", "end_training", "host"]
        )
        df_multiindex_date = pd.DataFrame(index=multiindex_date)
        list_of_df_with_additional_parameters.append(df_multiindex_date)

        # Create a multiindex, and a DF from it, with the info of the network: all parameters, if any
        # WARNING: all values must be hashable, so check that they are or make them!
        hashable_fields_to_log = classifier_network.fields_to_log
        for key in hashable_fields_to_log:
            if not isinstance(hashable_fields_to_log[key], typing.Hashable):
                if isinstance(hashable_fields_to_log[key], list):
                    hashable_fields_to_log[key] = tuple(hashable_fields_to_log[key])
                else:
                    hashable_fields_to_log[key] = str(hashable_fields_to_log[key])
                pass
            pass
        pass

        multiindex_network = pd.MultiIndex.from_product(
            [[hashable_fields_to_log[key]] for key in hashable_fields_to_log],
            names=[key for key in hashable_fields_to_log]
        )
        df_multiindex_network = pd.DataFrame(index=multiindex_network)
        list_of_df_with_additional_parameters.append(df_multiindex_network)

        # Create a multiindex, and a DF from it, with the info of the optimizer and scheduler
        hashable_dict_optimizer_and_scheduler_params = copy.deepcopy(dict_optimizer_and_scheduler_params)
        for key in dict_optimizer_and_scheduler_params:
            if not isinstance(hashable_dict_optimizer_and_scheduler_params[key], typing.Hashable):
                if isinstance(hashable_dict_optimizer_and_scheduler_params[key], list):
                    hashable_dict_optimizer_and_scheduler_params[key] = \
                        tuple(hashable_dict_optimizer_and_scheduler_params[key])
                else:
                    hashable_dict_optimizer_and_scheduler_params[key] = \
                        str(hashable_dict_optimizer_and_scheduler_params[key])
                pass
            pass
        pass
        multiindex_optimizer_and_scheduler_params = pd.MultiIndex.from_product(
            [[hashable_dict_optimizer_and_scheduler_params[key]] \
             for key in hashable_dict_optimizer_and_scheduler_params],
            names=[key for key in hashable_dict_optimizer_and_scheduler_params]
        )
        df_multiindex_optimizer_and_scheduler_params = pd.DataFrame(index=multiindex_optimizer_and_scheduler_params)
        list_of_df_with_additional_parameters.append(df_multiindex_optimizer_and_scheduler_params)

        # Create a multiindex, and a DF from it, with the info of the dataset:
        # all parameters but those containing certain keywords
        keys_of_interest_dataset_dict = []
        for key in dataset_dict:
            flag_busted = False
            for substring in ['loader', '_mean_', '_std_', '_max_', '_min_']:
                if substring in key:
                    flag_busted = True
                    break
            if not flag_busted:
                keys_of_interest_dataset_dict.append(key)
        pass
        multiindex_dataset = pd.MultiIndex.from_product(
            [[dataset_dict[key]] for key in keys_of_interest_dataset_dict],
            names=keys_of_interest_dataset_dict
        )
        df_multiindex_dataset = pd.DataFrame(index=multiindex_dataset)
        list_of_df_with_additional_parameters.append(df_multiindex_dataset)

        # Add them to the DF with the progress of the training
        # Merge all the additional DFs with params into one single merged DF without index
        df_no_index_merged_params = list_of_df_with_additional_parameters[0].reset_index()
        for df_params_i in list_of_df_with_additional_parameters[1:]:
            df_no_index_merged_params = pd.merge(df_no_index_merged_params, df_params_i.reset_index(), how="cross")
        pass

        # Merge them with the "df_training_progress" with no indices
        merged_multi_index = pd.MultiIndex.from_frame(pd.merge(
            df_no_index_merged_params, df_training_progress.index.to_frame().reset_index(drop=True), how='cross'
        ))
        df_params_and_progress = df_training_progress.set_index(merged_multi_index)

        # MLFlow logging
        if mlflow_run_id is not None:
            # print(f"dict_optimizer_and_scheduler_params = \n{dict_optimizer_and_scheduler_params}")
            mlflow.log_params(hashable_dict_optimizer_and_scheduler_params, run_id=mlflow_run_id)
            mlflow.log_params(dataset_dict, run_id=mlflow_run_id)
            mlflow.log_params(hashable_fields_to_log, run_id=mlflow_run_id)
            mlflow.log_params({'start_training': start_training, 'end_training': end_training, 'host': host},
                              run_id=mlflow_run_id)
            mlflow.log_metrics(
                {'best_loss': best_loss, 'best_acc': best_acc,
                 'proportion_non_convergent_best_model':
                     0.0 if proportion_non_convergent_best_model is None else proportion_non_convergent_best_model
                 },
                run_id=mlflow_run_id
            )
            if adversarial_type is not None:
                mlflow.log_metrics(
                    {'best_loss_adv': best_loss_adv, 'best_acc_adv': best_acc_adv},
                    run_id=mlflow_run_id
                )
            pass
        pass

        ###############################################################################################
        # SPEED METRICS:

        # Series 'epoch_fraction'-'acc' and 'epoch_fraction'-'loss' for both training and validation

        val_acc_series = df_params_and_progress.set_index('epoch_fraction')[('val', 'acc')]
        val_loss_series = df_params_and_progress.set_index('epoch_fraction')[('val', 'loss')]

        # ABSOLUTE SPEED METRICS: Extract the epoch fraction when a given acc was first obtained

        step = 0.05
        array_reference_accs = np.arange(step, 1.0 + step, step).tolist()
        array_epoch_fraction_per_val_accs = array_reference_accs.copy()

        # Extract the index of the first appearance of the accuracy
        for i, reference_acc_i in enumerate(array_reference_accs):
            epoch_fractions_true = val_acc_series[val_acc_series > reference_acc_i].index.to_numpy()
            array_epoch_fraction_per_val_accs[i] = epoch_fractions_true[0] if len(epoch_fractions_true) > 0 else None
        pass

        pandas_series_epoch_for_acc = pd.Series(array_epoch_fraction_per_val_accs, index=array_reference_accs)

        # RELATIVE SPEED METRICS FOR BOTH ACC AND LOSS: relative to the 'best_acc' and 'best_loss'
        # - To 'bess_acc': 5%-points below
        diff_to_best_acc = 0.05
        epoch_fractions_close_to_best_acc = \
            val_acc_series[val_acc_series > best_acc - diff_to_best_acc].index.to_numpy()
        epoch_fraction_close_to_best_acc = \
            epoch_fractions_close_to_best_acc[0] if len(epoch_fractions_close_to_best_acc) > 0 else None

        # - To 'bess_loss': (1+1e-3) of the best loss
        factor_to_best_loss = 1.0 + 1e-3
        epoch_fractions_close_to_best_loss = \
            val_loss_series[val_loss_series < best_loss * factor_to_best_loss].index.to_numpy()
        epoch_fraction_close_to_best_loss = \
            epoch_fractions_close_to_best_loss[0] if len(epoch_fractions_close_to_best_loss) > 0 else None

        ###############################################################################################
        # Store the best network plus its summary, PANDAS of the training: always (subject to potential deletion later)

        # Store the resulting DF in both pickle and csv formats!
        # To PICKLE
        df_params_and_progress.to_pickle(path=abs_log_file_name_pandas_pkl)
        # To CSV
        df_params_and_progress.to_csv(
            path_or_buf=abs_log_file_name_pandas_csv,
            header=True, index=True, index_label=True
        )

        # Best model and summary
        torch.save(best_model, abs_log_file_name_state_dict)
        with open(abs_log_file_name_summary, 'w') as f:
            f.write(f"Best loss: {best_loss:.4f}\n")
            f.write(f"Best acc:  {best_acc:.4f}\n")
            f.write(f"\nModel:\n\n{str(best_model)}\n")
            f.write(f"\nDataset:\n\n")
            for key in dataset_dict:
                f.write(f"\t{key}: {dataset_dict[key]}\n")
            pass
            # See if the trained network has a layer called 'prenormalization' and, if so, write its parameters
            f.write(f"\nPrenormalization layers (if any):\n")
            for name, layer in classifier_network.named_modules():
                if 'prenormalization' in name and isinstance(layer, torch.nn.BatchNorm2d):
                    print(f"{name}\n")
                    print(f"Layer:\n{str(layer)}\n")
                pass
            pass
            for name, layer in best_model.items():
                if 'prenormalization.running_mean' in name:
                    print(f" Calculated mean: {layer}\n")
                if 'prenormalization.running_var' in name:
                    print(f" Calculated var: {layer}\n")
                    print(f"(Calculated std: {torch.sqrt(layer)})\n")
                pass
            pass
        pass

        ###############################################################################################
        # Prepare and return the named tuple result from the run
        resultClassifierTraining = namedtuple(
            "LoggingFlagsTuple",
            ['experiment_folder', 'best_model', 'best_acc', 'best_loss',
             'pandas_series_epoch_for_acc', 'epoch_fraction_close_to_best_acc', 'epoch_fraction_close_to_best_loss',
             'pandas_df_log']
        )
        result_classifier_training = resultClassifierTraining(
            abs_experiment_folder, best_model, best_acc, best_loss,
            pandas_series_epoch_for_acc, epoch_fraction_close_to_best_acc, epoch_fraction_close_to_best_loss,
            df_params_and_progress
        )

        # MLFlow logging end run
        if mlflow_run_id is not None:
            if maximum_epochs > 0:
                mlflow.log_metrics({'epoch_fraction_close_to_best_acc': epoch_fraction_close_to_best_acc,
                                    'epoch_fraction_close_to_best_loss': epoch_fraction_close_to_best_loss},
                                   run_id=mlflow_run_id)
                # iterate over pandas series epoch for acc and log them
                for acc, epoch in pandas_series_epoch_for_acc.items():
                    mlflow.log_metrics({f'epoch_fraction_close_to_acc_{int(acc * 100)}': epoch}, run_id=mlflow_run_id)
                mlflow.log_artifact(abs_log_file_name_pandas_pkl, run_id=mlflow_run_id)
                mlflow.log_artifact(abs_log_file_name_pandas_csv, run_id=mlflow_run_id)
                mlflow.log_artifact(abs_log_file_name_state_dict, run_id=mlflow_run_id)
                mlflow.log_artifact(abs_log_file_name_summary, run_id=mlflow_run_id)
                #
                # MOVE AT THE BEGINNING OF THE FUNCTION (and based on an auxiliary function)
                # # Log the 'dataset_dict' but removing the iterables, so only the info of interest is logged
                # mlflow.log_dict(dataset_dict.dict_wo_dataloaders(), 'dataset_dict.json', run_id=mlflow_run_id)
            pass
        pass
        # Once all the logging has been finished, we delete the local log folder if required
        if flag_temp_log_folder:
            shutil.rmtree(local_log_folder)
        pass

    return result_classifier_training


#############################################################################################
#############################################################################################
# RESULT LOADING, REFORMATTING, AND PLOTTING
#############################################################################################
#############################################################################################

def _grid_multilevel_ax(cax, status=True):
    if isinstance(cax, Axes):
        cax.grid(status)
    else:
        for i in range(len(cax)):
            _grid_multilevel_ax(cax[i], status)
        pass
    pass


def _create_figure_and_subplots(subset, figure_of_merit, cax=None, figsize=None, same_figure_of_merit_along_row=True):
    """
    Assisstent function for creating the (fig, ax), or assess whether a pre-exisiting Axes ``cax`` is fit, for the \
    given pair of ``subset`` and ``figure_of_merit``.

    Parameters
    ----------
    subset : str or list[str] or tuple[str]
    figure_of_merit : str or list[str] or tuple[str]
    cax : :py:class:`matplotlib.axes.Axes`, optional
        Default: ``None``
    figsize : tuple[int], optional
        Default: ``None`` (equivalent to $6.4 \\times 4.8$ inches)
    same_figure_of_merit_along_row : bool

    Returns
    -------
    tuple[:py:class:`matplotlib.figure.Figure`, :py:class:`matplotlib.axes.Axes`]
    """

    # Make both inputs list for uniform handling
    if isinstance(subset, str):
        subset = [subset]  # We make a list (we will process it later, by default, as a list)
    pass
    if isinstance(figure_of_merit, str):
        figure_of_merit = [figure_of_merit]  # We make a list (we will process it later, by default, as a list)
    pass

    # Number of rows and cols for the given query
    nrows, ncols = None, None
    if (len(subset) == 1) or (len(figure_of_merit) == 1):
        (nrows, ncols) = (1, max(len(subset), len(figure_of_merit)))
    else:
        (nrows, ncols) = (len(figure_of_merit), len(subset)) if same_figure_of_merit_along_row \
            else (len(subset), len(figure_of_merit))
    pass

    # Define or reuse axes
    fig = None
    if cax is None:
        sharex = 'all'
        if same_figure_of_merit_along_row:
            sharey = 'row' if len(subset) > 1 else False
        else:
            sharey = 'col' if len(subset) > 1 else False
        pass
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize)
    else:
        # Check if the number of elements in "cax" is compatible
        if isinstance(cax, Axes):
            nrows_cax, ncols_cax = 1, 1
        else:
            nrows_cax = len(cax)
            ncols_cax = 1 if isinstance(cax[0], Axes) else len(cax[0])
        pass
        if (nrows == nrows_cax) and (ncols == ncols_cax):
            ax = cax
        else:
            raise Exception((f"For the provided query of subsets = {str(subset)} and " +
                             f"figures of merit = {str(figure_of_merit)} the expected axes would require a size of " +
                             f"({nrows}, {ncols}) but a 'cax' of size ({nrows_cax}, {ncols_cax}) has been provided!"))
        pass
    pass

    return fig, ax, (nrows, ncols)


def _calculate_series_epoch_fraction(df_params_and_progress):
    series_epoch_fraction = df_params_and_progress.loc[:, "abs_batch"] / \
                            df_params_and_progress.reset_index(level="num_batches_train").loc[:, "num_batches_train"]
    return series_epoch_fraction


def _infer_inset_parameters(df_experiment_group, subset, figure_of_merit, x_min, x_max, x_mode='batch'):
    #
    modality_x_axis = None
    label_x_axis = None
    if x_mode == 'batch':
        modality_x_axis = 'abs_batch'
        label_x_axis = '(abs) batch number'
    elif x_mode == 'time':
        modality_x_axis = 'elapsed_time'
        label_x_axis = 'elapsed time'
    elif x_mode == 'epoch':
        modality_x_axis = 'epoch_fraction'
        label_x_axis = 'epoch (and fraction)'
    else:
        raise Exception(f"Allowable values for argument 'x_mode' are: 'batch', 'time', and 'epoch'. Found: {x_mode}")
    pass
    #
    # We limit the DF to the two cols of interest
    pair_subset_figure_of_merit = (subset, figure_of_merit)
    df_2_col = pd.concat(
        [df_experiment_group[modality_x_axis], df_experiment_group[pair_subset_figure_of_merit]], axis=1
    ).reset_index(drop=True)
    #
    # Filter the values of the col within the indicated range [x_min, x_max]
    filtered_df = df_2_col[(df_2_col[modality_x_axis] < x_max) & (df_2_col[modality_x_axis] > x_min)]
    filtered_series = filtered_df[pair_subset_figure_of_merit]
    quantiles = filtered_series.quantile([0.1, 0.9])
    y_min = quantiles.iloc[0]
    y_max = quantiles.iloc[1]
    y_range = y_max - y_min
    margin = 0.1 * y_range
    #
    return y_min - margin, y_max + margin


def _infer_inset_parameters_experiment_groups(df_multiexperiment_params_and_progress, experiment_group_dictionary,
                                              subset, figure_of_merit, x_min, x_max, x_mode='batch'):
    # We create a DF with only the data corresponding to the experiments in 'experiment_group_dictionary'
    list_df_experiment_groups = []
    for eg, experiment_group_dictionary_i in enumerate(experiment_group_dictionary):
        #
        # Filter the group experiment
        df_experiment_group_i = df_multiexperiment_params_and_progress.xs(
            tuple(experiment_group_dictionary_i.values()),
            level=tuple(experiment_group_dictionary_i.keys()),
            drop_level=False
        )
        # Append to the list
        list_df_experiment_groups.append(df_experiment_group_i)
    pass
    concatenated_df_experiment_groups = pd.concat(list_df_experiment_groups)

    #
    return _infer_inset_parameters(concatenated_df_experiment_groups,
                                   subset=subset, figure_of_merit=figure_of_merit,
                                   x_min=x_min, x_max=x_max, x_mode=x_mode)


def plot_figures_of_merit_from_individual_experiment(
        df_experiment, subset, figure_of_merit,
        x_mode='batch', inset_x_range=None, inset_relative_position=(0.38, 0.98, 0.3, 0.6),
        grid=True, print_experiment_info=False, cax=None, figsize=None):
    """
    It plots the requested figure of merit in ``figure_of_merit``, for the requested single ``subset`` or \
    multiple subsets if an iterable is provided (e.g. a list with ``'train'`` and ``'val'``): in the case \
    of multiple subsets the same figure of merit for both would be displayed on the same axis. The displayed
    figures of merit are displayed visually indicating the different epochs.

    Additionally, ``figure_of_merit`` can also be an iterable considering several figures of merit \
    (e.g. a list with ``'loss'`` and ``'acc'``): in such case a row of axes would be generated, \
    with each axis corresponding to one single figure of merit.

    Parameters
    ----------
    df_experiment : :py:class:`pandas.DataFrame`
    subset : str or list[str] or tuple[str]
        Single ``subset`` or multiple subsets in an iterable, among ``'train'`` and ``'val'``
    figure_of_merit : str or list[str] or tuple[str]
        Single ``figure_of_merit`` or multiple figures of merit in an iterable, among ``'loss'`` and ``'acc'``
    x_mode : str, optional
        Value among ``'batch'``, ``'time'``, and ``'epoch'``. It selects which unit, whether the absolute index of the \
        batch, the elapsed time (in seconds), or the fraction of a complete epoch, is displayed on the x-axis of the \
        plot. \
        Default: ``'batch'``
    inset_x_range : tuple[int or float] or list[int or float], optional
        Definition of the x limits of the x-axis for an automatically generated (zoom) inset: 2D tuple/list. \
        Default: ``None`` (no inset)
    inset_relative_position : tuple[int or float] or list[int or float]
        Definition of the position of the inset in the images, which will be treated identically for all sub-axes. \
        Each dimension is $\\in \\[0,1\\]$ and their respective meaning is: *[x0, x1, y0, y1]*, \
        **and not position and with as in the definition** *[x0, y0, width, height]* **of** \
        :py:meth:`matplotlib.axes.Axes.inset_axes`. \
        Default: ``[0.38, 0.98, 0.3, 0.6]``
    grid : bool, optional
        Default: ``True``
    print_experiment_info : bool, optional
        If ``True`` the function prints a table with the parameters of the experiment. \
        Default: ``False``
    cax : :py:class:`matplotlib.axes.Axes`, optional
        Default: ``None``
    figsize : tuple[int], optional
        Default: ``None`` (equivalent to $6.4 \\times 4.8$ inches)

    Returns
    -------
    :py:class:`matplotlib.axes.Axes`

    """

    ##############################
    # React to the selected 'x_mode'
    ##############################

    modality_x_axis = None
    label_x_axis = None
    if x_mode == 'batch':
        modality_x_axis = 'abs_batch'
        label_x_axis = '(abs) batch number'
    elif x_mode == 'time':
        modality_x_axis = 'elapsed_time'
        label_x_axis = 'elapsed time'
    elif x_mode == 'epoch':
        # This option requires a certain re-normalization of the data: everything needs normalization
        # to the number of epochs, which requires a new column
        modality_x_axis = 'epoch_fraction'
        label_x_axis = 'epoch (and fraction)'
        # # Add column
        # df_experiment["epoch_fraction"] = \
        #     _calculate_series_epoch_fraction(df_experiment)
    else:
        raise Exception(f"Allowable values for argument 'x_mode' are: 'batch', 'time', and 'epoch'. Found: {x_mode}")
    pass

    ##############################
    # Check the format of "inset_x_range" if provided, and its size
    ##############################

    if inset_x_range is not None:
        if (isinstance(inset_x_range, tuple) or isinstance(inset_x_range, list)) and (len(inset_x_range) == 2):
            pass  # Alles gut
        else:
            raise Exception((f"The provided 'inset_x_range' is no 2D tuple/list or None, as expected, " +
                             f"but a {type(inset_x_range)}"))
        pass
    pass

    if (isinstance(inset_relative_position, tuple) or isinstance(inset_relative_position, list)) \
            and (len(inset_relative_position) == 4):
        pass  # Alles gut
    else:
        raise Exception((f"The provided 'inset_relative_position' is no 4D tuple/list or None, as expected, " +
                         f"but a {type(inset_relative_position)}"))
    pass

    ##############################
    # First of all: we check that there is only one experiment!!!
    ##############################

    params_separate_experiments = df_experiment.index.droplevel(['epoch', 'batch']).unique()
    num_experiments = len(params_separate_experiments)
    if num_experiments != 1:
        raise Exception(
            (
                    f"Function 'plot_figures_of_merit_from_individual_experiment' is designed for a DF depicting 1 single experiment, " +
                    f"however info corresponding to {num_experiments} experiments has been found.")
        )
    elif print_experiment_info:
        print(pd.DataFrame(index=params_separate_experiments).reset_index().T)
    pass

    ##############################
    # Checking how many subplots will be needed
    ##############################

    if isinstance(subset, str):
        subset = [subset]  # We make a list (we will process it later, by default, as a list)
    pass
    if isinstance(figure_of_merit, str):
        figure_of_merit = [figure_of_merit]  # We make a list (we will process it later, by default, as a list)
    pass
    #
    if cax is None:
        fig, ax = plt.subplots(nrows=1, ncols=len(figure_of_merit), sharex=True, figsize=figsize)
    else:
        ax = cax
    pass

    ##############################
    # Plot formats
    ##############################

    dict_format_plot = {"val": '-', "train": ':'}
    dict_linewidth_plot = {"val": 2, "train": 1.5}
    #

    ##############################
    # Display the experiment
    # NOTE: in this plot the subsets are drawn on the same axis; different figs of merit, different axes
    ##############################

    handles_plots = []
    #
    for i, figure_of_merit_i in enumerate(figure_of_merit):
        ax_i = ax if len(figure_of_merit) == 1 else ax[i]
        #
        if inset_x_range is not None:
            inset_y_range_axis_ij = [np.inf, -np.inf]
        pass
        #
        for j, subset_j in enumerate(subset):
            # Display
            ax_i.set_prop_cycle(None)
            #
            for z, epoch in enumerate(df_experiment.index.unique(level='epoch')):
                handle_plot = ax_i.plot(
                    df_experiment.xs(epoch, level="epoch").loc[:, modality_x_axis],
                    df_experiment.xs(epoch, level="epoch").loc[:, (subset_j, figure_of_merit_i)],
                    linestyle=dict_format_plot.get(subset_j), linewidth=dict_linewidth_plot.get(subset_j)
                )
                if z == 0:
                    handles_plots.extend(handle_plot)
                pass
            pass
            #
            if inset_x_range is not None:
                current_inset_y_range = _infer_inset_parameters(df_experiment, subset_j, figure_of_merit_i,
                                                                inset_x_range[0], inset_x_range[1], x_mode=x_mode)
                inset_y_range_axis_ij[0] = min(inset_y_range_axis_ij[0], current_inset_y_range[0])
                inset_y_range_axis_ij[1] = max(inset_y_range_axis_ij[1], current_inset_y_range[1])
            pass
        pass
        ax_i.set_title(figure_of_merit_i)
        ax_i.set_xlabel(label_x_axis)
        ax_i.grid(grid)
        #
        ax_i.legend(handles_plots, subset)
        #
        if ((inset_x_range is not None) and
                (not np.isnan(inset_y_range_axis_ij[0])) and (not np.isinf(inset_y_range_axis_ij[0]))):
            # print(f"inset_x_range = {inset_x_range}")
            # print(f"inset_y_range_axis_ij = {inset_y_range_axis_ij}")
            # The limits for the inset are: inset_x_range, inset_y_range_axis_ij
            # The position within the axis:
            inset_x0y0wh_relative_position = [inset_relative_position[0],
                                              inset_relative_position[2],
                                              inset_relative_position[1] - inset_relative_position[0],
                                              inset_relative_position[3] - inset_relative_position[2]]
            ax_i_ins = ax_i.inset_axes(inset_x0y0wh_relative_position,
                                       xlim=(inset_x_range[0], inset_x_range[1]),
                                       ylim=(inset_y_range_axis_ij[0], inset_y_range_axis_ij[1]))
            ax_i.indicate_inset_zoom(ax_i_ins, edgecolor="black")
            #
            # And this... is cool... call itself again with the parameters of 'ax_i' (and minor differences)!
            plot_figures_of_merit_from_individual_experiment(
                df_experiment, subset, figure_of_merit_i,
                x_mode=x_mode, inset_x_range=None,
                grid=grid, print_experiment_info=False, cax=ax_i_ins
            )
            ax_i_ins.set_title(None)
            ax_i_ins.set_xlabel(None)
            ax_i_ins.get_legend().remove()
        pass
        #
    pass

    return ax


def plot_figures_of_merit_comparison_intra_experiment_group(
        df_multiexperiment_params_and_progress, subset, figure_of_merit,
        filtering_value=None, filtering_level=None,
        fields_to_include_in_legend=None, legend_fontsize=None,
        x_mode='batch', inset_x_range=None, inset_relative_position=(0.38, 0.98, 0.3, 0.6),
        grid=True, print_experiment_info=False, cax=None, figsize=None):
    """
    It plots, jointly, multiple different experiments encoded in ``df_multiexperiment_params_and_progress`` for their \
    visual comparison: such ``df_multiexperiment_params_and_progress`` corresponds, foreseeably, to the output \
    of the function :py:func:`.load_figures_of_merit_from_experiments`. The displayed
    figures of merit are displayed visually indicating the different epochs.

    It filters, optionally, the experiments in ``df_multiexperiment_params_and_progress`` according to the filter \
    specification given jointly by ``filtering_value`` and ``filtering_level``. For the retained experiments \
    after the filtering the function displays the figure of merit, or plurality thereof, \
    requested in ``figure_of_merit``, for the subset, or plurality thereof,  requested in ``subset`` \
    (the plurality is expressed by a number of strings in an iterable): depending on the combined number of \
    figures of merit and subsets the results would be displayed in a row or an array of axes.

    Additionally, the function allows to identify each of the displayed experiments in the legend of the function \
    using the field, or plurality thereof, requested in ``fields_to_include_in_legend``.

    Parameters
    ----------
    df_multiexperiment_params_and_progress : :py:class:`pandas.DataFrame`
    subset : str or list[str] or tuple[str]
        Single ``subset`` or multiple subsets in an iterable, among ``'train'`` and ``'val'``
    figure_of_merit : str or list[str] or tuple[str]
        Single ``figure_of_merit`` or multiple figures of merit in an iterable, among ``'loss'`` and ``'acc'``
    filtering_value, filtering_level : str or tuple[str], optional
        :py:meth:`pandas.DataFrame.xs`
    fields_to_include_in_legend : str or tuple[str], optional
        Fields from the different experiments to be included in the legend in order to individually identify \
        individual (filtered) experiments. If ``None`` the legend is not included.
        Default: ``None``
    legend_fontsize : int, optional
        Fontsize forced for the legend. If no value is provided (``None``) the function attempts to automatically \
        infer a reasonable size. \
        Default: ``None``
    x_mode : str, optional
        Value among ``'batch'``, ``'time'``, and ``'epoch'``. It selects which unit, whether the absolute index of the \
        batch, the elapsed time (in seconds), or the fraction of a complete epoch, is displayed on the x-axis of the \
        plot. \
        Default: ``'batch'``
    inset_x_range : tuple[int or float] or list[int or float], optional
        Definition of the x limits of the x-axis for an automatically generated (zoom) inset: 2D tuple/list. \
        Default: ``None`` (no inset)
    inset_relative_position : tuple[int or float] or list[int or float]
        Definition of the position of the inset in the images, which will be treated identically for all sub-axes. \
        Each dimension is $\\in \\[0,1\\]$ and their respective meaning is: *[x0, x1, y0, y1]*, \
        **and not position and with as in the definition** *[x0, y0, width, height]* **of** \
        :py:meth:`matplotlib.axes.Axes.inset_axes`. \
        Default: ``[0.38, 0.98, 0.3, 0.6]``
    grid : bool, optional
        Default: ``True``
    print_experiment_info : bool, optional
        If ``True`` the function prints a table with the parameters of the experiment. \
        Default: ``False``
    cax : :py:class:`matplotlib.axes.Axes`, optional
        Default: ``None``
    figsize : tuple[int], optional
        Default: ``None`` (equivalent to $6.4 \\times 4.8$ inches)

    Returns
    -------
    :py:class:`matplotlib.axes.Axes`
    """

    ##############################
    # React to the selected 'x_mode'
    ##############################

    modality_x_axis = None
    label_x_axis = None
    if x_mode == 'batch':
        modality_x_axis = 'abs_batch'
        label_x_axis = '(abs) batch number'
    elif x_mode == 'time':
        modality_x_axis = 'elapsed_time'
        label_x_axis = 'elapsed time'
    elif x_mode == 'epoch':
        # This option requires a certain re-normalization of the data: everything needs normalization
        # to the number of epochs, which requires a new column
        modality_x_axis = 'epoch_fraction'
        label_x_axis = 'epoch (and fraction)'
        # # Add column
        # df_multiexperiment_params_and_progress["epoch_fraction"] = \
        #     _calculate_series_epoch_fraction(df_multiexperiment_params_and_progress)
    else:
        raise Exception(f"Allowable values for argument 'x_mode' are: 'batch', 'time', and 'epoch'. Found: {x_mode}")
    pass

    ##############################
    # Check the format of "inset_x_range" if provided, and its size
    ##############################

    if inset_x_range is not None:
        if (isinstance(inset_x_range, tuple) or isinstance(inset_x_range, list)) and (len(inset_x_range) == 2):
            pass  # Alles gut
        else:
            raise Exception((f"The provided 'inset_x_range' is no 2D tuple/list or None, as expected, " +
                             f"but a {type(inset_x_range)}"))
        pass
    pass

    if (isinstance(inset_relative_position, tuple) or isinstance(inset_relative_position, list)) \
            and (len(inset_relative_position) == 4):
        pass  # Alles gut
    else:
        raise Exception((f"The provided 'inset_relative_position' is no 4D tuple/list or None, as expected, " +
                         f"but a {type(inset_relative_position)}"))
    pass

    ##############################
    # Working with the provided experiment filters and examine how many experiments are there
    ##############################

    # First: we filter the input "df_multiexperiment_df_params_and_progress" and see how many experiments are there
    df_filtered_multiexperiment = []
    # Are "filtering_value" and "filtering_level" None?
    if (filtering_value is None) != (filtering_level is None):
        raise Exception(
            f"'filtering_value' and 'filtering_level' are incompatible: one is 'None' but the other is not!")
    pass
    if filtering_value is None:  # Then both are
        df_filtered_multiexperiment = df_multiexperiment_params_and_progress
    else:
        filtering_value = tuple(filtering_value) if isinstance(filtering_value, list) else filtering_value
        filtering_level = tuple(filtering_level) if isinstance(filtering_level, list) else filtering_level
        df_filtered_multiexperiment = df_multiexperiment_params_and_progress.xs(filtering_value, level=filtering_level,
                                                                                drop_level=False)
    pass

    # Count the number of different experiments
    multiindices_individual_experiments = df_filtered_multiexperiment.index.droplevel(["epoch", "batch"]).unique()
    num_experiments = len(multiindices_individual_experiments)

    # If indicated, print some info about the selected experiments
    if print_experiment_info:
        print(f"num_experiments = {num_experiments}")
    pass

    ##############################
    # Checking how many subplots will be needed
    ##############################

    if isinstance(subset, str):
        subset = [subset]  # We make a list (we will process it later, by default, as a list)
    pass
    if isinstance(figure_of_merit, str):
        figure_of_merit = [figure_of_merit]  # We make a list (we will process it later, by default, as a list)
    pass
    if (fields_to_include_in_legend is not None) and isinstance(fields_to_include_in_legend, str):
        fields_to_include_in_legend = [fields_to_include_in_legend]  # We make a list out of it in any case
    pass

    # Define or reuse axes
    fig, ax, (nrows, ncols) = _create_figure_and_subplots(
        subset, figure_of_merit, cax=cax, figsize=figsize, same_figure_of_merit_along_row=True
    )

    ##############################
    # Setting the styles of the different experiments to compare (which will be reused cyclically),
    # the fonts of the legends, etc.
    ##############################

    list_of_style_tuples = [
        ('solid', 'solid'),  # Same as (0, ()) or '-'
        ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
        ('dashed', 'dashed'),  # Same as '--'
        ('dashdot', 'dashdot'),  # Same as '-.'
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]
    linewidth = 1.2
    #
    handles_plots = []
    #
    # Inference of font sizes!
    if fields_to_include_in_legend is not None:
        #
        figwidth = 6.4 if figsize is None else figsize[0]
        figheight = 4.8 if figsize is None else figsize[1]
        #
        fontsize_legend = legend_fontsize
        if fontsize_legend is None:
            effective_legend_width = 0.5 * figwidth / ncols
            effective_legend_height = 0.5 * figheight / nrows
            #
            # Considered average field character length and each line height
            field_char_length = 6
            field_char_height = 2
            fontsize_legend = min(
                min(int((effective_legend_width / (len(fields_to_include_in_legend) * field_char_length)) * 100.0),
                    int((effective_legend_height / (num_experiments * field_char_height)) * 100.0)),
                10
            )
        pass
        #
        #
        effective_extra_title_width = figwidth / ncols if fig is None else figwidth
        fontsize_extra_title = min(10, int((8.0 * effective_extra_title_width) / len(fields_to_include_in_legend)))
        #
        extra_title_legend_fields = "(Legend fields: " + "; ".join(fields_to_include_in_legend) + ")"
        if fig is not None:
            fig.suptitle(extra_title_legend_fields, fontsize=fontsize_extra_title, y=-0.002)
        pass
    pass

    ##############################
    # Display the different experiments and consider, if set, the inset
    ##############################

    for i, figure_of_merit_i in enumerate(figure_of_merit):
        ax_i = ax if len(figure_of_merit) == 1 else ax[i]
        for j, subset_j in enumerate(subset):
            ax_ij = ax_i if len(subset) == 1 else ax_i[j]
            #
            # We take each experiment separately
            legend_ax_ij = []
            #
            if inset_x_range is not None:
                inset_y_range_axis_ij = [np.inf, -np.inf]
            pass
            #
            for k, multiindex_individual_experiment in enumerate(multiindices_individual_experiments):
                #
                # DF individual experiment
                df_experiment_k = df_filtered_multiexperiment.xs(multiindex_individual_experiment,
                                                                 level=multiindices_individual_experiments.names)
                #
                linestyle_k = list_of_style_tuples[k % len(list_of_style_tuples)][1]
                ax_ij.set_prop_cycle(None)
                #
                for z, epoch in enumerate(df_experiment_k.index.unique(level='epoch')):
                    handle_plot = ax_ij.plot(
                        df_experiment_k.xs(epoch, level="epoch").loc[:, modality_x_axis],
                        df_experiment_k.xs(epoch, level="epoch").loc[:, (subset_j, figure_of_merit_i)],
                        linestyle=linestyle_k, linewidth=linewidth
                    )
                    if z == 0:
                        handles_plots.extend(handle_plot)
                    pass
                pass
                #
                if fields_to_include_in_legend is not None:
                    list_of_fields_for_legend = []
                    for field in fields_to_include_in_legend:
                        item_to_add_to_list = "-" if field not in multiindices_individual_experiments.names \
                            else str(multiindices_individual_experiments.get_level_values(field)[k])
                        list_of_fields_for_legend.append(item_to_add_to_list)
                    pass
                    legend_ax_ij.append("; ".join(list_of_fields_for_legend))
                pass
                #
                if inset_x_range is not None:
                    current_inset_y_range = _infer_inset_parameters(df_experiment_k, subset_j, figure_of_merit_i,
                                                                    inset_x_range[0], inset_x_range[1], x_mode=x_mode)
                    inset_y_range_axis_ij[0] = min(inset_y_range_axis_ij[0], current_inset_y_range[0])
                    inset_y_range_axis_ij[1] = max(inset_y_range_axis_ij[1], current_inset_y_range[1])
                pass
                #
            pass
            #
            if (nrows == 1) or (i == (nrows - 1)):  # So only the last row has the xlabel
                ax_ij.set_xlabel(label_x_axis)
            pass
            ax_ij.set_ylabel(figure_of_merit_i)
            ax_ij.set_title(f"{subset_j}, {figure_of_merit_i}", fontsize=12)
            ax_ij.grid(grid)
            extra_title = ""
            if fields_to_include_in_legend is not None:
                ax_ij.legend(handles_plots, legend_ax_ij, fontsize=fontsize_legend)
                if fig is None:
                    ax_ij.xaxis.set_label_position('top')
                    ax_ij.set_xlabel(extra_title_legend_fields, fontsize=fontsize_extra_title)
                pass
            pass
            #
            # Add an inset if indicated!!!
            if ((inset_x_range is not None) and
                    (not np.isnan(inset_y_range_axis_ij[0])) and (not np.isinf(inset_y_range_axis_ij[0]))):
                # print(f"inset_x_range = {inset_x_range}")
                # print(f"inset_y_range_axis_ij = {inset_y_range_axis_ij}")
                # The limits for the inset are: inset_x_range, inset_y_range_axis_ij
                # The position within the axis:
                inset_x0y0wh_relative_position = [inset_relative_position[0],
                                                  inset_relative_position[2],
                                                  inset_relative_position[1] - inset_relative_position[0],
                                                  inset_relative_position[3] - inset_relative_position[2]]
                ax_ij_ins = ax_ij.inset_axes(inset_x0y0wh_relative_position,
                                             xlim=(inset_x_range[0], inset_x_range[1]),
                                             ylim=(inset_y_range_axis_ij[0], inset_y_range_axis_ij[1]))
                ax_ij.indicate_inset_zoom(ax_ij_ins, edgecolor="black")
                #
                # And this... is cool... call itself again with the parameters of 'ax_ij' (and minor differences)!
                plot_figures_of_merit_comparison_intra_experiment_group(
                    df_multiexperiment_params_and_progress, [subset_j], [figure_of_merit_i],
                    filtering_value=filtering_value, filtering_level=filtering_level,
                    fields_to_include_in_legend=None,
                    x_mode=x_mode, grid=grid, cax=ax_ij_ins)
                #
                ax_ij_ins.set_title(None)
                ax_ij_ins.set_xlabel(None)
                ax_ij_ins.set_ylabel(None)
            pass
            #
        pass
    pass
    #
    return ax


def plot_figures_of_merit_comparison_inter_experiment_groups(
        df_multiexperiment_params_and_progress, experiment_group_dictionary, subset, figure_of_merit,
        errorbar=('ci', 95), groups_in_legend=True, legend_fontsize=None,
        x_mode='batch', inset_x_range=None, inset_y_range=None, inset_relative_position=None,
        grid=True, print_experiment_info=False, cax=None, figsize=None):
    """
    It plots jointly several experiments, ideally different (stochastic) realizations/runs of the \
    same conceptual experiment (e.g. different starting random points for the same network and dataset), \
    encoded in ``df_multiexperiment_params_and_progress`` (corresponding, foreseeably, to the output \
    of the function :py:func:`.load_figures_of_merit_from_experiments`). The plot of each experiment group will \
    show the average of all experiments for the same batch number as a continuous line and, optionally, \
    a confidence interval or other error measure (see information about the accepted error bars in \
    Seaborn in the documentation of `seaborn.lineplot <https://seaborn.pydata.org/generated/seaborn.lineplot.html>`_.

    The displayed figures of merit are displayed visually indicating the different epochs by simply a minor \
    discontinuity between the continuous line corresponding to each epoch: therefore, for a large amount of epochs,
    display under the options for ``'batch'`` and ``'time'`` for ``x_mode`` can be little informative about \
    epoch alignment. For a display intrinsically informative as regards epochs use ``x_mode`` = ``'epoch'``, \
    wherein the x-axis is normalized so each whole number corresponds to a whole epoch.

    Additionally, if the \
    argument ``inset_x_range`` is not ``None`` but a tuple or list of two elements, an inset will be added: \
    if the argument ``inset_y_range`` is not provided (=``None``) the function will automatically calculate \
    the "zooming range" in the *y* direction from the values registered in the indicated range for each one of \
    the involved plots; however, if ``inset_y_range`` is provided, it must be provided in the form of \
    a 2D tuple/list if `figure_of_merit` is one single value or a dictionary whose keys are coincident with the \
    values of `figure_of_merit` and whose values are 2D tuple/list intended \
    to be the y-dir zooming for said figure of merit.

    Each experiment group is indicated using a dictionary, so each different experiment group is entered as one \
    dictionary in the list/tuple of dictionaries ``experiment_group_dictionary``, wherein the keys represent \
    the respective level name of the ``df_multiexperiment_params_and_progress`` and the corresponding values \
    represent precisely so: e.g.

        ``experiment_group_dictionary[0] = {'conv_like_type': 'ibnn_lite', 'lambda_trainable': False}``

    If only one group of experiments is to be displayed, ``experiment_group_dictionary`` will be simply one \
    single dictionary.

    For the selected experiment group(s) the selected figure of merit, or plurality thereof, \
    requested in ``figure_of_merit``, for the subset, or plurality thereof,  requested in ``subset`` \
    (the plurality is expressed by a number of strings in an iterable) is/are displayed: depending on the combined \
    number of figures of merit and subsets the results would be displayed in a row or an array of axes.

    Additionally, the function allows to identify each of the displayed experiment group in the legend of the function \
    using the field, or plurality thereof, requested in ``fields_to_include_in_legend``.

    Parameters
    ----------
    df_multiexperiment_params_and_progress : :py:class:`pandas.DataFrame`
    experiment_group_dictionary : dict or tuple[dict] or list[dict]
    subset : str or list[str] or tuple[str]
        Single ``subset`` or multiple subsets in an iterable, among ``'train'`` and ``'val'``
    figure_of_merit : str or list[str] or tuple[str]
        Single ``figure_of_merit`` or multiple figures of merit in an iterable, among ``'loss'`` and ``'acc'``
    errorbar : str, (str, int) tuple, (str, float) tuple
        Name of errorbar method (either “ci”, “pi”, or others ), or a tuple with a method name and a level parameter. \
        It corresponds to the error bars in accepted by \
        `seaborn.lineplot <https://seaborn.pydata.org/generated/seaborn.lineplot.html>`_. If ``None``, no error bar \
        and only the bare line. \
        Default: ``('ci', 95)`` (confidence interval for $\\pm 2\\sigma$)
    groups_in_legend : bool, optional
        Whether the group definitions are to be included in the legend. \
        Default: True
    legend_fontsize : int, optional
        Fontsize forced for the legend. If no value is provided (``None``) the function attempts to automatically \
        infer a reasonable size. \
        Default: ``None``
    x_mode : str, optional
        Value among ``'batch'`` and ``'epoch'``. It selects which unit, whether the absolute index of the \
        batch or the fraction of a complete epoch, is displayed on the x-axis of the \
        plot. **WARNING: the option** ``'time'``, **originally available and using, for the x axis, the** \
        **elapsed time (in seconds), has been removed as an option because** \
        **it does not appear to sort timestamps well when grouping experiments!** \
        Default: ``'batch'``
    inset_x_range : tuple[int or float] or list[int or float], optional
        Definition of the x limits of the x-axis for an automatically generated (zoom) inset: 2D tuple/list. \
        If ``x_mode`` is ``'time'`` the values are indistinctly in seconds (``int`` or ``float``) or ``timedelta64``. \
        Default: ``None`` (no inset)
    inset_y_range : tuple[int or float] or list[int or float] or dict, optional
        Definition of the y limits of the y-axis for the generation of (zoom) inset: ``None`` indicates that the \
        range is to be estimated automatically; 2D tuple/list, if \
        `figure_of_merit` has one single element, or dict having the elements of `figure_of_merit` as keys and \
        corresponding 2D tuple/list as values. \
        Default: ``None`` (automatic range calculation)
    inset_relative_position : tuple[int or float] or list[int or float] or dict, optional
        Definition of the position of the inset in the images: if a 4D tuple/list is provided said values will \
        be used identically for all sub-axes; however, as with ``inset_y_range``, ``inset_relative_position`` also \
        accepts a dictionary with values (4D tuple/list) for each potential figure of merit. \
        Each dimension is $\\in [0,1]$ and their respective meaning is: *[x0, x1, y0, y1]*, \
        **and not position and with as in the definition** *[x0, y0, width, height]* **of** \
        :py:meth:`matplotlib.axes.Axes.inset_axes`. \
        Default: ``None`` (corresponding to ``{'loss': (0.38, 0.98, 0.5, 0.9), 'acc': (0.38, 0.98, 0.1, 0.5)}``)
    grid : bool, optional
        Default: ``True``
    print_experiment_info : bool, optional
        If ``True`` the function prints a table with the parameters of the experiment. \
        Default: ``False``
    cax : :py:class:`matplotlib.axes.Axes`, optional
        Default: ``None``
    figsize : tuple[int], optional
        Default: ``None`` (equivalent to $6.4 \\times 4.8$ inches)

    Returns
    -------
    :py:class:`matplotlib.axes.Axes`
    """

    ##############################
    # Uniformize the processing of the input arguments
    ##############################

    if isinstance(subset, str):
        subset = [subset]  # We make a list (we will process it later, by default, as a list)
    pass
    if isinstance(figure_of_merit, str):
        figure_of_merit = [figure_of_merit]  # We make a list (we will process it later, by default, as a list)
    pass

    ##############################
    # React to the selected 'x_mode'
    ##############################

    modality_x_axis = None
    label_x_axis = None
    if x_mode == 'batch':
        modality_x_axis = 'abs_batch'
        label_x_axis = '(abs) batch number'
    # elif x_mode == 'time':
    #     modality_x_axis = 'elapsed_time'
    #     label_x_axis = 'elapsed time'
    elif x_mode == 'epoch':
        # This option requires a certain re-normalization of the data: everything needs normalization
        # to the number of epochs, which requires a new column
        modality_x_axis = 'epoch_fraction'
        label_x_axis = 'epoch (and fraction)'
    else:
        # raise Exception(f"Allowable values for argument 'x_mode' are: 'batch', 'time', and 'epoch'. Found: {x_mode}")
        raise Exception(f"Allowable values for argument 'x_mode' are: 'batch' and 'epoch'. Found: {x_mode}")
    pass

    # Add the column 'epoch_fraction' if it was not part of the DataFrame
    if 'epoch_fraction' not in df_multiexperiment_params_and_progress.columns:
        df_multiexperiment_params_and_progress["epoch_fraction"] = \
            _calculate_series_epoch_fraction(df_multiexperiment_params_and_progress)
    pass

    ##############################
    # Check the format of "inset_x_range" and its size if provided
    ##############################

    if inset_x_range is not None:
        if not (isinstance(inset_x_range, (tuple, list)) and len(inset_x_range) == 2):
            raise Exception((f"The provided 'inset_x_range' is no 2D tuple/list or None, as expected, " +
                             f"but a {type(inset_x_range)}"))
        elif x_mode == 'time':
            formatted_inset_x_range = inset_x_range
            for i, t in enumerate(inset_x_range):
                if isinstance(t, (int, float)):
                    formatted_inset_x_range[i] = np.timedelta64(t, 's')
                elif not isinstance(t, np.timedelta64):
                    raise Exception(
                        (f"The provided 'inset_x_range' is no 2D tuple/list of int/float or None, as expected, " +
                         f"but a {type(inset_x_range)}"))
                pass
            pass
            inset_x_range = tuple(formatted_inset_x_range)
        pass
    pass

    ##############################
    # Check the format of "inset_y_range" and its size if provided, and format it into dict
    ##############################

    if not isinstance(inset_y_range, dict):
        dict_inset_y_range = {}
        for figure_of_merit_i in figure_of_merit:
            dict_inset_y_range[figure_of_merit_i] = copy.deepcopy(inset_y_range)
        pass
        inset_y_range = dict_inset_y_range
    pass

    for figure_of_merit_i in figure_of_merit:
        if not ((inset_y_range[figure_of_merit_i] is None) or
                (isinstance(inset_y_range[figure_of_merit_i], (tuple, list)) and
                 len(inset_y_range[figure_of_merit_i]) == 2)):
            raise Exception((f"The provided 'inset_y_range' is no 2D tuple/list or None, as expected, " +
                             f"but a {type(inset_y_range[figure_of_merit_i])}"))
        pass
    pass

    ##############################
    # Check the format of "inset_relative_position" and its size if provided, and format it into dict
    # (analogous to "inset_y_range")
    ##############################

    if inset_relative_position is None:
        inset_relative_position = {'loss': (0.38, 0.98, 0.5, 0.9), 'acc': (0.38, 0.98, 0.1, 0.5)}
    elif not isinstance(inset_relative_position, dict):
        dict_inset_relative_position = {}
        for figure_of_merit_i in figure_of_merit:
            dict_inset_relative_position[figure_of_merit_i] = copy.deepcopy(inset_relative_position)
        pass
        inset_relative_position = dict_inset_relative_position
    pass

    for figure_of_merit_i in figure_of_merit:
        if not (isinstance(inset_relative_position[figure_of_merit_i], (tuple, list)) and
                len(inset_relative_position[figure_of_merit_i]) == 4):
            raise Exception((f"The provided 'inset_relative_position' is no 4D tuple/list or None, as expected, " +
                             f"but a {type(inset_relative_position[figure_of_merit_i])}"))
        pass
    pass

    ##############################
    # First: we adapt to list and check if all the requested groups are valid
    ##############################

    if isinstance(experiment_group_dictionary, dict):
        # Only one dict: make it a list with one single experiment
        experiment_group_dictionary = [experiment_group_dictionary]
    elif (isinstance(experiment_group_dictionary, list) or isinstance(experiment_group_dictionary, tuple)) and \
            isinstance(experiment_group_dictionary[0], dict):
        # If list or tuple, make list
        experiment_group_dictionary = list(experiment_group_dictionary)
    pass

    valid_experiment_group_dictionary = []
    for experiment_group_dictionary_i in experiment_group_dictionary:
        flag_valid_group = True
        for key in experiment_group_dictionary_i:
            if not (key in df_multiexperiment_params_and_progress.index.names):
                print((f"WARNING: A field '{key}' requested for an entry of a group in 'experiment_group_dictionary' " +
                       f"is not included in the index of the input DataFrame. " +
                       f"As a result the group corresponding to the following entry will be ignored:\n" +
                       f"{experiment_group_dictionary_i}"))
                flag_valid_group = False
                break
            elif not (experiment_group_dictionary_i[key] in \
                      df_multiexperiment_params_and_progress.index.get_level_values(key)):
                # print((f"WARNING: The queried value {experiment_group_dictionary_i[key]} for field '{key}' " +
                #        f"requested for an entry of 'experiment_group_dictionary' " +
                #        f"is not included in the index of the input DataFrame. " +
                #        f"As a result the group corresponding to the following entry will be ignored:\n" +
                #        f"{experiment_group_dictionary_i}"))
                flag_valid_group = False
                break
            pass
        pass
        if flag_valid_group:
            valid_experiment_group_dictionary.append(experiment_group_dictionary_i)
    pass

    ##############################
    # Checking how many subplots will be needed
    ##############################

    # Define or reuse axes
    fig, ax, (nrows, ncols) = _create_figure_and_subplots(
        subset, figure_of_merit, cax=cax, figsize=figsize, same_figure_of_merit_along_row=True
    )

    # Inference of font sizes!
    if groups_in_legend:
        #
        figwidth = 6.4 if figsize is None else figsize[0]
        figheight = 4.8 if figsize is None else figsize[1]
        #
        fontsize_legend = legend_fontsize
        if fontsize_legend is None:
            effective_legend_width = 0.9 * figwidth / ncols
            effective_legend_height = 0.5 * figheight / nrows
            #
            # Considered average field character length and each line height
            max_experiment_group_definition_length = 0
            for experiment_group_dictionary_i in valid_experiment_group_dictionary:
                max_experiment_group_definition_length = max(
                    len(str(experiment_group_dictionary_i)), max_experiment_group_definition_length
                )
            pass
            field_char_height = 2
            fontsize_legend = min(
                min(int((effective_legend_width / max_experiment_group_definition_length) * 100.0),
                    int((effective_legend_height / (
                            len(valid_experiment_group_dictionary) * field_char_height)) * 100.0)),
                10
            )
        pass
        #
    pass

    ##############################
    # Treating the different experiment groups: filter the input "df_multiexperiment_df_params_and_progress"
    ##############################

    list_tableau_colors = list(mcolors.TABLEAU_COLORS.keys())

    for i, figure_of_merit_i in enumerate(figure_of_merit):
        ax_i = ax if len(figure_of_merit) == 1 else ax[i]
        for j, subset_j in enumerate(subset):
            ax_ij = ax_i if len(subset) == 1 else ax_i[j]
            #
            handles_plots = []
            legend_ax_ij = []
            #
            if inset_x_range is not None:
                # If not given calculate it from the data
                inset_y_range_axis_ij = []
                if inset_y_range[figure_of_merit_i] is None:  # If not given calculate it from the data
                    inset_y_range_axis_ij = _infer_inset_parameters_experiment_groups(
                        df_multiexperiment_params_and_progress, valid_experiment_group_dictionary,
                        subset=subset_j, figure_of_merit=figure_of_merit_i,
                        x_min=inset_x_range[0], x_max=inset_x_range[1], x_mode=x_mode)
                else:
                    inset_y_range_axis_ij = inset_y_range[figure_of_merit_i]
                pass
            pass
            #
            for eg, experiment_group_dictionary_i in enumerate(valid_experiment_group_dictionary):
                #
                # Filter the group experiment
                df_experiment_group_i = df_multiexperiment_params_and_progress.xs(
                    tuple(experiment_group_dictionary_i.values()),
                    level=tuple(experiment_group_dictionary_i.keys()),
                    drop_level=False
                )

                # Num. of experiments in group (for annotation)
                num_experiments = len(df_experiment_group_i.index.droplevel(['epoch', 'batch']).unique())
                str_describing_experiment_group = f"{num_experiments} x {str(experiment_group_dictionary_i)}"
                if print_experiment_info:
                    print(str_describing_experiment_group)
                pass

                if num_experiments > 0:
                    #
                    sns.lineplot(data=df_experiment_group_i,
                                 x=modality_x_axis, y=(subset_j, figure_of_merit_i),
                                 color=list_tableau_colors[eg % len(list_tableau_colors)],
                                 errorbar=errorbar, sort=True, ax=ax_ij)
                    #
                    for z, epoch in enumerate(df_experiment_group_i.index.unique(level='epoch')):
                        # sns.lineplot(data=df_experiment_group_i.xs(epoch, level="epoch").reset_index(),
                        #                            x=modality_x_axis, y=(subset_j, figure_of_merit_i),
                        #                            color=list_tableau_colors[eg % len(`plist_tableau_colors)],
                        #                            errorbar=errorbar, ax=ax_ij)
                        #
                        df_experiment_group_i_end_of_epoch = df_experiment_group_i.set_index(
                            [df_experiment_group_i.index, "epoch_fraction"]
                        ).xs(float(epoch), level="epoch_fraction", drop_level=False)
                        #
                        sns.lineplot(
                            data=df_experiment_group_i_end_of_epoch,
                            x=modality_x_axis, y=(subset_j, figure_of_merit_i),
                            marker='o',
                            color=list_tableau_colors[eg % len(list_tableau_colors)],
                            ax=ax_ij)
                        #
                    pass
                    #
                    legend_ax_ij.append(str_describing_experiment_group)
                    #
                    # Artist (proxy artist, in fact) for the legend
                    proxy_line_for_legend = mlines.Line2D(
                        [], [], color=list_tableau_colors[eg % len(list_tableau_colors)],
                        marker=None, markersize=15
                    )
                    #
                    handles_plots.append(proxy_line_for_legend)
                    #
                    # if inset_x_range is not None:
                    #     inset_y_range_axis_ij = inset_y_range[figure_of_merit_i]
                    #     if inset_y_range_axis_ij is None:
                    #         current_inset_y_range = _infer_inset_parameters(df_experiment_group_i, subset_j,
                    #                                                         figure_of_merit_i,
                    #                                                         inset_x_range[0], inset_x_range[1],
                    #                                                         x_mode=x_mode)
                    #         inset_y_range_axis_ij[0] = min(inset_y_range_axis_ij[0], current_inset_y_range[0])
                    #         inset_y_range_axis_ij[1] = max(inset_y_range_axis_ij[1], current_inset_y_range[1])
                    #     pass
                    # pass

                pass

            pass
            #
            ax_ij.set_xlabel(label_x_axis)
            ax_ij.set_title(f"{subset_j}, {figure_of_merit_i}", fontsize=12)
            ax_ij.grid(grid)
            #
            if groups_in_legend:
                ax_ij.legend(handles_plots, legend_ax_ij, fontsize=fontsize_legend)
            pass
            #
            # Add an inset if indicated!!!
            if ((inset_x_range is not None) and
                    (not np.isnan(inset_y_range_axis_ij[0])) and (not np.isinf(inset_y_range_axis_ij[0]))):
                # print(f"inset_x_range = {inset_x_range}")
                # print(f"inset_y_range_axis_ij = {inset_y_range_axis_ij}")
                # The limits for the inset are: inset_x_range, inset_y_range_axis_ij
                # The position within the axis:

                inset_relative_position_ij = inset_relative_position[figure_of_merit_i]

                inset_x0y0wh_relative_position = [inset_relative_position_ij[0],
                                                  inset_relative_position_ij[2],
                                                  inset_relative_position_ij[1] - inset_relative_position_ij[0],
                                                  inset_relative_position_ij[3] - inset_relative_position_ij[2]]
                if x_mode != 'time':
                    ax_ij_ins = ax_ij.inset_axes(
                        inset_x0y0wh_relative_position,
                        xlim=(inset_x_range[0], inset_x_range[1]),
                        ylim=(inset_y_range_axis_ij[0], inset_y_range_axis_ij[1])
                    )
                else:
                    ax_ij_ins = ax_ij.inset_axes(
                        inset_x0y0wh_relative_position,
                        xlim=(inset_x_range[0].astype('float'), inset_x_range[1].astype('float')),
                        ylim=(inset_y_range_axis_ij[0], inset_y_range_axis_ij[1])
                    )
                pass
                ax_ij.indicate_inset_zoom(ax_ij_ins, edgecolor="black")
                #
                # And this... is cool... call itself again with the parameters of 'ax_ij' (and minor differences)!
                plot_figures_of_merit_comparison_inter_experiment_groups(
                    df_multiexperiment_params_and_progress, valid_experiment_group_dictionary,
                    [subset_j], [figure_of_merit_i],
                    errorbar=errorbar, groups_in_legend=False, legend_fontsize=None,
                    x_mode=x_mode, inset_x_range=None, grid=grid, cax=ax_ij_ins)
                #
                ax_ij_ins.set_title(None)
                ax_ij_ins.set_xlabel(None)
                ax_ij_ins.set_ylabel(None)
            pass
            #
        pass
        #
    pass

    return ax


def load_figures_of_merit_from_experiments(filenames_experiments):
    """
    Loader of the files containing the figures of merit of the experiments. The function accepts either a string \
    or a list of strings, and for (each of) the string(s):

    - If it is a file, it loads the file using :py:func:`pandas.read_pickle`.

    - If it is a folder: (1) it inspects if it contains files with the extension ``*.pkl`` and loads them using \
      :py:func:`pandas.read_pickle`; (2) it inspects recursively all the folder tree down the folder: it checks \
      whether the folder contains folders containing ``*.pkl`` files, and extracts their names/paths as well.

    Parameters
    ----------
    filenames_experiments : str or list[str]
        Strings will be interpreted as files or folders depending on the exploration for the given name.

    Returns
    -------
    :py:class:`pandas.DataFrame`
    """

    def _pkl_names_from_tentative_name(tentative_name: str):
        """
        The function provides a list containing filenames:
        1) It returns a list with only `tentative_name` if it happens to be a file;
        2) It unravels the list of contained ``*.pkl`` files if the provided ``tentative_name`` is not a file but \
           a folder. It checks also if it contains folders one level down, and does the same for those child folders.
        In either case it returns a list of strings (or an empty list, if it is a folder without ``*.pkl`` files).

        Parameters
        ----------
        tentative_name : str

        Returns
        -------
        list[str]
        """
        list_of_filenames = []
        if os.path.isfile(tentative_name):
            list_of_filenames = [os.path.abspath(tentative_name)]
        elif os.path.isdir(tentative_name):
            list_of_child_filenames = [os.path.abspath(os.path.join(tentative_name, f)) \
                                       for f in os.listdir(tentative_name)]
            list_of_filenames = []
            for child_filename in list_of_child_filenames:
                if os.path.isfile(child_filename) and (os.path.splitext(child_filename)[-1] == '.pkl'):
                    list_of_filenames.extend([child_filename])
                elif os.path.isdir(child_filename):
                    list_of_filenames.extend(_pkl_names_from_tentative_name(child_filename))
                pass
        else:
            raise Exception(f"The provided name does not represent either a file or a folder: {tentative_name}")
        pass
        return list_of_filenames

    pass

    # If the argument "filenames_experiments" is just a string, make it a list of one element anyway
    if isinstance(filenames_experiments, str):
        filenames_experiments = [filenames_experiments]
    pass

    # Extract the .pkl files of each entry in the list "filenames_experiments" using the auxiliary function
    total_list_of_filenames_experiments = []
    for filename_experiment in filenames_experiments:
        total_list_of_filenames_experiments.extend(_pkl_names_from_tentative_name(filename_experiment))
    pass
    # Make it unique
    total_list_of_filenames_experiments = np.unique(np.array(total_list_of_filenames_experiments)).tolist()

    # Read all the DF to be merged!
    list_df_params_and_progress = [
        pd.read_pickle(filename_experiment) for filename_experiment in total_list_of_filenames_experiments
    ]

    # Create the index names to be reinjected in the concatenated/merged DF:
    merged_index_names = []
    for df_params_and_progress in list_df_params_and_progress:
        merged_index_names.extend(list(df_params_and_progress.index.names))
    pass
    merged_index_names = np.unique(np.array(merged_index_names)).tolist()

    # Change the order (messed-up by "np.unique") to make sure that the first and last levels are meaningful
    last_index_names = ["epoch", "batch"]
    first_index_names = ["start_training", "end_training", "host", "dataset_name", "conv_like_type"]
    for last_index_name in last_index_names:
        if last_index_name in merged_index_names:
            merged_index_names.remove(last_index_name)
            merged_index_names.append(last_index_name)
        pass
    pass
    for first_index_name in first_index_names[-1::-1]:
        if first_index_name in merged_index_names:
            merged_index_names.remove(first_index_name)
            merged_index_names.insert(0, first_index_name)
        pass
    pass

    # Total combined DF
    df_multiexperiment_params_and_progress = pd.concat(
        [df_params_and_progress.reset_index() for df_params_and_progress in list_df_params_and_progress],
        ignore_index=True
    ).set_index(merged_index_names)

    return df_multiexperiment_params_and_progress


def register_stats_hooks(model, stats_tracker):
    """
    Registra hooks en todas las capas relevantes para capturar estadísticas.
    """

    def hook_fn(module, input, output, name=None):
        # Solo registrar estadísticas si el módulo tiene nombre
        if name is None:
            name = module.__class__.__name__

        # Calculamos estadísticas del output
        if isinstance(output, torch.Tensor):
            mean = output.mean().item()
            std = output.std().item()
            max = output.max().item()
            min = output.min().item()
            stats_tracker.update(name, mean, std, max, min)

    # Registramos hooks en todos los módulos
    for this_name, module in model.named_modules():
        if "SM" in module.__class__.__name__:  # Solo si el módulo tiene un nombre (evitamos el módulo raíz)
            module.register_forward_hook(
                lambda mod, inp, out, name=module.__class__.__name__ + this_name: hook_fn(mod, inp, out, name)
            )


class LayerStatsTracker:
    def __init__(self):
        self.layer_stats = {}  # Diccionario para estadísticas por capa

    def update(self, layer_name, mean, std, max=None, min=None):
        """Actualiza estadísticas para una capa específica."""
        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = {'mean': [], 'std': [], 'max': [], 'min': []}

        self.layer_stats[layer_name]['mean'].append(mean)
        self.layer_stats[layer_name]['std'].append(std)
        self.layer_stats[layer_name]['max'].append(max)
        self.layer_stats[layer_name]['min'].append(min)

    def log_to_mlflow(self, epoch=None):
        """Registra estadísticas en MLflow."""
        import mlflow
        import numpy as np

        for layer_name, stats in self.layer_stats.items():
            mean_values = np.array(stats['mean'])
            std_values = np.array(stats['std'])
            max_values = np.array(stats['max'])
            min_values = np.array(stats['min'])

            prefix = f"epoch_{epoch}_" if epoch is not None else ""

            # Registrar estadísticas de la capa
            mlflow.log_metric(f"{prefix}{layer_name}_mean", float(mean_values.mean()))
            mlflow.log_metric(f"{prefix}{layer_name}_std", float(std_values.mean()))
            mlflow.log_metric(f"{prefix}{layer_name}_max", float(max_values.max()))
            mlflow.log_metric(f"{prefix}{layer_name}_min", float(min_values.min()))

    def reset(self):
        """Reinicia el tracker para la siguiente época."""
        self.layer_stats = {}


def run_experiment_group_from_toml_file(
        experiment_group_specification_file, function_run_individual_experiment_from_dict,
        function_deactivate_irrelevant_parameters_in_dict=None,
        verbose='medium'):
    """
    This function opens the indicated file `experiment_group_specification_file` containing a \
    TOML group experiment specification and runs the experiments resulting from the Cartesian product of all the \
    combinations suggested therein, combined with the base specification provided in the field \
    'base_experiment_specification' of `experiment_group_specification_file`. Later, each individual experiment \
    is run by the function `function_run_individual_experiment_from_dict`, which has to accept \
    the following parameters:

        - the individual experiment specification as a dictionary;
        - the experiment name, which is the value of the field 'experiment_name' in the group specification file; and
        - the verbosity level, which is a string that can be 'high', 'medium' (default), 'low', or 'none'.

    The function `function_deactivate_irrelevant_parameters_in_dict` can be provided to deactivate the \
    hyperparameters not relevant for the specific experiment, which is useful to avoid duplicated experiments \
    resulting from the Cartesian product of the hyperparameters. If not provided, or if the provided \
    `function_deactivate_irrelevant_parameters_in_dict` is not effective deactivating useless hyperparameters, it is \
    likely that duplicated experiments will occur.

    The parameters comprised in the file `experiment_group_specification_file` are all TOML arrays; \
    each value of the array corresponds to a different experiment; \
    the combination of multiple parameters having multiple values, using \
    the Cartesian product, results in the total number of experiments to be run. A

    Parameters
    ----------
    experiment_group_specification_file : str
        The path to the group experiment specification file, which is a TOML file containing the \
        hyperparameters to be used in the experiments of the group
    function_run_individual_experiment_from_dict : callable
        Function to run the individual experiment, which has to accept the individual experiment \
        specification as a dictionary. The function is expected to run the experiment and log it in MLFlow, if so \
        indicated
    function_deactivate_irrelevant_parameters_in_dict : callable, optional
        Function to deactivate the hyperparameters not relevant for the specific experiment, \
        which is useful to avoid duplicated experiments resulting from the Cartesian product of the hyperparameters.
        Default: ``None``, which means that no deactivation is performed
    verbose : str, optional
        Default: ``'medium'``
    """

    ######################################
    ######################################
    # Resolve the group experiment configuration file into a list of dictionaries wherein each dictionary contains
    # the base experiment configuration with the specific set of hyperparameters of the i-th experiment incorporated
    ######################################
    ######################################

    dict_general_info_group_experiment, \
        list_of_individual_dict_experiment_specifications_pre_filtering, \
        list_of_paths_to_existing_hyperparameter_elements \
        = read_experiment_group_specification_file(experiment_group_specification_file)

    ######################################
    ######################################
    # REMOVING REDUNDANCY IN THE EXPERIMENTS RESULTING FROM THE CARTESIAN PRODUCT OF THE HYPERPARAMETERS
    #######################################
    # USAGE:
    #   1) Go through the list of dictionaries with the hyperparameter combinations and put to zero, None, or False \
    #      the hyperparameters not relevant for the specific experiment
    #   2) Run the function "duplicated_experiment_specification_removal" to remove duplicated entries in such list
    ######################################
    ######################################

    ######################################
    # 1) Go through the list of dictionaries with the hyperparameter combinations and remove irrelevant combinations
    ######################################

    if function_deactivate_irrelevant_parameters_in_dict is not None and \
            callable(function_deactivate_irrelevant_parameters_in_dict):
        #
        for i, raw_dict_experiment_specification_i in enumerate(
                list_of_individual_dict_experiment_specifications_pre_filtering):
            list_of_individual_dict_experiment_specifications_pre_filtering[i] = \
                function_deactivate_irrelevant_parameters_in_dict(
                    list_of_individual_dict_experiment_specifications_pre_filtering[i]
                )
        pass
        #
    pass

    ######################################
    # 2) Run the function "duplicated_experiment_specification_removal"
    ######################################

    list_of_individual_dict_experiment_specifications_after_filtering = duplicated_experiment_specification_removal(
        list_of_individual_dict_experiment_specifications_pre_filtering
    )

    ######################################
    ######################################
    # Perform the individual experiments!
    ######################################
    ######################################

    # Print the summary of the resulting batch of experiments
    num_filtered_combinations = len(list_of_individual_dict_experiment_specifications_after_filtering)
    print(f"    - Total number of hyperparameter combinations after filtering: {num_filtered_combinations:6d}")

    # And run the individual experiments
    for i, dict_experiment_specification_i in enumerate(
            list_of_individual_dict_experiment_specifications_after_filtering
    ):
        #
        print(f"\n")
        print(f"*****************************************************************************")
        print(f"*** {i + 1:5d}-th EXPERIMENT ({i + 1}/" +
              f"{num_filtered_combinations}, {100.0 * i / num_filtered_combinations:5.3f} %)")
        # print(f"*****************************************************************************")
        # for key in dict_experiment_specification_i:
        #     print(f"***    {key} = {dict_experiment_specification_i[key]}")
        # pass
        print(f"*****************************************************************************")
        for path_to_existing_hyperparameter_element in list_of_paths_to_existing_hyperparameter_elements:
            hyperparameter_value_in_experiment_specification_i = get_multilevel_dict_element(
                dict_experiment_specification_i, path_to_existing_hyperparameter_element,
                default_returned_element='<DOES NOT APPLY>'
            )
            print(f"***    {'.'.join(path_to_existing_hyperparameter_element)} = " +
                  f"{hyperparameter_value_in_experiment_specification_i}")
        pass
        print(f"*****************************************************************************")
        print("")
        #
        # Pass the specification of the individual experiment for training
        function_run_individual_experiment_from_dict(
            dict_experiment_specification_i,
            dict_general_info_group_experiment,
            verbose=verbose)
        #
    pass


def main_entry_point_parser_to_experiment_function(parser: argparse.ArgumentParser, function):
    """
    It parses the arguments in `parser` according to the defined rules, which includes one or more \
    experiment specification files, and other optional arguments with a prefix. In particular:

    - <experiment_specification_file_01 experiment_specification_file_02 ...>.
    - `-m` or `--mlflow_logging` for the address of the MLflow server: it defaults to `http://161.111.21.41:80`.
    - `-v` or `--verbosity` for the level of verbosity of the subsequent logs: it defaults to `'medium'`.

    If one or several experiment specification files are (validly) read, a loop processing each one \
    of them using the callable argument function `function` is started. The `function` has to accept \
    the experiment name and the verbosity level as arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add the arguments to.
    function : callable
        The function to call with the arguments of the parser.
    """

    assert isinstance(parser, argparse.ArgumentParser), \
        "The argument `parser` must be an instance of `argparse.ArgumentParser`."
    assert callable(function), "The argument `function` must be a callable function."

    # Add the arguments to the parser
    parser.add_argument("experiment_specification_files", nargs='*',
                        metavar="<experiment_specification_file_01 experiment_specification_file_02 ...>",
                        help=("Each (non-prefixed) argument is regarded as an specification file " +
                              "triggering a (group of) new experiment(s)."))
    # parser.add_argument("-m", "--mlflow_logging", default="http://161.111.21.41:80")
    parser.add_argument("-m", "--mlflow_logging", default="vmg")
    parser.add_argument('-v', '--verbosity', default='medium',
                        metavar="verbosity_level",
                        help=("Level, comprised among 'high', 'medium' (default), 'low', and 'none', " +
                              "indicating the level of detail of the progress logs printed during the experiment."))
    #
    args = parser.parse_args()
    #
    if len(args.experiment_specification_files) < 1:
        raise Exception(f"At least one experiment specification file (positional argument) must be provided!")
    else:
        print(f"\n\n\n")
        print(f"*****************************************************************************")
        print(f"***")
        print(f"***  TOTAL OF {len(args.experiment_specification_files)} GROUP EXPERIMENT FILES TO PROCESS:")
        print(f"***")
        for experiment_specification_file in args.experiment_specification_files:
            print(f"***  {experiment_specification_file}")
        pass
        print(f"***")
        print(f"*****************************************************************************")
        print(f"\n\n\n")
        #
        for ind, experiment_specification_file in enumerate(args.experiment_specification_files):
            print(f"\n\n\n")
            print(f"*****************************************************************************")
            print(f"*****************************************************************************")
            print(f"***")
            print(
                f"***  GROUP EXPERIMENT FILE {ind + 1}/{len(args.experiment_specification_files)}: {experiment_specification_file}")
            print(f"***")
            print(f"*****************************************************************************")
            print(f"*****************************************************************************")
            print(f"\n\n")
            function(experiment_specification_file, mlflow_logging=args.mlflow_logging, verbose=args.verbosity)
        pass
    pass
