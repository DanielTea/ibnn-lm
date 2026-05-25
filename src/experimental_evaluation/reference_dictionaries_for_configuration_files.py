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
import copy
import builtins

from experimental_evaluation.configuration_file_reading_utils import \
    get_keypaths_exisiting_dict_elements, get_multilevel_dict_element, set_multilevel_dict_element, indented_print_dict

########################################################################################################################
########################################################################################################################
# AUXILIARY FUNCTIONS
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Auxiliary function getting a type, or a tuple of types, from a string or a list of strings
########################################################################################################################
def get_type_from_string(type_name):
    #
    if isinstance(type_name, str):
        return getattr(builtins, type_name)
    elif isinstance(type_name, list) and all(isinstance(s, str) for s in type_name):
        return tuple([getattr(builtins, s) for s in type_name])
    else:
        raise ValueError("Input must be a string or a list of strings representing built-in types.")

########################################################################################################################
# Auxiliary function getting through a dict with strings meaning types and returning a dict with the actual types
########################################################################################################################
def scaffolding_toml_to_dict(scaffolding_toml_string):
    #
    parsed_scaffolding_with_str = tomli.loads(scaffolding_toml_string)
    parsed_scaffolding_with_types = copy.deepcopy(parsed_scaffolding_with_str)
    for elem_path in get_keypaths_exisiting_dict_elements(parsed_scaffolding_with_types):
        set_multilevel_dict_element(
            parsed_scaffolding_with_types,
            elem_path,
            get_type_from_string(
                get_multilevel_dict_element(parsed_scaffolding_with_types, elem_path, None)
            )
        )
    pass
    #
    return parsed_scaffolding_with_types
########################################################################################################################


########################################################################################################################
########################################################################################################################
# TOML FILES CONTAINING THE SCAFFOLDING (FIELDS + TYPES) FOR EACH (important) BLOCK OF THE CONFIGURATION FILES
########################################################################################################################
########################################################################################################################

_group_file_header_scaffolding_toml = \
    """
    experiment_name = 'str'
    purpose = 'str'
    base_experiment_specification_file = 'str'

    mlflow_logging = 'bool'
    local_log_folder = ['str', 'bool']
    """

_experiment_name_to_retrieve_scaffolding_toml = \
    """
    experiment_name_to_retrieve = 'str'
    """

_classifier_scaffolding_toml = \
    """
    [classifier]
    
        net = 'str'
    
        conv_like_type = 'str'
        conv_like_type_position = 'str'
    
        prenormalization = 'bool'
    
        [classifier.architecture_specific]
            # CLASSIFIER-SPECIFIC PARAMETERS!!!
            ###############################
            # FOR 'multi_layer':
            ###############################
            [classifier.architecture_specific.multi_layer]
                num_conv_like_layers =  'int'
                channels_per_conv_layer = ['list', 'int']
                m_kernel_size_per_conv_layer = ['list', 'bool']
                phi_activation_per_conv_layer = ['list', 'bool']
                batch_normalization_per_conv_layer = ['list', 'bool']
                maxpool_reduction_per_conv_block = ['list', 'bool']
            ###############################
            # FOR 'vggx':
            ###############################
            [classifier.architecture_specific.vggx]
                conv_block_specification = 'list'
                base_channels =  'int'
                m_kernel_size_per_conv_layer = ['list', 'bool']
            ###############################
            # FOR 'alexnet':
            ###############################
            [classifier.architecture_specific.alexnet]
                conv_block_specification = 'list'
                channels_per_conv_layer = ['list', 'int']
                m_kernel_size_per_conv_layer = ['list', 'bool']
            ###############################
            # FOR 'efficientnetv2s':
            ###############################
            [classifier.architecture_specific.efficientnetv2s]
                m_kernel_size_per_conv_layer = ['list', 'bool']
                num_hidden_channels_2 = 'int'
    
        [classifier.fully_connected]
        
            fc_num_layers = 'int'
            fc_num_units_intermediate_layers = 'int'
            fc_batch_normalization = 'bool'
            fc_dropout = ['int', 'float']
            penciled_decision = 'bool'
            softmax_output = 'bool'
    
        [classifier.conv_like_layer]
        
            [classifier.conv_like_layer.traditional]
            
                overall_batch_normalization_conv_layers = 'bool'
                overall_maxpool_reduction_conv_blocks = 'int'
                
                overall_phi_activation_conv_layers = 'str'
                overall_m_kernel_size_conv_layers = ['int', 'float']
                
                m_independent_channels = 'bool'
                m_padding = 'str'
                m_padding_mode = 'str'
                m_initialization = 'str'
                m_trainable = 'bool'
                
                b_type = 'str'
                initial_b = ['int', 'float', 'list']
                b_trainable = 'bool'
        
            [classifier.conv_like_layer.nonlinear_bias]
            
                sigma_activation = 'str'
                
                sigma_x_compress = ['int', 'float']
                sigma_y_stretch = ['int', 'float']
                sigma_x_offset = ['int', 'float']
                sigma_y_offset = ['int', 'float']
                sigma_x_compress_trainable = 'bool'
                sigma_y_stretch_trainable = 'bool'
                sigma_x_offset_trainable = 'bool'
                sigma_y_offset_trainable = 'bool'
                
                lambda_type = 'str'
                initial_lambda = ['int', 'float', 'list']
                lambda_trainable = 'bool'
                
                w_kernel_size = ['int', 'float']
                w_independent_channels = 'bool'
                w_padding_mode = 'str'
                w_initialization = 'dict'
                w_trainable = 'bool'
            
                [classifier.conv_like_layer.nonlinear_bias.cross_conv_computation]
                
                    calculation_mode = 'str'
                    num_sampling_points = 'int'
                    range_std_sigma = 'list'
                    memory_saving_version = 'bool'
                
            [classifier.conv_like_layer.fixed_point]
            
                batched_fixed_point = 'bool'
                
                f_tau = ['int', 'float']
                f_solver = 'str'
                f_max_iter = 'int'
                f_tol = ['int', 'float']
                
                b_solver = 'str'
                b_max_iter = 'int'
                b_tol = ['int', 'float']
                
                abs_error_threshold = ['int', 'float']
    """

_dataset_scaffolding_toml = \
    """
    [dataset]
        name = 'str'
        colorspace = 'str'
        force_im_size = ['list', 'bool']
        train_proportion = ['int', 'float']
        val_proportion = ['int', 'float']
        batch_size = 'int'
        generator_seed = ['int', 'bool']
        root_folder = 'str'
    """

_training_scaffolding_toml = \
    """
    [training]
    
        maximum_epochs = 'int'
        loss_function = 'str'
        validations_per_epoch = 'int'
        validation_on_test_subset = 'bool'
        epochs_sm_based_warmup = ['int', 'bool']
        early_stop_epochs = ['int', 'bool']
    
        [training.optimizer]
            type = 'str'
            initial_lr = ['float', 'int']
            arguments = 'dict'
        
        [training.scheduler]
            type = ['str', 'bool']
            arguments = 'dict'
        
        [training.adversarial]
            type = ['str', 'bool']
            arguments = 'dict'
            proportion = ['float', 'int']
"""

_attack_scaffolding_toml = \
    """
    [attack]
    
        type = 'str'
        loss = 'str'
        generate_using_sm = 'bool'
        validation_on_test_subset = 'bool'
        
        [attack.parameters]
            [attack.parameters.fgsm]
                epsilon = ['float', 'int']
            [attack.parameters.pgd]
                epsilon = ['float', 'int']
                alpha = ['float', 'int']
                num_iter = 'int'
            [attack.parameters.pixle]
                x_dimensions = 'int'
                y_dimensions = 'int'
                pixel_mapping = 'str'
                restarts = 'int'
                max_iter = 'int'
                update_each_iteration  = 'bool'
            [attack.parameters.onepixel]
                pixel_count = 'int'
                max_iter = 'int'
                popsize = 'int'
    """

_params_to_filter_scaffolding_toml = \
    """
    [params_to_filter]
        trainable_section = 'list'
    """

_metrics_to_filter_scaffolding_toml = \
    """
    [params_to_filter]
        trainable_section = 'list'
    """

_retraining_scaffolding_toml = \
    """
    [retraining]
        trainable_section = 'str'
    """

########################################################################################################################
########################################################################################################################
# FUNCTION RETURNING A DICTIONARY WITH THE TYPED SCAFFOLDING OF EACH BLOCK
########################################################################################################################
########################################################################################################################

def get_dict_of_typed_scaffoldings_for_conf_files(experiment_type : str):
    """
    Function returning a dictionary with the typed scaffolding of each important block of the configuration files.

    The purpose of this function or, more precisely, of the dictionary it returns, is to have a reference \
    for the expected structure of the configuration files, so that the user can check if their configuration \
    files are correctly structured and if the types of the values are correct.

    Parameters
    ----------
    experiment_type : str
        The type of experiment to be performed. It can be either 'training', 'aa_evaluation' or 'retraining'

    Returns
    -------
    scaffolding_group_file_header : dict
        The scaffolding for the group file header block, which is compulsory (for the group files) \
        for all types of experiments
    dict_of_complusory_block_scaffoldings : dict
        It contains pairs key-value with the name of the block and the typed scaffolding of the block, \
        for the blocks that are compulsory for the indicated type of experiment
    dict_of_optional_block_scaffoldings : dict
        It contains pairs key-value with the name of the block and the typed scaffolding of the block, \
        for the blocks that are optional for the indicated type of experiment
    """

    # First, we obtain the typed-scaffolding dict of all the "possible" blocks
    dict_of_scaffoldings = {
        "group_file_header": scaffolding_toml_to_dict(_group_file_header_scaffolding_toml),
        "classifier": scaffolding_toml_to_dict(_classifier_scaffolding_toml),
        "dataset": scaffolding_toml_to_dict(_dataset_scaffolding_toml),
        "training": scaffolding_toml_to_dict(_training_scaffolding_toml),
        "experiment_name_to_retrieve": scaffolding_toml_to_dict(_experiment_name_to_retrieve_scaffolding_toml),
        "params_to_filter": scaffolding_toml_to_dict(_params_to_filter_scaffolding_toml),
        "metrics_to_filter": scaffolding_toml_to_dict(_metrics_to_filter_scaffolding_toml),
        "attack": scaffolding_toml_to_dict(_attack_scaffolding_toml),
        "retraining": scaffolding_toml_to_dict(_retraining_scaffolding_toml),
    }

    ###
    # And now we form the required dictionaries for each type of experiment
    ###

    list_of_compulsory_blocks = []
    list_of_optional_blocks = []

    if experiment_type == "training":
        list_of_compulsory_blocks = ['dataset', 'classifier', 'training']
        list_of_optional_blocks = []
    elif experiment_type == "aa_evaluation":
        list_of_compulsory_blocks = ['experiment_name_to_retrieve']
        list_of_optional_blocks = ['params_to_filter', 'metrics_to_filter', 'attack', 'classifier']
    elif experiment_type == "retraining":
        list_of_compulsory_blocks = ['experiment_name_to_retrieve']
        list_of_optional_blocks = ['params_to_filter', 'metrics_to_filter', 'retraining', 'classifier', 'training']
    else:
        raise ValueError("experiment_type must be either 'training', 'aa_evaluation' or 'retraining'.")
    pass

    scaffolding_group_file_header = dict_of_scaffoldings["group_file_header"]
    #
    dict_of_complusory_block_scaffoldings = {}
    for block in list_of_compulsory_blocks:
        dict_of_complusory_block_scaffoldings[block] = dict_of_scaffoldings[block]
    pass
    #
    dict_of_optional_block_scaffoldings = {}
    for block in list_of_optional_blocks:
        dict_of_optional_block_scaffoldings[block] = dict_of_scaffoldings[block]
    #
    return scaffolding_group_file_header, dict_of_complusory_block_scaffoldings, dict_of_optional_block_scaffoldings