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
from abc import ABC, abstractmethod
from collections import OrderedDict
import traceback

import typing  # To check if hashable
from typing import Sequence

import copy
import inspect

import math
import numpy as np

import torch
from torch import nn

# from torchvision.models.alexnet import AlexNet
from torchvision.models.efficientnet import _MBConvConfig, MBConvConfig, FusedMBConvConfig, MBConv, FusedMBConv, \
    _efficientnet_conf

from modified_rf import _dict_conv_like_layers, processed_constructor_kwargs_for_conv_like_layer
from modified_rf import SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer
from modified_rf import kernel_size_check_and_reformat_into_tuple, resolve_kernel_size_for_im_size

from .utils import _create_standard_block_conv_i, _create_standard_FC_blocks, \
    _create_standard_head_module, _create_standard_backbone_module


#########################################################################################
#########################################################################################
#########################################################################################
# CLASSIFIER CLASSES
#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
# Abstract Layer ModifiedRFLayer
#########################################################################################
#########################################################################################


class ClassifierBaseModel(ABC, nn.Module):
    """
    This class aims at unifying all the functions common to all the custom classifiers that we will implement \
    for testing the INRF and ibnn_internal layers, so tests on the classifiers can can be performed totally uniformly. \
    To that end the "semi-abstract" class defines a series of implemented functions relying on a number of \
    standardized attributions for all inheriting classes, so said functions will act as \
    "abstract methods" controlling totally the interplay between inheriting classifier subclasses and the exisiting \
    classes/functions for training and logging classifiers.

    The correct operation of the methods defined in this :py:class:`.ClassifierBaseModel` relies on the correct \
    definition of the following compulsory attributes in the inheriting classes: ``self._extra_state_dict``, \
    ``self._fields_to_log``, and ``self._nn`` (the latter is the functional neural network of the class, \
     whose forward method is internally called \
     by the method :py:meth:`.ClassifierBaseModel.forward` when such operation is performed by an object \
    of the inheriting class). The requirements for the compulsory attributes listed above are described in detail in \
    the documentation of the method :py:meth:`.ClassifierBaseModel.logging_compliance_checker`: check it for more details.

    **The requirements of the inheriting classes regarding the above aspects and abstract methods is checked by the \
    method** :py:meth:`.ClassifierBaseModel.logging_compliance_checker`: **check its documentation for more details on said \
    requirements.**

    **The recommended use** of the  inheritance of :py:class:`.ClassifierBaseModel` involves, \
    **immediately after constructing the object of the inheriting class** (and in any case before its training), \
    **calling the method** :py:meth:`.ClassifierBaseModel.logging_compliance_checker` to \
    assess that the logging of the network during and after training would be correct:

    - Class definition::

        class SpecificClassifier(ClassifierBaseModel):

            def __init__(self, ...): # Constructor of the inheriting class
                super().__init__()   # Since ClassifierBaseModel inherits from nn.Module, this call is recommended here
                ...                  # Specific code of the constructor

    - Object creation and compliance check::

        child_class_obj = SpecificClassifier(...)       # Create an object of the class
        child_class_obj.logging_compliance_checker()    # Check compliance with the requirements of the parent class

    *NOTE:* The class is abstract and cannot be instantiated directly.
    """

    @abstractmethod
    def __init__(self):
        #
        nn.Module.__init__(self)
        #
        self._device = None
        #
        self._last_warning_lack_of_convergence = None
        self._last_layer_name_lack_of_convergence = None
        self._last_abs_error_threshold_lack_of_convergence = None

    pass

    def to_device(self, device: str):
        """
        This method moves the network to the device indicated in the argument, and changes the state of the attribute
        `self.device` to the device indicated in the argument.

        Parameters
        ----------
        device : str
            Device to which the network will be moved. It can be 'cpu' or 'cuda', or any other device name.
        """

        assert isinstance(device, str), \
            f"Invalid 'device': string expected, {device} found!."

        self._device = device
        self._nn.to(device)

    @property
    def device(self):
        """
        This method returns the device to which the network is currently moved.

        Returns
        -------
        str
            Device to which the network is currently moved.
        """
        return self._device

    def logging_compliance_checker(self):
        """
        This method checks the compliance of the child class/object with the requirements of the parent/abstract class \
        :py:class:`.ClassifierBaseModel`. These requirements are defined by a number of private attributes for \
        the class and their contained fields.

        The requirements are limited to the attributes ``self._fields_to_log``, ``self._extra_state_dict``,  and \
        ``self._nn`` (see also :py:class:`.ClassifierBaseModel` for more details). The method checks the correct \

        Regarding the requirements of each one:

        - **self._extra_state_dict** (:py:class:`dict`): it must exist and must contain, at least, these two fields:

          - ``self._extra_state_dict['net']`` (:py:class:`str`): the type of network used,

          - ``self._extra_state_dict['softmax_output']`` (:py:class:`bool`): whether batch normalization \
            is used in the network,

          - ``self._extra_state_dict['constructor_kwargs']`` (:py:class:`dict`): the kwargs that would \
            directly construct an object identical (but for the trainable parameters) to the current one.

        - **self._fields_to_log** (:py:class:`dict`): it must exist and must contain the fields to be logged \
          when training the network. Its minimum required fields are:

          - ``self._fields_to_log['net']`` (:py:class:`str`): the type of network used,

          - ``self._fields_to_log['in_size']`` (:py:class:`tuple` of 2 :py:class:`int` x>0), \
            ``self._fields_to_log['in_channels']`` (:py:class:`int`)

          - ``self._fields_to_log['out_classes']`` (:py:class:`int`),

            ``self._fields_to_log['softmax_output']`` (:py:class:`bool`),

          - ``self._fields_to_log['conv_like_type']`` (:py:class:`str`),

          - ``self._fields_to_log['m_padding']`` (:py:class:`str`), \
            ``self._fields_to_log['m_padding_mode']`` (:py:class:`str`), \
            ``self._fields_to_log['m_independent_channels']`` (:py:class:`bool`),
            ``self._fields_to_log['m_initialization']`` (:py:class:`str`), \
            ``self._fields_to_log['m_trainable']`` (:py:class:`bool`), 'b_type' (:py:class:`str`),

          - ``self._fields_to_log['initial_b']`` (:py:class:`float` or :py:class:`tuple` [:py:class:`float`]), \
            ``self._fields_to_log['b_trainable']`` (:py:class:`bool`)

          (in the case of *'conv_like_type'* being ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ```'ibnn'``, \
          additionally):

          - ``self._fields_to_log['sigma_activation']`` (:py:class:`str`), \

          - ``self._fields_to_log['sigma_x_compress']`` (:py:class:`float`), \
            ``self._fields_to_log['sigma_x_compress_trainable']`` (:py:class:`bool`),

          - ``self._fields_to_log['sigma_y_stretch']`` (:py:class:`float`), \
            ``self._fields_to_log['sigma_y_stretch_trainable']`` (:py:class:`bool`),

          - ``self._fields_to_log['sigma_x_offset']`` (:py:class:`float`), \
            ``self._fields_to_log['sigma_x_offset_trainable']`` (:py:class:`bool`),

          - ``self._fields_to_log['sigma_y_offset']`` (:py:class:`float`), \
            ``self._fields_to_log['sigma_y_offset_trainable']`` (:py:class:`bool`)

          - ``self._fields_to_log['lambda_type']`` (:py:class:`str`), \
            ``self._fields_to_log['initial_lambda']`` (:py:class:`float` or :py:class:`tuple` or :py:class:`list` [:py:class:`float`]), \
            ``self._fields_to_log['lambda_trainable']`` (:py:class:`bool`),

          (in the case of *'conv_like_type'* being ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ```'ibnn'``, additionally):

          - ``self._fields_to_log['w_kernel_size']`` (2D :py:class:`tuple` [:py:class:`int`] with x>0 or \
            2D :py:class:`tuple` [:py:class:`float`] with 0<x<1),

          - ``self._fields_to_log['w_padding_mode']`` (:py:class:`str`), \
            ``self._fields_to_log['w_independent_channels']`` (:py:class:`bool`), \
            ``self._fields_to_log['w_initialization']`` (:py:class:`str`), \
            ``self._fields_to_log['w_trainable']`` (:py:class:`bool`)

          (in the case of *'conv_like_type'* being ``'ibnn_internal'``, ```'ibnn'``):

          - ``self._fields_to_log['batched_fixed_point']`` (:py:class:`bool`), \
            ``self._fields_to_log['f_solver']`` (:py:class:`str`), \
            ``self._fields_to_log['b_solver']`` (:py:class:`str`).

        - **self._nn** (:py:class:`torch.nn.Module` or subclasses of it such as :py:class:`torch.nn.Sequential` or \
          :py:class:`torch.nn.ModuleList`): the functional neural network whose forward method is internally called \
          by the method :py:meth:`.ClassifierBaseModel.forward` when such operation is performed by an object \
          of the inheriting class.

        The method will raise an exception if the attributes are not present in the inheriting class.

        Returns
        -------
        bool

        """

        ###################################################
        # Checks on the property method 'self.fields_to_log'
        ###################################################

        self._check_property_fields_to_log()

        ###################################################
        # Checks on the method 'self.get_extra_state()'
        ###################################################

        self._check_method_get_extra_state()

        # If no exception has been raised, the class is compliant
        return True

    def _check_property_fields_to_log(self):
        """
        This method checks the correct definition of the method `fields_to_log` in the class by checking the \
        correct definition of 'self._fields_to_log' and the existence of all its compulsory fields.
        """

        ### Is '_fields_to_log' an attribute of the implemented child class?

        ### Checks existence of '_fields_to_log'
        assert hasattr(self, '_fields_to_log'), \
            f"Attribute '_fields_to_log' not found in the class {type(self).__name__}."

        ### Checks that '_fields_to_log' is a dictionary
        assert isinstance(self._fields_to_log, dict), \
            f"Attribute '_fields_to_log' in the class {type(self).__name__} must be a dictionary, " + \
            f"{type(self._fields_to_log)} found."

        ### Checks on its fields
        ### Function checking the presence and type of the indicated keys in the dictionary 'self._fields_to_log'
        def _check_the_provided_key_and_type_dictionary(key_type_dict):
            for key in key_type_dict:
                assert key in self._fields_to_log, \
                    f"Compulsory key '{key}' not found in the dictionary " + \
                    f"{type(self).__name__}._fields_to_log."
                assert isinstance(self._fields_to_log[key], key_type_dict[key]), \
                    f"Compulsory Key '{key}' in the dictionary {type(self).__name__}._fields_to_log " + \
                    f"must be of type {key_type_dict[key]}, {type(self._fields_to_log[key])} found."
            pass

        pass

        ###################################################
        # Second, check the presence of certain compulsory fields in the dictionary 'self._fields_to_log' and their types
        ###################################################

        ### Fields compulsory for networks composed of all conv-like layers
        dict_keys_and_types_all_conv_like_types = {
            'net': str,
            'in_size': (tuple, list, float, int),  # It will be checked separately
            'in_channels': int,
            'out_classes': int, 'fc_num_layers': int,
            'softmax_output': bool,
            'conv_like_type': str,  # extra checks for 'conv_like_type' later
            'm_kernel_size_per_conv_layer': (tuple, float, int),  # It will be checked separately
            # 'm_padding': str, 'm_padding_mode': str, 'm_independent_channels': bool, # "m_padding" NOT IN ALL LAYERS!!
            'm_initialization': str, 'm_trainable': bool,
            'b_type': str, 'initial_b': (int, float, tuple), 'b_trainable': bool
        }
        #
        _check_the_provided_key_and_type_dictionary(dict_keys_and_types_all_conv_like_types)
        #
        # Extra checks for 'conv_like_type'
        assert self._fields_to_log['conv_like_type'] in _dict_conv_like_layers, \
            f"Invalid 'conv_like_type' in 'self._fields_to_log': {self._fields_to_log['conv_like_type']} found, " + \
            f"expected one of {list(_dict_conv_like_layers.keys())}."
        #
        # 'in_size' will be checked separately
        self._fields_to_log['in_size'] = kernel_size_check_and_reformat_into_tuple(self._fields_to_log['in_size'])
        #
        # 'm_kernel_size_per_conv_layer' will be checked separately: it must be a tuple of length 2, or a list of tuples of length 2!
        assert isinstance(self._fields_to_log['m_kernel_size_per_conv_layer'], tuple), \
            f"Invalid 'm_kernel_size_per_conv_layer' in 'self._fields_to_log': {self._fields_to_log['m_kernel_size_per_conv_layer']} found, " + \
            f"expected a tuple."
        if all(isinstance(x, tuple) and len(x) == 2 for x in self._fields_to_log['m_kernel_size_per_conv_layer']):
            # It is a list of tuples of length 2
            self._fields_to_log['m_kernel_size_per_conv_layer'] = tuple(
                kernel_size_check_and_reformat_into_tuple(x) for x in self._fields_to_log['m_kernel_size_per_conv_layer']
            )
        else:  # If it is only a tuple
            self._fields_to_log['m_kernel_size_per_conv_layer'] = \
                kernel_size_check_and_reformat_into_tuple(self._fields_to_log['m_kernel_size_per_conv_layer'])
        pass

        if self._fields_to_log['conv_like_type'] in ['inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            #
            dict_keys_and_types_all_conv_like_types = {
                'sigma_activation': str,
                'sigma_x_compress': (int, float), 'sigma_y_stretch': (int, float),
                'sigma_x_offset': (int, float), 'sigma_y_offset': (int, float),
                'sigma_x_compress_trainable': bool, 'sigma_y_stretch_trainable': bool,
                'sigma_x_offset_trainable': bool, 'sigma_y_offset_trainable': bool,
                'lambda_type': str, 'initial_lambda': (int, float, tuple, list, str), 'lambda_trainable': bool
            }
            _check_the_provided_key_and_type_dictionary(dict_keys_and_types_all_conv_like_types)
            #
            if self._fields_to_log['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
                #
                dict_keys_and_types_all_conv_like_types = {
                    'w_kernel_size': (tuple, list, float, int),  # it will be checked separately
                    'w_padding_mode': str, 'w_independent_channels': bool,
                    'w_initialization': (str, dict, int, float, torch.Tensor), 'w_trainable': bool,
                }
                _check_the_provided_key_and_type_dictionary(dict_keys_and_types_all_conv_like_types)
                #
                # 'w_kernel_size' will be checked separately with a specific auxiliary function
                self._fields_to_log['w_kernel_size'] = kernel_size_check_and_reformat_into_tuple(
                    self._fields_to_log['w_kernel_size']
                )
                #
                if self._fields_to_log['conv_like_type'] in ['ibnn_internal', 'ibnn']:
                    #
                    dict_keys_and_types_all_conv_like_types = {
                        'batched_fixed_point': bool, 'f_solver': str, 'b_solver': str
                    }
                    _check_the_provided_key_and_type_dictionary(dict_keys_and_types_all_conv_like_types)
                    #
                pass
            pass

    def _check_method_get_extra_state(self):
        """
        This method checks the correct definition of the method `get_extra_state` in the class and the \
        existence of all its compulsory fields.
        """

        ### Checks existence of '_extra_state_dict'
        assert hasattr(self, '_extra_state_dict'), \
            f"Attribute '_extra_state_dict' not found in the class {type(self).__name__}."

        ### Checks that '_extra_state_dict' is a dictionary
        assert isinstance(self._extra_state_dict, dict), \
            f"Attribute '_extra_state_dict' in the class {type(self).__name__} must be a dictionary, " + \
            f"{type(self._extra_state_dict)} found."

        # In detail: check explicitly that the dictionary 'self._extra_state_dict' contains the required fields \
        # 'net' and 'constructor_kwargs' and that the latter fulfills its requirements.

        ### FIELD 'net'

        assert 'net' in self._extra_state_dict, \
            f"Compulsory field 'net' not found in the dictionary {type(self).__name__}._extra_state_dict."

        assert isinstance(self._extra_state_dict['net'], str), \
            f"Field 'net' in the dictionary {type(self).__name__}._extra_state_dict must be a string, " + \
            f"{type(self._extra_state_dict['net'])} found."

        ### FIELD 'constructor_kwargs'

        assert 'constructor_kwargs' in self._extra_state_dict, \
            f"Compulsory field 'constructor_kwargs' not found in the dictionary " + \
            f"{type(self).__name__}._extra_state_dict."

        assert isinstance(self._extra_state_dict['constructor_kwargs'], dict), \
            f"Field 'constructor_kwargs' in the dictionary {type(self).__name__}._extra_state_dict " + \
            f"must be a dictionary, {type(self._extra_state_dict['constructor_kwargs'])} found."

        # Check that the field 'constructor_kwargs' in the dictionary 'self._extra_state_dict' \
        # is valid
        self._check_property_constructor_kwargs()

    def _check_property_constructor_kwargs(self):
        """
        This method checks the correct definition of the method `constructor_kwargs` in the class by checking the \
        correct definition of 'self._extra_state_dict['constructor_kwargs'] \
        and the existence of all its compulsory fields.
        """

        ### Checks existence of '_extra_state_dict'
        assert hasattr(self, '_extra_state_dict'), \
            f"Attribute '_extra_state_dict' not found in the class {type(self).__name__}."

        ### Checks existence of 'self._extra_state_dict['constructor_kwargs']'
        assert 'constructor_kwargs' in self._extra_state_dict, \
            f"Field 'constructor_kwargs' not found in the attribute 'self._extra_state_dict' of the " + \
            f"class {type(self).__name__}."

        ### Checks that 'constructor_kwargs' is a dictionary
        assert isinstance(self._extra_state_dict['constructor_kwargs'], dict), \
            f"Attribute '_constructor_kwargs' in the class {type(self).__name__} must be a dictionary, " + \
            f"{type(self._extra_state_dict['constructor_kwargs'])} found."

        # In detail: check explicitly that the dictionary 'constructor_kwargs' in fact can generate a new object \
        # with the same parameters as the current one, except for the trainable parameters.

        probe_object = type(self)(**self._extra_state_dict['constructor_kwargs'])
        assert isinstance(probe_object, type(self)), \
            f"Object created by the dictionary 'constructor_kwargs' is not of the same class as the current object."

        try:
            probe_object.load_state_dict(self.state_dict(), strict=True)
        except Exception as err:
            raise Exception(
                f"Created object of class {type(self).__name__} appears to incorrectly define the returned dict of " +
                f"method 'get_extra_state()', \nand in particular the field 'constructor_kwargs' therein, and thus " +
                f"it does not comply with the parent class 'ClassifierBaseModel'. \n" +
                f"Reason: the probe object created in the method 'logging_compliance_checker()' with said dictionary " +
                f"caused the following exception \nwhen the 'state_dict' of the original object was loaded: " +
                f"\n{err}"
            )
        pass

    @property
    def fields_to_log(self):
        """
        This property method returns a dictionary with the fields that must be logged when training the network.
        """
        self._check_property_fields_to_log()
        return copy.deepcopy(self._fields_to_log)

    @property
    def constructor_kwargs(self):
        """
        This property method returns a dictionary with the arguments required to construct an object identical \
        to the current one, except for the trainable parameters.
        """
        self._check_property_constructor_kwargs()
        return copy.deepcopy(self._extra_state_dict['constructor_kwargs'])

    def get_extra_state(self):
        """
        This method returns a dictionary with the extra state of the network, which is the state that is not \
        contained in the state dictionary of the network itself. The dictionary must contain the following fields: \
        'net' and 'constructor_kwargs'.
        """
        # self._check_method_get_extra_state()
        return copy.deepcopy(self._extra_state_dict)

    # NOT ABSTRACT ANYMORE: @abstractmethod
    def set_extra_state(self, extra_state_dict):
        """
                Although the methods *set_extra_state* usual load stored parameters, we will mostly use it to make sure that \
                the parameters stored in ``extra_state_dict``, and corresponding to a previous network, are compatible with \
                the parameters of this current network.
                We will allow, though, if required, the change of type of the convolutional layers of the network, ruled by \
                both the 'self._conv_like_type' and 'self._conv_like_type_position' parameters, simply giving a warning.
                """

        ##############################################################################
        # REMEMBER: the 'self._extra_state_dict' is a dict that contains:
        #   - ``self._extra_state_dict['net']`` (:py:class:`str`): the type of network used
        #   - ``self._extra_state_dict['batch_normalization']`` (:py:class:`bool`): whether batch normalization is used
        #   - ``self._extra_state_dict['softmax_output']`` (:py:class:`bool`): whether batch normalization is used
        #   - ``self._extra_state_dict['constructor_kwargs']`` (:py:class:`dict`):
        #       the kwargs that would directly construct an object identical (but for the trainable parameters)
        #       to the current one.
        ##############################################################################

        # Check the keys 'net', 'batch_normalization', and 'softmax_output'
        for key in extra_state_dict.keys():
            if key in ['net', 'batch_normalization', 'softmax_output']:
                if self._extra_state_dict[key] != extra_state_dict[key]:
                    raise Exception(
                        f"Provided extra_state_dict['{key}'] does not match current self._extra_state_dict['{key}']: " +
                        f"{extra_state_dict[key]} != {self._extra_state_dict[key]}; match of these fields is required."
                    )
                pass
            pass
        pass

        ############## !!!
        # General principle: if any field of the 'extra_state_dict' provided as argument is `None` it's not transferred!
        ############## !!!

        # Check 'constructor_kwargs': does it exist, does it fit?
        if 'constructor_kwargs' not in extra_state_dict.keys():
            raise Exception(
                f"Provided extra_state_dict['constructor_kwargs'] does not exist, no checks can be performed!"
            )
        else:
            for key in extra_state_dict['constructor_kwargs'].keys():
                ######## !!!
                if extra_state_dict['constructor_kwargs'][key] is None:
                    # Do not transfer None values
                    continue
                ######## !!!
                if key not in self._extra_state_dict['constructor_kwargs'].keys():
                    print(f"Warning: key '{key}' of the argument 'extra_state_dict['constructor_kwargs']'" +
                          f"not present in the current network.")
                elif key in ['conv_like_type', 'conv_like_type_position']:
                    # Warning only
                    if self._extra_state_dict['constructor_kwargs'][key] != extra_state_dict['constructor_kwargs'][key]:
                        print((
                                f"Warning: the value '{key}' of the current (target) network of the load/transfer process " +
                                f"is '{self._extra_state_dict['constructor_kwargs'][key]}', whereas the past (source) network " +
                                f"had, however, '{extra_state_dict['constructor_kwargs'][key]}'; " +
                                f"the latter value will not be transferred into the receiving network."
                        ))
                    pass
                elif key in ['m_padding_mode', 'w_padding_mode']:
                    # Warning and transfer
                    if self._extra_state_dict['constructor_kwargs'][key] != extra_state_dict['constructor_kwargs'][key]:
                        print((
                                f"Warning: the value '{key}' of the current (target) network of the load/transfer process " +
                                f"is '{self._extra_state_dict['constructor_kwargs'][key]}', whereas the past (source) network " +
                                f"had, however, '{extra_state_dict['constructor_kwargs'][key]}'; " +
                                f"the latter value will not be transferred into the receiving network."
                        ))
                        self._extra_state_dict['constructor_kwargs'][key] = \
                            copy.deepcopy(extra_state_dict['constructor_kwargs'][key])
                    pass
                else:
                    if self._extra_state_dict['constructor_kwargs'][key] != extra_state_dict['constructor_kwargs'][
                        key]:
                        print((f"Warning: Provided extra_state_dict['constructor_kwargs']['{key}'] " +
                               f"does not match current self._extra_state_dict['constructor_kwargs']['{key}']: " +
                               f"{extra_state_dict['constructor_kwargs'][key]} != {self._extra_state_dict['constructor_kwargs'][key]}."))
                        self._extra_state_dict['constructor_kwargs'][key] = \
                            copy.deepcopy(extra_state_dict['constructor_kwargs'][key])
                pass
            pass
        pass

    def load_state_dict_but_exceptions(self, state_dict, list_exceptions, strict=True, assign=False):
        """
        This method replicates the method :py:meth:`torch.nn.Module.load_state_dict`, but allowing to exclude \
        the load of those parameters and attributes whose names contain any of the strings in \
        `list_exceptions`, which is a string or a list/tuple of strings.
        It returns the same output as the method :py:meth:`torch.nn.Module.load_state_dict`.

        Parameters
        ----------
        state_dict: dict
        list_exceptions: str or list[str] or tuple[str]
            If a string, it indicates the substring that, if contained in the name of a parameter or attribute, \
            will cause that parameter/attribute to be excluded from the load operation.
            If a list/tuple of strings, any parameter/attribute whose name contains any of the strings in the list/tuple \
            will be excluded from the load operation.
        strict: bool, optional
            See :py:meth:`torch.nn.Module.load_state_dict`
            Default: ``True``
        assign: bool, optional
            See :py:meth:`torch.nn.Module.load_state_dict`
            Default: ``False``

        Returns
        -------
        missing_keys: list
            See :py:meth:`torch.nn.Module.load_state_dict`
        unexpected_keys: list
            See :py:meth:`torch.nn.Module.load_state_dict`
        """

        # Check the type of 'list_exceptions'
        if list_exceptions is None:
            list_exceptions = []
        elif isinstance(list_exceptions, str):
            list_exceptions = [list_exceptions]
        elif isinstance(list_exceptions, (list, tuple)):
            list_exceptions = list(list_exceptions)
            for item in list_exceptions:
                assert isinstance(item, str), \
                    f"Invalid 'list_exceptions': string or list/tuple of strings expected, " + \
                    f"{type(list_exceptions)} found."
        else:
            raise TypeError(f"Invalid 'list_exceptions': string or list/tuple of strings expected, " +
                            f"{type(list_exceptions)} found.")
        pass

        # Get the state_dict of the current network
        current_state_dict = copy.deepcopy(self.state_dict())

        # Create a filtered state_dict wherein the parameters/attributes whose names contain any of the \
        # strings in 'list_exceptions' are substituted by the value in the current state dict of the current network
         # (i.e., they are not loaded from the provided 'state_dict')

        def recursive_state_dict_filtering(current_state_dict_i, state_dict_i):
            filtered_input_state_dict_i = {}
            list_of_removed_keys_i = []
            list_of_kept_keys_i = []
            if isinstance(state_dict_i, dict):
                if isinstance(current_state_dict_i, dict):
                    for k in state_dict_i.keys():
                        if any(exc in k for exc in list_exceptions):
                            if k in current_state_dict_i:
                                filtered_input_state_dict_i[k] = copy.deepcopy(current_state_dict_i[k])
                                list_of_kept_keys_i.append(k)
                            else:
                                list_of_removed_keys_i.append(k)
                            pass
                        else:
                            filtered_input_state_dict_i[k], child_list_of_removed_keys, child_list_of_kept_keys = \
                                recursive_state_dict_filtering(
                                    current_state_dict_i.get(k, None), state_dict_i[k]
                                )
                            list_of_kept_keys_i.extend([k + ">" + elem for elem in child_list_of_kept_keys])
                            list_of_removed_keys_i.extend([k + ">" + elem for elem in child_list_of_removed_keys])
                        pass
                    pass
                else:
                    for k in state_dict_i.keys():
                        if any(exc in k for exc in list_exceptions):
                            list_of_removed_keys_i.append(k)
                        else:
                            filtered_input_state_dict_i[k], child_list_of_removed_keys, child_list_of_kept_keys = \
                                recursive_state_dict_filtering(None, state_dict_i[k])
                            list_of_kept_keys_i.extend([k + ">" + elem for elem in child_list_of_kept_keys])
                            list_of_removed_keys_i.extend([k + ">" + elem for elem in child_list_of_removed_keys])
                        pass
                    pass
                pass
            else:
                filtered_input_state_dict_i = copy.deepcopy(state_dict_i)
            pass
            #
            return filtered_input_state_dict_i, list_of_removed_keys_i, list_of_kept_keys_i

        # filtered_input_state_dict = {}
        # list_of_removed_keys = []   # These keys are directly removed from the incoming state_dict
        # list_of_kept_keys = []  # These keys are kept in the current (receiver) network
        #
        # for key in state_dict.keys():
        #     if any(exc in key for exc in list_exceptions):
        #         if key in current_state_dict:
        #             filtered_input_state_dict[key] = current_state_dict[key]
        #             list_of_kept_keys.append(key)
        #         else:
        #             list_of_removed_keys.append(key)
        #         pass
        #     else:
        #         filtered_input_state_dict[key] = state_dict[key]
        #     pass
        # pass

        filtered_input_state_dict, list_of_removed_keys, list_of_kept_keys = \
            recursive_state_dict_filtering(current_state_dict, state_dict)

        # Give always a warning with the list of kept and removed keys: if none was removed, say so
        print(f"Note: the following exceptions were considered for the load operation: " +
              f"{list_exceptions if len(list_exceptions) > 0 else 'NONE'}.")
        print(f"Warning: the following keys of the 'sender' network state_dict were excluded from the load " +
              f"and are not present either in the 'receptor' network: " +
              f"{list_of_removed_keys if len(list_of_removed_keys) > 0 else 'NONE'}.")
        print(f"Warning: the following keys of the 'sender' network state_dict were excluded from the load; " +
              f"since present in the 'receptor' network their values were kept as they stood: " +
              f"{list_of_kept_keys if len(list_of_kept_keys) > 0 else 'NONE'}.")

        # Use the loading method of the parent class nn.Module and return its result
        return super().load_state_dict(filtered_input_state_dict, strict=strict, assign=assign)

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        Parameters
        ----------
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        #######
        # Structure of the network
        #######
        # self._nn
        #     prenormalization
        #     backbone
        #         ...
        #     head
        #         ...
        #######

        for submodule in self._nn.modules():
            # The conv-like layer(s) ...
            if isinstance(submodule,
                          (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer)):
                submodule.random_initialization(distribution=distribution, gain=gain, additive=additive)
            # and the batch-normalization and the fully-connected layer
            elif isinstance(submodule, (nn.BatchNorm2d, nn.Linear)):
                for param_tensor in submodule.parameters():
                    nn.init.uniform_(param_tensor,
                                     a=param_tensor.flatten().mean().item() - gain / 2,
                                     b=param_tensor.flatten().mean().item() + gain / 2)
                pass
            pass
        pass

    def prenormalization_initialization_to_dataset_statistics(self, dataset_dict):
        """
        Initialization of the input statistics (`running_mean` and `running_var`) of the `prenormalization` layer \
        (a :py:class:`~torch.nn.BatchNorm2d` layer) to the statistics of the provided :py:class:`.LoadedDatasetDict` \
        `dataset_dict`.

        Parameters
        ----------
        dataset_dict : :py:class:`.LoadedDatasetDict`
        """

        #######
        # Structure of the network
        #######
        # self._nn
        #     prenormalization
        #     backbone
        #         ...
        #     head
        #         ...
        #######

        flag_achieved = False
        if ('prenormalization' in self._extra_state_dict) and (self._extra_state_dict['prenormalization'] == True) and \
                hasattr(self._nn, 'prenormalization') and \
                isinstance(self._nn.prenormalization, torch.nn.BatchNorm2d) and \
                'final_mean_train' in dataset_dict and 'final_std_train' in dataset_dict:
            self._nn.prenormalization.running_mean[:] = dataset_dict['final_mean_train'][:]
            self._nn.prenormalization.running_var[:] = torch.pow(dataset_dict['final_std_train'], 2)[:]
            flag_achieved = True
        pass
        #
        return flag_achieved

    def forward(self, x):
        """
        Forward pass of the network which, since the network is a Sequential object, is simply the forward pass \
        of the Sequential object.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        # Check if the input is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, {type(x)} found.")

        # Make the input be on the same device as the model (if possible)
        if self.device is not None:
            if x.device.type != self.device:
                x = x.to(self.device)
            pass
        pass

        # Calculate!
        result_forward = None
        #
        batch_normalization_present = False
        batch_normalization_per_conv_layer = self._extra_state_dict.get('batch_normalization_per_conv_layer', None)
        fc_batch_normalization = self._extra_state_dict.get('fc_batch_normalization', None)
        batch_normalization_present = True if batch_normalization_per_conv_layer == True else batch_normalization_present
        batch_normalization_present = True if fc_batch_normalization == True else batch_normalization_present
        #
        if self.training and batch_normalization_present and x.ndim > 3 and x.size(-4) == 1:
            # print((f"\t\tWARNING: Input batch has only 1 im, so BatchNorm1d would cause an error in training: " +
            #        f"EVAL used instead!!!"))
            self._nn.eval()
            result_forward = self._nn(x)
            self._nn.train()
        else:
            result_forward = self._nn(x)
        pass

        # Assess, if there is any layer in the complete network which is an ibnn_internal or ibnn layer, whether the \
        # evaluation yielded some "lack-of-convergence" warning; this is done, per ibnn_internal/ibnn layer, by checking \
        # their method 'IBNNInternalLayer.get_last_forward_convergence_info()" after a forward pass.

        B = x.size(0) if x.ndim > 3 else 1
        # Warnings: initially to false
        self._last_warning_lack_of_convergence = torch.zeros((B,), dtype=torch.bool, device=self.device)
        # Name of layers: initially to None
        self._last_layer_name_lack_of_convergence = [None]*B
        # Absolute error threshold: initially to NaN
        self._last_abs_error_threshold_lack_of_convergence = torch.empty((B,), dtype=torch.float, device=self.device)
        self._last_abs_error_threshold_lack_of_convergence[:] = np.nan

        for name_layer_i, ibnn_like_layer_i in self._nn.named_modules():
            if isinstance(ibnn_like_layer_i, (IBNNInternalLayer, IBNNLayer)):
                (warning_above_threshold_y_out_layer_i, last_forward_abs_error_threshold_layer_i,
                 info_layer_i, last_forward_batched_fixed_point_layer_i) = \
                    ibnn_like_layer_i.get_last_forward_convergence_info()
                # Get the images where this layer i has suffered an error and write such info in the corrsp. structure
                flags_new_warnings = warning_above_threshold_y_out_layer_i * \
                                     torch.logical_not(self._last_warning_lack_of_convergence)
                # Warnings: new Trues, to True
                self._last_warning_lack_of_convergence[flags_new_warnings] = True
                # The obtained variable is a tensor with as many elements as the batch size, \
                # indicating whether the corresponding image has reached a fixed point with an absolute error
                self._last_abs_error_threshold_lack_of_convergence[flags_new_warnings] = \
                    last_forward_abs_error_threshold_layer_i
                for ind in torch.where(flags_new_warnings)[0].tolist():
                    self._last_layer_name_lack_of_convergence[ind] = name_layer_i
                pass
            pass
        pass

        return result_forward

    def get_last_forward_convergence_info(self):
        """
        Get the convergence information for the last :py:meth:`.forward`, aimed at summarizing the fixed point \
        convergence status of all the ibnn_internal/ibnn layers composing the classifier. \
        If no calculation has been performed by the object so far ``None`` is returned.

        The information returned by this method stems from the information provided by \
        :py:meth:`IBNNInternalLayer.get_last_forward_convergence_info` for all the ibnn_internal/ibnn layers of the network; \
        that info, e.g., a tensor with as many bools as the processed batch indicating whether the \
        corresponding image has reached a fixed point with an absolute error above the threshold \
        stablished for the network. As a result, this method returns:

        - a single flag indicating that at least one of the ibnn_internal-like layers of the net showed a warning
        - a 1D bool tensor "warning_lack_of_convergence" with as many bools as the size of the batch and indicating \
          if ANY of the ibnn_internal-like layers of the net showed a warning according to its settings.
        - a (1D) list with as many elements as images in the batch containing, at each element, the name of the layer \
          which caused the problem (and None if no problem); if multiple problems for that image in several layers \
          only the info of the earlies layer struggling with convergence is returned.
        - a 1D float tensor with the absolute error threshold used for the fixed point calculation warning that \
          gave the warning (and NaN if no problem)

        Returns
        -------
        general_warning_lack_of_convergence : bool
            Any of the images did not converge in any of the ibnn_internal-like layers of the net; further information is \
            to be obtained by the other return values of this method
        warning_lack_of_convergence : torch.Tensor
            Tensor of bools indicating whether the corresponding image has reached, for any of the layers of the net, \
            convergence problems in the calculation of the fixed point
        layer_name_lack_of_convergence : list[str]
            List of strings with as many elements as the batch size, indicating the name of the layer \
            which caused the problem (and None if no problem); if multiple problems for that image in several layers \
            only the info of the earlies layer struggling with convergence is returned
        abs_error_threshold_lack_of_convergence : torch.Tensor
            Tensor of floats with the absolute error threshold used for the fixed point calculation warning that \
            gave the warning (and NaN if no problem)
        """

        return (
            torch.any(self._last_warning_lack_of_convergence),
            copy.deepcopy(self._last_warning_lack_of_convergence),
            copy.deepcopy(self._last_layer_name_lack_of_convergence),
            copy.deepcopy(self._last_abs_error_threshold_lack_of_convergence)
        )


#########################################################################################
#########################################################################################
# CLASSES: MultilayerClassifier model
#########################################################################################
#########################################################################################


class MultilayerClassifier(ClassifierBaseModel):
    """
    Base class for multilayer neural network classifiers.

    This class provides certain common functionality for classifier models with multiple layers: this functionality \
    includes:

    - checking the validity of the input parameters which are common to all multilayer classifiers (which are the \
      explicit arguments of the constructor of this present class), that is:

        - the input (spatial) size, number of input channels, and number for output classes;
        - the type of convolutional-like layers, and their position;
        - the distribution of the convolutional-like layers in blocks, wherein each block might be ended by a \
          batch normalization layer and/or a maxpooling layer; and the number of channels and the sizes if the \
          filters of such convolutional-like layers;
        - the activation function used in the convolutional-like layers, and the mixing or not of the channels,;
        - the characteristics of the head of the network, based on a number of fully connected layers;
        - the presence of prenormalization at the beginning of the network, and the presence of \
          a softmax layer as its output layer; and
        - the device where the classifier is created.

    - checking the validity of the parameters relevant for the specific convolutional-like layers \
      selected for the classifier which are either implicitly provided (i.e. in `kwargs`), and inferring those \
      not provided at all and left to the default values of convolutional-like layer classes;

    - storing the parameters of the constructor of the (child) class in a dictionary \
      `self._extra_state_dict['constructor_kwargs']` for future reproduction of the exact network; and

    - storing the parameters to be logged during training in a dictionary \
      `self._fields_to_log` for future logging of the training process.

    The specific multilayer classifiers will be defined as inheriting from this class \
    :py:class:`.MultilayerClassifier` (whose constructor is called in their constructor); \
    the parent class assesses the validity of the constructor arguments considered in its constructor \
    (i.e. the constructor of the class :py:class:`.MultilayerClassifier`), which might not be all the \
    arguments of the constructor of the child class, and stores them; and the parent class stores all the rest \
    of constructor arguments of the child class, but those without checks: the child class will have to make those \
    checks by itself if considered necessary.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``
    conv_block_specification : list[int] or tuple[int]
        List of integers whose length indicates the number of tunable convolutional blocks of the network, and whose \
        $n$-th entry, $> 0$, indicates the number of equal layers in the same block.
    channels_per_conv_layer : tuple|list[int], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements), \
        wherein each entry indicates the number \
        of output channels of each convolutional-like layer of said block
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements) \
        and as many as `channels_per_conv_layer`. Each element of the list/tuple follows the convention of the \
        convolutional-like layers (:py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` and :py:class:`.IBNNLayer`) for their argument `m_kernel_size`
    phi_activation_per_conv_layer : tuple|list[str]
        Activation function used in the convolutional-like layers
    batch_normalization_per_conv_layer : tuple|list[bool]
        Whether batch normalization is to be used, indicated for the position after each convolutional-like layer
    maxpool_reduction_per_conv_block : tuple|list[int], optional
        Default: ``[1, ..., 1]`` (no reduction in any block)
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs)
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs)
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued
    fc_batch_normalization : bool, optional
        Whether batch normalization is to be used in the intermediate fully connected layers
    fc_dropout : float, optional
        Dropout probability for the fully connected layers.
        Default: ``0.0`` (no dropout)
    penciled_decision : bool, optional
        If ``True``, the (first, if more than one) FC layer is parallel to the vector (1,...,1).
        Default: ``False``
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer.
        Default: ``False``
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: ``True``
    device : str, optional
        Value among ``'cuda'`` and ``'cpu'``, device where the classifier is created. Error if ``'cuda'`` but \
        no GPU is available.
        Default: ``'cuda'`` if GPU available, ``'cpu'`` otherwise
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.IBNNLiteLayer`, and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_like_type, conv_like_type_position,
                 conv_block_specification, channels_per_conv_layer,
                 m_kernel_size_per_conv_layer,
                 phi_activation_per_conv_layer, batch_normalization_per_conv_layer=True,
                 maxpool_reduction_per_conv_block=1,
                 m_independent_channels=None, w_kernel_size=None, w_independent_channels=None,
                 fc_num_layers=None, fc_num_units_intermediate_layers=None, fc_batch_normalization=True,
                 fc_dropout=0.0, penciled_decision=False, softmax_output=False, prenormalization=False,
                 device=None,
                 **kwargs):

        # It inherits from ClassifierBaseModel
        super().__init__()

        # Initialize the structures required by ClassifierBaseModel
        self._extra_state_dict = {'constructor_kwargs': {}}
        self._fields_to_log = {}

        # Initialize the net
        self._nn = None

        ############################################
        ############################################
        # Check and store parameters
        ############################################
        ############################################

        ############################################
        # Store the network name
        ############################################

        self._extra_state_dict['net'] = type(self).__name__

        ############################################
        # Store the network input/output sizes
        ############################################

        # Check and store input size
        assert isinstance(in_size, (tuple, list)) and len(in_size) == 2 and \
               all(isinstance(x, int) and x > 0 for x in in_size), \
            f"Invalid 'in_size': tuple/list of 2 integers > 0 expected, {in_size} found!."
        self._extra_state_dict['in_size'] = copy.deepcopy(in_size)

        # Check and store number of input channels
        assert isinstance(in_channels, int) and in_channels > 0, \
            f"Invalid 'in_channels': integer > 0 expected, {in_channels} found!."
        self._extra_state_dict['in_channels'] = copy.deepcopy(in_channels)

        # Check and store number of output classes
        assert isinstance(out_classes, int) and out_classes > 0, \
            f"Invalid 'out_classes': integer > 0 expected, {out_classes} found!."
        self._extra_state_dict['out_classes'] = copy.deepcopy(out_classes)

        ############################################
        # Store the types and positions of the convolutional-like layers
        ############################################

        # Check and store the type of convolutional-like layers and their position
        assert isinstance(conv_like_type, str), \
            f"Invalid 'conv_like_type': string expected, {conv_like_type} found!."
        assert conv_like_type in _dict_conv_like_layers, \
            f"Invalid 'conv_like_type': {conv_like_type} not found in the list of available layers, that is, " + \
            f"{_dict_conv_like_layers.keys()}."
        self._extra_state_dict['conv_like_type'] = copy.deepcopy(conv_like_type)

        # Check and store the flag indicating whether the convolutional-like layer is used everywhere
        conv_like_type_position_allowable_values = ['everywhere', 'first', 'last']
        assert isinstance(conv_like_type_position, str), \
            f"Invalid 'conv_like_type_position': str expected, {conv_like_type_position} found!."
        assert conv_like_type_position in conv_like_type_position_allowable_values, \
            f"Invalid 'conv_like_type_position': {conv_like_type_position} not found in the list of allowable " + \
            f"values, that is, {conv_like_type_position_allowable_values}."
        self._extra_state_dict['conv_like_type_position'] = copy.deepcopy(conv_like_type_position)

        ############################################
        # Store the specification of blocks, and everything defined "per conv. layer"
        ############################################

        # Check if 'conv_block_specification' is a list/tuple.
        # And check whether the other arguments which are to be
        # linked to it are also lists/tuples and have the correct length: these arguments are:
        # 'channels_per_conv_layer', 'm_kernel_size_per_conv_layer', 'maxpool_reduction_per_conv_block',
        # 'batch_normalization_per_conv_layer', 'phi_activation_per_conv_layer'
        # And store the values of all these arguments in the state dict.

        num_conv_layers = None
        num_conv_blocks = None
        assert isinstance(conv_block_specification, (list, tuple)), \
            f"Invalid 'conv_block_specification': list or tuple expected, {type(conv_block_specification)} found!."
        assert all([isinstance(elem, int) and elem > 0 for elem in conv_block_specification]), \
            f"Invalid 'conv_block_specification': list of integers > 0 expected, {conv_block_specification} found!."
        # This gives the number of total convolutional-like layers in the net
        num_conv_layers = int(np.array(conv_block_specification).sum())
        num_conv_blocks = len(conv_block_specification)
        # Store the specification
        self._extra_state_dict['conv_block_specification'] = copy.deepcopy(tuple(conv_block_specification))

        _tmp_dict = {'maxpool_reduction_per_conv_block': maxpool_reduction_per_conv_block,
                     'channels_per_conv_layer': channels_per_conv_layer,
                     'm_kernel_size_per_conv_layer': m_kernel_size_per_conv_layer,
                     'batch_normalization_per_conv_layer': batch_normalization_per_conv_layer,
                     'phi_activation_per_conv_layer': phi_activation_per_conv_layer}

        for key, value in _tmp_dict.items():
            assert isinstance(value, (list, tuple)), \
                f"Invalid '{key}': list or tuple expected, {type(value)} found!."
            if key in ['maxpool_reduction_per_conv_block']:
                # Length must fit with the number of blocks
                assert len(value) == num_conv_blocks, \
                    (f"Invalid '{key}': list or tuple of length {num_conv_blocks} expected, due to " +
                     f"'conv_block_specification'={conv_block_specification}" +
                     f" {value} found!.")
                assert all([isinstance(elem, int) and elem > 0 for elem in value]), \
                    f"Invalid '{key}': list of integers > 0 expected, {value} found!."
            else:
                # Common for all the other keys: length must fit
                assert len(value) == num_conv_layers, \
                    (f"Invalid '{key}': list or tuple of length {num_conv_layers} expected, due to " +
                     f"'conv_block_specification'={conv_block_specification}" +
                     f" {value} found!.")
                # Now, the specific types of each key
                if key in ['channels_per_conv_layer']:
                    assert all([isinstance(elem, int) and elem > 0 for elem in value]), \
                        f"Invalid '{key}': list of integers > 0 expected, {value} found!."
                elif key == 'm_kernel_size_per_conv_layer':
                    pass
                    # Further checks later on 'm_kernel_size_per_conv_layer', once it has been stored
                elif key == 'batch_normalization_per_conv_layer':
                    assert all([isinstance(elem, bool) for elem in value]), \
                        f"Invalid '{key}': list of booleans expected, {value} found!."
                elif key == 'phi_activation_per_conv_layer':
                    assert all([isinstance(elem, str) for elem in value]), \
                        f"Invalid '{key}': list of strings expected, {value} found!."
                pass
            pass
            # Store the specification
            self._extra_state_dict[key] = copy.deepcopy(tuple(_tmp_dict[key]))
        pass

        # Check, and raise an exception if necessary, if the requested 'maxpool_reduction_per_conv_block' would
        # "desintegrate" the image before the end of the last block!
        total_maxpool_reduction = np.prod(self._extra_state_dict['maxpool_reduction_per_conv_block'])
        if total_maxpool_reduction > min(self._extra_state_dict['in_size']):
            raise Exception(
                (f"Invalid 'maxpool_reduction_per_conv_block' " +
                 f"{self._extra_state_dict['maxpool_reduction_per_conv_block']}): the total resulting reduction, " +
                 f"{total_maxpool_reduction}, is >= than the input size {self._extra_state_dict['in_size']}.")
            )
        pass

        # Further checks later on 'm_kernel_size_per_conv_layer'
        self._extra_state_dict['m_kernel_size_per_conv_layer'] = tuple(
            [kernel_size_check_and_reformat_into_tuple(m_kernel_size_i) \
             for m_kernel_size_i in self._extra_state_dict['m_kernel_size_per_conv_layer']]
        )

        ############################################
        # Other parameters of the convolutional-like layers
        ############################################

        # Check and store the flag indicating whether the channels of the convolutional-like layers are independent
        if m_independent_channels is None:
            raise Exception(f"Invalid 'm_independent_channels': it must be explicitly set to True or False; " +
                            f"no value, or None, provided.")
        else:
            assert isinstance(m_independent_channels, bool), \
            f"Invalid 'm_independent_channels': boolean expected, {m_independent_channels} found!."
        self._extra_state_dict['m_independent_channels'] = copy.deepcopy(m_independent_channels)

        # We check and store the values of w_kernel_size_proportion and w_independent_channels \
        # if they are relevant for the selected convolutional-like layer: otherwise, not part of the state dict.
        if self._extra_state_dict['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            #
            # Check and store the size/proportion of the spatial extent of the input image to each specific \
            # convolutional-like layer covered by the kernel $\\Omega$
            try:
                if w_kernel_size is None:
                    raise Exception(f"Invalid 'w_kernel_size': it must be explicitly set to a valid value; " +
                                    f"no value, or None, provided.")
                else:
                    self._extra_state_dict['w_kernel_size'] = \
                        copy.deepcopy(kernel_size_check_and_reformat_into_tuple(w_kernel_size, make_odd=True))
                pass
            except Exception as err:
                raise Exception(f"Error when checking 'w_kernel_size':\n{err}")
            #
            # Check and store the flag indicating whether the channels of the convolutional-like layers are independent
            if w_independent_channels is None:
                raise Exception(f"Invalid 'w_independent_channels': it must be explicitly set to True or False; " +
                                f"no value, or None, provided.")
            else:
                assert isinstance(w_independent_channels, bool), \
                    f"Invalid 'w_independent_channels': boolean expected, {w_independent_channels} found!."
                self._extra_state_dict['w_independent_channels'] = copy.deepcopy(w_independent_channels)
            pass
            #
        pass

        ############################################
        # Check and store the specification of the FC layers
        ############################################

        # Check and store the number of fully connected layers
        assert isinstance(fc_num_layers, int) and fc_num_layers > 0, \
            f"Invalid 'fc_num_layers': integer > 0 expected, {fc_num_layers} found!."
        self._extra_state_dict['fc_num_layers'] = copy.deepcopy(fc_num_layers)

        # Check and store the number of intermediate units in the fully connected layers
        if self._extra_state_dict['fc_num_layers'] == 1:
            if fc_num_units_intermediate_layers is not None and fc_num_units_intermediate_layers != -1:
                print((f"Warning: the provided 'fc_num_units_intermediate_layers' is not -1, but it should be for " +
                       f"the indicated single FC layer: setting it to -1."))
                self._extra_state_dict['fc_num_units_intermediate_layers'] = None
        else:
            assert isinstance(fc_num_units_intermediate_layers, int) and fc_num_units_intermediate_layers > 0, \
                (f"Invalid 'fc_num_units_intermediate_layers': integer > 0 expected, " +
                 f"{fc_num_units_intermediate_layers} found!.")
        pass
        self._extra_state_dict['fc_num_units_intermediate_layers'] = copy.deepcopy(fc_num_units_intermediate_layers)

        # Check and store 'fc_dropout':
        assert isinstance(fc_dropout, (int, float)) and 0 <= fc_dropout < 1, \
            f"Invalid 'fc_dropout': number in [0,1) expected, {fc_dropout} found!."
        self._extra_state_dict['fc_dropout'] = copy.deepcopy(float(fc_dropout))

        # Check and store 'fc_batch_normalization':
        assert isinstance(fc_batch_normalization, bool), \
            f"Invalid 'fc_batch_normalization': boolean expected, {fc_batch_normalization} found!"
        self._extra_state_dict['fc_batch_normalization'] = copy.deepcopy(fc_batch_normalization)

        ############################################
        # Check and store the flag indicating whether prenormalization, softmax
        ############################################

        dict_bools_to_check = {
            'prenormalization': prenormalization,
            'penciled_decision': penciled_decision,
            'softmax_output': softmax_output
        }
        for key, value in dict_bools_to_check.items():
            assert isinstance(value, bool), \
                f"Invalid '{key}': boolean expected, {value} found!."
            self._extra_state_dict[key] = copy.deepcopy(value)
        pass

        ######################################################
        ### Get all the fields that the conv-like layer would take as arguments by default in order to include them
        ### in the constructor and fields to log
        ######################################################

        # Fields in the constructor of the layers to leave out: the fields that will be filled individually per layer
        fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'm_kernel_size', 'm_groups',
                            'initial_lambda', 'w_groups']

        # Dictionary, but for the fields in the list 'fields_leave_out', for the selected conv-like layer
        conv_like_layer_kwargs = processed_constructor_kwargs_for_conv_like_layer(
            _dict_conv_like_layers[self._extra_state_dict['conv_like_type']],
            fields_leave_out=fields_leave_out,
            dict_explicit_args=None, dict_non_explicit_args=kwargs,
            flag_add_all_other_args=False
        )

        # Process 'initial_lambda', if present, specially: if it is a vector it must have as many elements as \
        # conv. layers in the net; if it is a single value, it is expanded to a vector with as many elements as \
        # conv. layers in the net, all with the same value.
        if conv_like_type in ['inrfv1', 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            if not 'initial_lambda' in kwargs:
                raise Exception(f"Invalid 'initial_lambda': it must be explicitly set to a valid value; " +
                                f"no value, or None, provided.")
            else:
                initial_lambda = kwargs['initial_lambda']
                # We make the 'initial_lambda' always be a vector with 1 lambda per conv-like layer
                # (even if all lambdas are the same)
                new_initial_lambda = None
                if conv_like_type_position in ['everywhere', 'all']:
                    if isinstance(initial_lambda, (list, tuple)):
                        assert len(initial_lambda) == num_conv_layers and all([isinstance(elem, (int, float)) for elem in initial_lambda]), \
                            (f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers} expected " +
                             f"(as many as conv-like layers); " +
                             f"{initial_lambda} found!.")
                        new_initial_lambda = tuple([float(elem) for elem in initial_lambda])
                    elif isinstance(initial_lambda, (int, float)):
                        new_initial_lambda = tuple([float(initial_lambda)]*num_conv_layers)
                    else:
                        raise TypeError(f"Invalid 'initial_lambda': expecting a scalar or a list/tuple of length " +
                                        f"{num_conv_layers} (as many as conv-like layers); " +
                                        f"{initial_lambda} found!")
                    pass
                elif conv_like_type_position in ['first', 'last'] or num_conv_layers == 1:
                    if isinstance(initial_lambda, (int, float)): # We make it a list of length 1
                        new_initial_lambda = [0.0]*num_conv_layers
                        if conv_like_type_position == 'first':
                            new_initial_lambda[0] = float(initial_lambda)
                        else:
                            new_initial_lambda[-1] = float(initial_lambda)
                        pass
                    elif isinstance(initial_lambda, (list, tuple)):
                        assert len(initial_lambda) == num_conv_layers, \
                            (f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers} expected " +
                             f"(as many as conv-like layers); " +
                             f"{initial_lambda} found!.")
                        assert all([isinstance(elem, (int, float)) for elem in initial_lambda]), \
                            (f"Invalid 'initial_lambda': list/tuple of numbers expected, " +
                             f"{initial_lambda} found!")
                        # Check that only the first or last element is non-zero
                        if conv_like_type_position == 'first':
                            assert all([float(elem) == 0.0 for elem in initial_lambda[1:]]), \
                                (f"Invalid 'initial_lambda': only the first element can be non-zero " +
                                 f"if 'conv_like_type_position' is 'first', " +
                                 f"{initial_lambda} found!")
                        else:
                            assert all([float(elem) == 0.0 for elem in initial_lambda[:-1]]), \
                                (f"Invalid 'initial_lambda': only the last element can be non-zero " +
                                 f"if 'conv_like_type_position' is 'last', " +
                                 f"{initial_lambda} found!")
                        pass
                        new_initial_lambda = [float(elem) for elem in initial_lambda]
                    else:
                        raise TypeError(f"Invalid 'initial_lambda': list/tuple of length {num_conv_layers} expected " +
                                        f"(as many as conv-like layers); " +
                                        f"{initial_lambda} found!")
                    # If it is a list of 1 element, expand to a list spaning the total number of conv-like layers
                    if isinstance(initial_lambda, (list, tuple)) and len(initial_lambda) == 1:
                        initial_lambda = [float(initial_lambda[0])]*num_conv_layers
                    pass
                else:
                    raise Exception(f"Invalid 'conv_like_type_position': {conv_like_type_position} not recognized!.")
                pass
                #
                # USE THE NEW initial_lambda
                initial_lambda = tuple(new_initial_lambda)
            pass
            # ALTHOUGH, CONCEPTUALLY, IT WOULD BE BETTER TO NAME THE INITIAL LAMBDA OF THE NETWORK WITH A DIFFERENT \
            # THAN THE 'initial_lambda' USED PER LAYER, AND TO PROCESS IT SEPARATELY FROM THE 'conv_like_layer_kwargs',\
            # THIS CHANGE WILL NOT BE PERFORMED NOW (POSSIBLY IN THE FUTURE).
            self._extra_state_dict['initial_lambda'] = copy.deepcopy(initial_lambda)
            conv_like_layer_kwargs['initial_lambda'] = initial_lambda
        pass

        ######################################################
        # Try to populate the constructor with the processed data
        ######################################################

        # Using the arguments provided for the class
        list_constructor_kwarg_names = list(inspect.signature(type(self)).parameters.keys())
        for kwarg_name in list_constructor_kwarg_names:
            if kwarg_name in self._extra_state_dict:
                self._extra_state_dict['constructor_kwargs'][kwarg_name] = \
                    copy.deepcopy(self._extra_state_dict[kwarg_name])
            elif kwarg_name in kwargs:
                self._extra_state_dict['constructor_kwargs'][kwarg_name] = \
                    copy.deepcopy(kwargs[kwarg_name])
            pass
        pass

        # Using the kwargs requested by the convolutional-like layer (if not already included)
        for key, value in conv_like_layer_kwargs.items():
            if key not in self._extra_state_dict['constructor_kwargs']:
                self._extra_state_dict['constructor_kwargs'][key] = copy.deepcopy(value)
            pass
        pass

        ############################################
        # Copy all fields of
        #    - self._extra_state_dict but self._extra_state_dict['constructor_kwargs'], and of
        #    - conv_like_layer_kwargs,
        # into self._fields_to_log
        ############################################

        # From "self._extra_state_dict" (but for "self._extra_state_dict['constructor_kwargs']", itself a dictionary)
        for key, value in self._extra_state_dict.items():
            if key != 'constructor_kwargs':
                self._fields_to_log[key] = copy.deepcopy(value)
            pass
        pass

        # From "self._extra_state_dict['constructor_kwargs']" if not yet included
        if 'constructor_kwargs' in self._extra_state_dict:
            for key, value in self._extra_state_dict['constructor_kwargs'].items():
                if key not in self._fields_to_log:
                    self._fields_to_log[key] = copy.deepcopy(value)
                pass
            pass
        pass

        ######################################################
        # ADD THE NUMBER OF CONV-LIKE LAYERS FOR LOGGING FOR CONVENIENCE
        ######################################################

        if not 'num_conv_layers' in self._fields_to_log:
            self._fields_to_log['num_conv_layers'] = num_conv_layers
        pass

        ############################################
        # Check and store the device
        ############################################

        # Computation device
        computation_device = None
        if device is None:  # default
            computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu'], \
                f"Invalid device: 'cuda' or 'cpu' expected, {device} found!."
            if device == 'cuda' and not torch.cuda.is_available():
                raise Exception(f"Invalid device: 'cuda' selected but no GPU available.")
            computation_device = device
        pass
        self.chosen_device = computation_device

    @staticmethod
    def random_initialization_subnetwork(subnetwork, distribution='normal', gain=1e-3, additive=True):
        """
        Function that iteratively randomizes the parameters of a subnetwork.

        Parameters
        ----------
        subnetwork : torch.nn.Module or torch.nn.Sequential
        distribution : str, optional
            Value among 'normal' and 'uniform'. Default: 'normal'
        gain : int or float, optional
            Gain/scale applied to the distribution. Default: 1e-3
        additive : bool, optional
            If True, randomness is added around current value; otherwise around zero. Default: True
        """
        if isinstance(subnetwork, nn.Sequential):
            for layer in subnetwork:
                MultilayerClassifier.random_initialization_subnetwork(layer,
                                                                      distribution=distribution,
                                                                      gain=gain,
                                                                      additive=additive)
        elif isinstance(subnetwork, (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer)):
            subnetwork.random_initialization(distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, nn.Module):
            for param_tensor in subnetwork.parameters():
                current_average = param_tensor.flatten().mean().item() if additive else 0.0
                if distribution == 'normal':
                    nn.init.normal_(param_tensor,
                                    mean=current_average, std=0.5 * gain)
                elif distribution == 'uniform':
                    nn.init.uniform_(param_tensor,
                                     a=current_average - 0.5 * gain,
                                     b=current_average + 0.5 * gain)

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        Randomizes the trainable parameters of the network.

        Parameters
        ----------
        distribution : str, optional
            Value among 'normal' and 'uniform'. Default: 'normal'
        gain : int or float, optional
            Gain/scale applied to the distribution. Default: 1e-3
        additive : bool, optional
            If True, randomness is added around current value; otherwise around zero. Default: True
        """
        self.random_initialization_subnetwork(self._nn, distribution=distribution,
                                              gain=gain, additive=additive)


#########################################################################################
#########################################################################################
# CLASSES: MultiHiddenLayerClassifier model
#########################################################################################
#########################################################################################


class MultiHiddenLayerClassifier(MultilayerClassifier):
    """
    Trainable classifier for images composed of n hidden layers followed by fully-connected layer(s).
    In greater detail:

    - The n hidden layers, somehow based on convolutions, are selected by the combination of the arguments \
    `conv_like_type` and `conv_like_type_position`:

        - `conv_like_type` is one of the \
          following options: `'sm'` (:py:class:`.SMLayer`), `'inrfv1'` (:py:class:`.INRFv1Layer`), \
          `'inrfv2'` (:py:class:`.INRFv2Layer`), `'inrfv3'` (:py:class:`.INRFv3Layer`), \
          `'ibnn_internal'` (:py:class:`.IBNNInternalLayer`), or \
          `'ibnn'` (:py:class:`.IBNNLayer`);
          related arguments dependent of the type of selected convolutional-like layer.

        - `conv_like_type_position` takes the values ``'everywhere'``, ``'first'``, and ``'last'``, which indicates \
          the position(s) where the specified `conv_like_type` is applied in the network (using the standard \
          convolution `'sm'` in the rest of them).

    - The output of each hidden layer can be mediated by the following \
      optional operations: batch normalization and/or maxpool. The provided values are used for both hidden layers.

    - The classification output can be the raw output of the fully-connected layer or, optionally, its *softmax*.

    Regarding the decision based on a FC-based head: the argument `penciled_decision` (which defaults to ``False``), \
    if set to `True`, makes the (first, if more than one) FC layer, although trainable, always parallel to the vector \
    (1,...,1): this behavior is \
    achieved through the parameterization :py:class:`.utils.PencilOfPlanes`.

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, with ``w_kernel_size`` = ``()`` \
    (empty tuple), **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``).
    conv_block_specification : tuple|list[int]
        List of integers whose length indicates the number of convolutional blocks of the network, and whose \
        $n$-th entry, $> 0$, indicates the number of convolutional sub-layer in said block.
    channels_per_conv_layer : tuple|list[int], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements), \
        wherein each entry indicates the number \
        of output channels of each convolutional-like layer of said block.
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements) \
        and as many as `channels_per_conv_layer`. Each element of the list/tuple follows the convention of the \
        convolutional-like layers (:py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` and :py:class:`.IBNNLayer`) for their argument `m_kernel_size`
    phi_activation_per_conv_layer : tuple|list[str]
        Activation function used in the convolutional-like layers
    batch_normalization_per_conv_layer : tuple|list[bool]
        Whether batch normalization is to be used, indicated for the position after each convolutional-like layer
    maxpool_reduction_per_conv_block : tuple|list[int], optional
        Default: ``[1, ..., 1]`` (no reduction in any layer)
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs)
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs)
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued
    fc_batch_normalization : bool, optional
        Whether batch normalization is to be used in the intermediate fully connected layers
    fc_dropout : float, optional
        fc_dropout probability for the fully connected layers.
        Default: ``0.0`` (no fc_dropout)
    penciled_decision : bool, optional
        If ``True``, the (first, if more than one) FC layer is parallel to the vector (1,...,1).
        Default: ``False``
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer.
        Default: ``False``
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: ``True``
    device : str, optional
        Value among ``'cuda'`` and ``'cpu'``, device where the classifier is created. Error if ``'cuda'`` but \
        no GPU is available.
        Default: ``'cuda'`` if GPU available, ``'cpu'`` otherwise
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.IBNNLiteLayer`, and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_like_type, conv_like_type_position,
                 conv_block_specification, channels_per_conv_layer,
                 m_kernel_size_per_conv_layer,
                 phi_activation_per_conv_layer, batch_normalization_per_conv_layer=None,
                 maxpool_reduction_per_conv_block=None,
                 m_independent_channels=None, w_kernel_size=None, w_independent_channels=None,
                 fc_num_layers=None, fc_num_units_intermediate_layers=None, fc_batch_normalization=True,
                 fc_dropout=0.0, penciled_decision=False, softmax_output=False, prenormalization=False,
                 device=None,
                 **kwargs):

        # Default values for 'maxpool_reduction_per_conv_block' and 'batch_normalization_per_conv_layer':
        if isinstance(conv_block_specification, (list, tuple)) and all([isinstance(x, int) and x > 0 for x in conv_block_specification]):
            num_blocks = len(conv_block_specification)
            num_conv_layers = int(np.array(conv_block_specification).sum())
            if maxpool_reduction_per_conv_block is None:
                maxpool_reduction_per_conv_block = [1]*num_blocks
            if batch_normalization_per_conv_layer is None:
                batch_normalization_per_conv_layer = [True]*num_conv_layers
            pass
        else:
            raise Exception(f"Invalid conv_block_specification: {conv_block_specification}: " +
                            f"list/tuple of positive integers expected, {conv_block_specification} found!")
        pass

        ######################################################
        # RUN THE CONSTRUCTOR OF THE PARENT CLASS MultilayerClassifier SO:
        # - THE STRUCTURES OF 'nn.Module' ARE CREATED
        # - THE SPECIFIC CHECKS OF 'MultilayerClassifier' ARE RUN
        ######################################################

        MultilayerClassifier.__init__(
            self,
            in_size=in_size, in_channels=in_channels, out_classes=out_classes,
            conv_like_type=conv_like_type, conv_like_type_position=conv_like_type_position,
            conv_block_specification=conv_block_specification,
            channels_per_conv_layer=channels_per_conv_layer,
            m_kernel_size_per_conv_layer=m_kernel_size_per_conv_layer,
            phi_activation_per_conv_layer=phi_activation_per_conv_layer,
            batch_normalization_per_conv_layer=batch_normalization_per_conv_layer,
            maxpool_reduction_per_conv_block=maxpool_reduction_per_conv_block,
            m_independent_channels=m_independent_channels,
            w_kernel_size=w_kernel_size, w_independent_channels=w_independent_channels,
            fc_num_layers=fc_num_layers, fc_num_units_intermediate_layers=fc_num_units_intermediate_layers,
            fc_batch_normalization=fc_batch_normalization, penciled_decision=penciled_decision,
            fc_dropout=fc_dropout, softmax_output=softmax_output, prenormalization=prenormalization, device=device,
            **kwargs
        )

        # MultilayerClassifier makes multiple checks: for instance, it checks that the sum of the entries in
        # "conv_block_specification" equals the length of the arguments targeting individual layers
        # (e.g. 'channels_per_conv_layer', 'phi_activation_per_conv'...). However, in MultiHiddenLayerClassifier
        # we are always allowing blocks of ONE SINGLE LAYER: that means that each entry of 'conv_block_specification'
        # MUST be equal to 1. We enforce this here:
        assert all([x == 1 for x in self._extra_state_dict['conv_block_specification']]), \
            f"In MultiHiddenLayerClassifier all blocks must be of a single layer, " \
            f"but conv_block_specification={self._extra_state_dict['conv_block_specification']} found!"

        ######################################################
        # INITIALIZE THE STRUCTURES REQUIRED BY 'ClassifierBaseModel' WHICH WILL BE FILLED IN THE CONSTRUCTOR
        ######################################################

        prenormalization_module = None
        backbone_module = None
        head_module = None

        ######################################################
        ### LAYER PARAMETER SEPARATION:
        ### The will generate the required constructor kwargs for 'sm' and for the selected conv-like layer
        ### to ease their use for populating the backbone of the network
        ######################################################

        # Fields in the constructor of the layers to leave out: the fields that will be filled individually per layer
        fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'phi_activation', 'm_kernel_size', 'm_groups', 'w_groups']

        # Dictionary of explicit args in the network that we want to force
        # dict_explicit_args_sm = {'phi_activation': self._extra_state_dict['phi_activation'],
        #                          'm_padding': 'same'}
        # dict_explicit_args_sm = {'m_padding': 'same'}
        dict_explicit_args_sm = {}
        dict_explicit_args_conv_like_layer = copy.deepcopy(dict_explicit_args_sm)
        if self._extra_state_dict['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            dict_explicit_args_conv_like_layer['w_kernel_size'] = self._extra_state_dict['w_kernel_size']
        pass

        # Dictionary, but for the fields in the list 'fields_leave_out', for the SMLayers
        sm_kwargs = processed_constructor_kwargs_for_conv_like_layer(
            _dict_conv_like_layers['sm'],
            fields_leave_out=fields_leave_out,
            dict_explicit_args=dict_explicit_args_sm, dict_non_explicit_args=kwargs,
            flag_add_all_other_args=False
        )

        # Dictionary, but for the fields in the list 'fields_leave_out', for the selected conv-like layer
        conv_like_layer_kwargs = processed_constructor_kwargs_for_conv_like_layer(
            _dict_conv_like_layers[self._extra_state_dict['conv_like_type']],
            fields_leave_out=fields_leave_out,
            dict_explicit_args=dict_explicit_args_conv_like_layer, dict_non_explicit_args=kwargs,
            flag_add_all_other_args=True
        )

        ######################################################
        ### Add all the args in the constructor and the 'conv_like_layer_kwargs' into the constructor kwargs
        ######################################################

        dict_to_get_in_constructor_kwargs = {
            'in_size': in_size, 'in_channels': in_channels, 'out_classes': out_classes,
            'conv_like_type': conv_like_type, 'conv_like_type_position': conv_like_type_position,
            'conv_block_specification': conv_block_specification, 'channels_per_conv_layer': channels_per_conv_layer,
            'm_kernel_size_per_conv_layer': m_kernel_size_per_conv_layer,
            'phi_activation_per_conv_layer': phi_activation_per_conv_layer,
            'batch_normalization_per_conv_layer': batch_normalization_per_conv_layer,
            'maxpool_reduction_per_conv_block': maxpool_reduction_per_conv_block,
            'm_independent_channels': m_independent_channels,
            'w_kernel_size': w_kernel_size, 'w_independent_channels': w_independent_channels,
            'fc_num_layers': fc_num_layers, 'fc_num_units_intermediate_layers': fc_num_units_intermediate_layers,
            'fc_batch_normalization': fc_batch_normalization,
            'prenormalization': prenormalization, 'penciled_decision': penciled_decision, 'softmax_output': softmax_output, 'device': device
        }

        for key in dict_to_get_in_constructor_kwargs.keys():
            if key not in self._extra_state_dict:
                self._extra_state_dict['constructor_kwargs'][key] = \
                    copy.deepcopy(dict_to_get_in_constructor_kwargs[key])
            else:
                # If already present, we keep the value in the constructor kwargs
                self._extra_state_dict['constructor_kwargs'][key] = copy.deepcopy(self._extra_state_dict[key])
            pass
        pass

        # ... plus everything retrieved from the needs of the conv-like layer
        for key in conv_like_layer_kwargs:
            self._extra_state_dict['constructor_kwargs'][key] = copy.deepcopy(conv_like_layer_kwargs[key])
        pass

        ######################################################
        ### Include all the arguments in the 'self._extra_state_dict['constructor_kwargs'] into 'self._fields_to_log'
        ### BUT ONLY IF THEY ARE NOT ALREADY PRESENT!!!
        ### And make it hashable if not already
        ######################################################
        for key in self._extra_state_dict['constructor_kwargs']:
            if key not in self._fields_to_log:
                field_to_store = self._extra_state_dict['constructor_kwargs'][key]
                if not isinstance(field_to_store, typing.Hashable):
                    if isinstance(field_to_store, list):
                        field_to_store = tuple(field_to_store)
                    else:
                        field_to_store = str(field_to_store)
                    pass
                pass
                self._fields_to_log[key] = copy.deepcopy(field_to_store)
            pass
        pass

        ######################################################
        ### FORCE ALL FIELDS IN "self._fields_to_log" TO BE HASHABLE... by transforming to tuple or str if necessary
        ######################################################
        for key in self._fields_to_log:
            if not isinstance(self._fields_to_log[key], typing.Hashable):
                if isinstance(self._fields_to_log[key], list):
                    self._fields_to_log[key] = tuple(self._fields_to_log[key])
                else:
                    self._fields_to_log[key] = str(self._fields_to_log[key])
                pass
            pass
        pass

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: BACKBONE
        ######################################################
        ######################################################

        # 'backbone_module' is a sequential to be filled with the blocks of the backbone. In order to give names,
        # we use an OrderedDict, which is then converted to a nn.Sequential
        ordered_dict_backbone_module = OrderedDict()

        ######################################################
        # Building the block of the first-and-only conv-like layer
        ######################################################

        num_tunable_layers = len(self._extra_state_dict['channels_per_conv_layer'])

        # Output size and channels for the next layer, taken first from the input images
        out_size_block_conv_i_minus_1 = self._extra_state_dict['in_size']
        out_channels_block_conv_i_minus_1 = self._extra_state_dict['in_channels']

        for ind_layer in range(num_tunable_layers):
            #
            # Set the parameters to use in this layer; this mostly regards the type of conv-like layer and the \
            # initial lambda, if relevant
            conv_like_type_i = 'sm'
            conv_like_layer_kwargs_i = None
            if (self._extra_state_dict['conv_like_type_position'] == 'everywhere' or
                    (self._extra_state_dict['conv_like_type_position'] == 'first' and ind_layer == 0) or
                    (self._extra_state_dict['conv_like_type_position'] == 'last' and ind_layer == num_tunable_layers - 1)):
                conv_like_type_i = self._extra_state_dict['conv_like_type']
            if conv_like_type_i == 'sm':
                conv_like_layer_kwargs_i = copy.deepcopy(sm_kwargs)
            else:
                conv_like_layer_kwargs_i = copy.deepcopy(conv_like_layer_kwargs)
                initial_lambda = conv_like_layer_kwargs_i.get('initial_lambda', None)
                if initial_lambda is not None:
                    if isinstance(initial_lambda, (list, tuple)):
                        conv_like_layer_kwargs_i['initial_lambda'] = initial_lambda[ind_layer]
                    elif isinstance(initial_lambda, (int, float)):
                        conv_like_layer_kwargs_i['initial_lambda'] = float(initial_lambda)
                    else:
                        raise Exception(f"Invalid initial_lambda: float or list/tuple of floats expected, "
                                        f"but {type(initial_lambda)} found!")
                    pass
                pass
            pass
            #
            # Build the block
            block_conv_i, name_block_conv_i, out_size_block_conv_i = \
                _create_standard_block_conv_i(
                    conv_like_type=conv_like_type_i,
                    in_channels=out_channels_block_conv_i_minus_1,
                    out_channels=self._extra_state_dict['channels_per_conv_layer'][ind_layer],
                    phi_activation=self._extra_state_dict['phi_activation_per_conv_layer'][ind_layer],
                    m_kernel_size=self._extra_state_dict['m_kernel_size_per_conv_layer'][ind_layer],
                    m_independent_channels=self._extra_state_dict['m_independent_channels'],
                    w_independent_channels=self._extra_state_dict.get('w_independent_channels', None),
                    in_size=out_size_block_conv_i_minus_1,
                    ind_block=ind_layer,
                    maxpool_reduction=self._extra_state_dict['maxpool_reduction_per_conv_block'][ind_layer],
                    batch_normalization=self._extra_state_dict['batch_normalization_per_conv_layer'][ind_layer],
                    **conv_like_layer_kwargs_i)
            #
            ordered_dict_backbone_module.update({name_block_conv_i: block_conv_i})
            #
            # Update the output size and channels for the next layer
            out_size_block_conv_i_minus_1 = out_size_block_conv_i
            out_channels_block_conv_i_minus_1 = self._extra_state_dict['channels_per_conv_layer'][ind_layer]
            #
        pass

        ######################################################
        # Pack the ordered dict into the backbone module
        ######################################################

        backbone_module = nn.Sequential(ordered_dict_backbone_module)

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: HEAD
        ######################################################
        ######################################################

        num_features_after_flatten = \
            int(out_channels_block_conv_i_minus_1 * np.prod(out_size_block_conv_i_minus_1))

        head_module = _create_standard_head_module(
            num_features_in=num_features_after_flatten, num_features_out=self._extra_state_dict['out_classes'],
            num_layers=self._extra_state_dict['fc_num_layers'],
            num_features_intermediate_layers=self._extra_state_dict['fc_num_units_intermediate_layers'],
            batch_normalization=self._extra_state_dict['fc_batch_normalization'],
            dropout=self._extra_state_dict['fc_dropout'],
            penciled_decision=self._extra_state_dict['penciled_decision'],
            softmax_output=self._extra_state_dict['softmax_output']
        )

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK:
        ### CREATE sef.prenormalization and integrate everything into self._nn
        ### AND MOVE IT TO THE DESIRED DEVICE
        ######################################################
        ######################################################

        # Prenormalization
        prenormalization_module = nn.Identity() if not prenormalization else \
            nn.BatchNorm2d(num_features=in_channels, affine=False, track_running_stats=True)

        # Create the full network by combining backbone and head
        self._nn = nn.Sequential(OrderedDict([
            ('prenormalization', prenormalization_module),
            ('backbone', backbone_module),
            ('head', head_module)
        ]))

        # Computation device
        computation_device = None
        if device is None:  # default
            computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu'], \
                f"Invalid device: 'cuda' or 'cpu' expected, {device} found!."
            if device == 'cuda' and not torch.cuda.is_available():
                raise Exception(f"Invalid device: 'cuda' selected but no GPU available.")
            computation_device = device
        pass
        self.to_device(computation_device)

        ######################################################
        ######################################################

    pass


pass

#########################################################################################
#########################################################################################
# CLASSES: VGGxClassifier model
#########################################################################################
#########################################################################################


class VGGxClassifier(MultilayerClassifier):
    """
    Customizable classifier inspired by the family of neural network architectures VGG [Simonyan2015]_. \
    The present class :py:class:`.VGGxClassifier`  allows the creation of a classifier containing the number and \
    type of convolutional-like layers indicated in the constructor, as well as a number of final fully-connected layer \
    also specified therein.

    The (strict) VGG networks originally published comprise multiple convolutional blocks with the structure \
    $(N \\times \\mathrm{CNN})+\\mathrm{maxpool}$, in some cases $\\mathrm{CNN}+\\mathrm{maxpool}$ and \
    in others $(2 \\times \\mathrm{CNN})+\\mathrm{maxpool}$, having a reduced spatial extent but doubling the \
    number of channels and halving the spatial resolution: the first block has convolutional layers generating \
    $64$ channels, from an input of in principle one channel; and in general the convolutional layer(s) \
    of the $n$-th block generates $64 \\, n$ channels. (As a note, the number *<x>* in the name VGG<x> refers to the \
    number of *weight layers*, that is, removing the pooling layers.) The CNN layers use ``'same'`` padding \
    in the original description of the network.
    After the convolutional blocks the network of the original publication [Simonyan2015]_ has 3 fully-connected \
    layers, wherein their respective (flattened) in- and output sizes are:

    $$
    \\textrm{length} \\rightarrow \\boxed{FC1} \\rightarrow 4096 \\rightarrow
    \\boxed{FC2} \\rightarrow 4096 \\rightarrow \\boxed{FC3} \\rightarrow 1000
    $$

    with 1000 the number of classes in [Simonyan2015]_ (*i.e.* of the ImageNet dataset), followed by a softmax output.

    The present class :py:class:`.VGGxClassifier` allows the definition of a network following, roughly, \
    the general principles of the original VGG structure but providing additionally the following potential variations:

    - The option of substituting CNN layers by other convolutional-like layers, namely INRF or ibnn_internal layers; \
      this option includes substituting all CNN layers or only the first one.

    - It allows for kernels *m* of different size than the original $3 \\times 3$ size presumed for all convolutional \
      layers by the original VG`G.

    - The constructor allows the specification of the number of convolutional blocks and the number of \
      convolutional-like layers present in each one of the blocks, as well as the number of exit channels that \
      the first layer of the first block has: this last parameter, embodied by the argument `base_channels`, \
      would correspond to the original value of $64$ of the original VGG and would cause the subsequent $n$-th block \
      to have convolutional-like layers with $(\\textrm{base_channels} \\times n)$ output channels.

    - It allows for maxpool with a custom factor `maxpool_reduction` at the end of each block, \
      unlike the original VGG which has a fixed factor of $2$.

    - It allows to specify the number of intermediate units and the number of FC layers.

    - Unlike in the original publication [Simonyan2015]_ we leave the softmax output as an option.

    - The class allows for batch normalization layers which, if applied, are applied after each and every \
    convolutional-like and FC layer.

    References
    ----------
    .. [Simonyan2015] *K. Simonyan, A. Zisserman*. Very Deep Convolutional Networks for Large-Scale Image Recognition, \
            CoRR abs/1409.1556, 2014 (in ICLR 2015). `link <https://arxiv.org/pdf/1409.1556>`_

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, with ``w_kernel_size`` = ``()`` \
    (empty tuple), **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``)
    conv_block_specification : list[int] or tuple[int]
        List of integers whose length indicates the number of convolutional blocks of the network, and whose \
        $n$-th entry, $> 0$, indicates the number of convolutional sub-layer in said block.
    base_channels : int, optional
        Number of output channels of the convolutional-like layers of the first block, and base for the number \
        of output channels of the convolutional-like layers of the $n$-th block according to \
        $(\\textrm{base_channels} \\times n)$. Default: ``64`` (as in the original VGG)
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$. Default: 3
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued.
        Default: ``4096``
    maxpool_reduction : int, optional
        If greater than 1, the indication is the reduction factor used at the end of each and every block indicated \
        in the net specification `conv_block_specification`. \
        Default: `1`
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: `True`
    batch_normalization : bool, optional
        Default: `True`
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer. Default: ``False``
    phi_activation : str, optional
        Activation function used in the convolutional-like layers. Default: ``'relu'``
    m_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float, optional
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image. \
        Default: ``(3, 3)`` (as in the original VGG)
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs). Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image. The argument is set with a default ``None``, but the \
        constructor **raises an Exception if no value, or** ``None``, **is provided** and the type of indicated \
        ``conv_like_type`` is among ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs). Default: ``True``
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_block_specification, conv_like_type, conv_like_type_position,
                 base_channels=64, fc_num_layers=3, fc_num_units_intermediate_layers=4096,
                 maxpool_reduction=1, prenormalization=False, batch_normalization=True, softmax_output=False,
                 phi_activation='relu', m_kernel_size=(3, 3), m_independent_channels=False,
                 w_kernel_size=None, w_independent_channels=True, fc_dropout=0.0,
                 **kwargs):

        ######################################################
        # RUN THE PARENT CLASS CONSTRUCTOR SO THE STRUCTURES OF 'nn.Module' ARE CREATED
        ######################################################
        super().__init__(in_size, in_channels, out_classes,
                         batch_normalization, softmax_output)

        ######################################################
        # INITIALIZE THE STRUCTURES REQUIRED BY 'ClassifierBaseModel' WHICH WILL BE FILLED IN THE CONSTRUCTOR
        ######################################################

        self._extra_state_dict = {}
        self._extra_state_dict['constructor_kwargs'] = {}  # TO BE FILLED LATER!!!

        self._fields_to_log = {}

        prenormalization_module = None
        backbone_module = None
        head_module = None
        self._nn = None

        #

        self._extra_state_dict['net'] = type(self).__name__
        self._fields_to_log['net'] = type(self).__name__

        ######################################################
        ### INITIAL, GENERAL PARAMETERS: checks and storage
        ### Remember: as inheriting from ClassifierBaseModel, this class has to create the following structures:
        ### - self._fields_to_log
        ### - self._extra_state_dict containing, at least, the fields 'net' and 'constructor_kwargs', the latter \
        ###  generating the same object
        ######################################################

        # Check and store the input size
        self._extra_state_dict['constructor_kwargs']['in_size'] \
            = copy.deepcopy(kernel_size_check_and_reformat_into_tuple(in_size, make_odd=False))

        # Check and store the number of input channels
        assert isinstance(in_channels, int) and in_channels > 0, \
            f"Invalid 'in_channels': integer > 0 expected, {in_channels} found!."
        self._extra_state_dict['constructor_kwargs']['in_channels'] = copy.deepcopy(in_channels)

        # Check and store the number of output classes
        assert isinstance(out_classes, int) and out_classes > 0, \
            f"Invalid 'out_classes': integer > 0 expected, {out_classes} found!."
        self._extra_state_dict['constructor_kwargs']['out_classes'] = copy.deepcopy(out_classes)

        # Check and store the specification of the convolutional blocks
        assert isinstance(conv_block_specification, (list, tuple)), \
            f"Invalid 'conv_block_specification': list or tuple expected, {type(conv_block_specification)} found!."
        assert all([isinstance(elem, int) and elem > 0 for elem in conv_block_specification]), \
            f"Invalid 'conv_block_specification': list of integers > 0 expected, {conv_block_specification} found!."
        self._extra_state_dict['constructor_kwargs']['conv_block_specification'] = \
            copy.deepcopy(tuple(conv_block_specification))

        # Check and store the type of convolutional-like layer
        assert isinstance(conv_like_type, str), \
            f"Invalid 'conv_like_type': string expected, {conv_like_type} found!."
        assert conv_like_type in _dict_conv_like_layers, \
            f"Invalid 'conv_like_type': {conv_like_type} not found in the list of available layers, that is, " + \
            f"{_dict_conv_like_layers.keys()}."
        self._extra_state_dict['constructor_kwargs']['conv_like_type'] = copy.deepcopy(conv_like_type)

        # Check and store the flag indicating whether the convolutional-like layer is used everywhere
        conv_like_type_position_allowable_values = ['everywhere', 'first', 'last']
        assert isinstance(conv_like_type_position, str), \
            f"Invalid 'conv_like_type_position': str expected, {conv_like_type_position} found!."
        assert conv_like_type_position in conv_like_type_position_allowable_values, \
            f"Invalid 'conv_like_type_position': {conv_like_type_position} not found in the list of allowable " + \
            f"values, that is, {conv_like_type_position_allowable_values}."
        self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] \
            = copy.deepcopy(conv_like_type_position)

        # Check and store the number of base channels
        assert isinstance(base_channels, int) and base_channels > 0, \
            f"Invalid 'base_channels': integer > 0 expected, {base_channels} found!."
        self._extra_state_dict['constructor_kwargs']['base_channels'] = copy.deepcopy(base_channels)

        # Check and store the number of fully connected layers
        assert isinstance(fc_num_layers, int) and fc_num_layers > 0, \
            f"Invalid 'fc_num_layers': integer > 0 expected, {fc_num_layers} found!."
        self._extra_state_dict['constructor_kwargs']['fc_num_layers'] = copy.deepcopy(fc_num_layers)

        # Check and store the number of intermediate units in the fully connected layers
        if self._extra_state_dict['constructor_kwargs']['fc_num_layers'] == 1:
            if fc_num_units_intermediate_layers is not None:
                print((f"Warning: the provided 'fc_num_units_intermediate_layers' is not -1, but it should be for " +
                       f"the indicated single FC layer: setting it to -1."))
                self._extra_state_dict['constructor_kwargs']['fc_num_units_intermediate_layers'] = None
        else:
            assert isinstance(fc_num_units_intermediate_layers, int) and fc_num_units_intermediate_layers > 0, \
                f"Invalid 'fc_num_units_intermediate_layers': integer > 0 expected, {fc_num_units_intermediate_layers} found!."
        pass
        self._extra_state_dict['constructor_kwargs']['fc_num_units_intermediate_layers'] \
            = copy.deepcopy(fc_num_units_intermediate_layers)

        # Check and store the maxpool_reduction factor
        assert isinstance(maxpool_reduction, int) and maxpool_reduction > 0, \
            f"Invalid 'maxpool_reduction': int x>0 expected, object of type {type(maxpool_reduction)} " + \
            f"and value {maxpool_reduction} found!."
        self._extra_state_dict['constructor_kwargs']['maxpool_reduction'] = copy.deepcopy(maxpool_reduction)

        # Check and store the flag indicating whether batch normalization is used
        assert isinstance(prenormalization, bool), \
            f"Invalid 'prenormalization': boolean expected, {prenormalization} found!."
        self._extra_state_dict['constructor_kwargs']['prenormalization'] = copy.deepcopy(prenormalization)
        self._extra_state_dict['prenormalization'] = copy.deepcopy(prenormalization)

        # Check and store the flag indicating whether batch normalization is used
        assert isinstance(batch_normalization, bool), \
            f"Invalid 'batch_normalization': boolean expected, {batch_normalization} found!."
        self._extra_state_dict['constructor_kwargs']['batch_normalization'] = copy.deepcopy(batch_normalization)
        self._extra_state_dict['batch_normalization'] = copy.deepcopy(batch_normalization)

        # Check and store the flag indicating whether the output is passed through a softmax layer
        assert isinstance(softmax_output, bool), \
            f"Invalid 'softmax_output': boolean expected, {softmax_output} found!."
        self._extra_state_dict['constructor_kwargs']['softmax_output'] = copy.deepcopy(softmax_output)
        self._extra_state_dict['softmax_output'] = copy.deepcopy(softmax_output)

        # Check and store the activation function used in the convolutional-like layers
        assert isinstance(phi_activation, str), \
            f"Invalid 'phi_activation': string expected, {phi_activation} found!."
        self._extra_state_dict['constructor_kwargs']['phi_activation'] = copy.deepcopy(phi_activation)

        # Check and store the size/proportion of the spatial extent of the input image to each specific \
        # convolutional-like layer covered by the kernel $\\Omega$
        try:
            self._extra_state_dict['constructor_kwargs']['m_kernel_size'] = \
                copy.deepcopy(kernel_size_check_and_reformat_into_tuple(m_kernel_size, make_odd=True))
        except Exception as err:
            raise Exception(f"Error when checking 'm_kernel_size': {err}")

        # Check and store the flag indicating whether the channels of the convolutional-like layers are independent
        assert isinstance(m_independent_channels, bool), \
            f"Invalid 'm_independent_channels': boolean expected, {m_independent_channels} found!."
        self._extra_state_dict['constructor_kwargs']['m_independent_channels'] = copy.deepcopy(m_independent_channels)

        # We check and store the values of w_kernel_size_proportion and w_independent_channels \
        # if they are relevant for the selected convolutional-like layer: otherwise, not part of the state dict.
        if self._extra_state_dict['constructor_kwargs']['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal',
                                                                              'ibnn']:
            #
            # Check and store the size/proportion of the spatial extent of the input image to each specific \
            # convolutional-like layer covered by the kernel $\\Omega$
            try:
                self._extra_state_dict['constructor_kwargs']['w_kernel_size'] = \
                    copy.deepcopy(kernel_size_check_and_reformat_into_tuple(w_kernel_size, make_odd=True))
            except Exception as err:
                raise Exception(f"Error when checking 'w_kernel_size':\n{err}")
            #
            # Check and store the flag indicating whether the channels of the convolutional-like layers are independent
            assert isinstance(w_independent_channels, bool), \
                f"Invalid 'w_independent_channels': boolean expected, {w_independent_channels} found!."
            self._extra_state_dict['constructor_kwargs']['w_independent_channels'] \
                = copy.deepcopy(w_independent_channels)
            #
        pass

        # Store the rest of arguments added as 'kwargs' in 'constructor_kwargs' of the 'self._extra_state_dict'
        for key in kwargs:
            self._extra_state_dict['constructor_kwargs'][key] = copy.deepcopy(kwargs[key])
        pass

        # Include all the arguments in the 'self._extra_state_dict['constructor_kwargs'] into 'self._fields_to_log'
        for key in self._extra_state_dict['constructor_kwargs']:
            self._fields_to_log[key] = copy.deepcopy(self._extra_state_dict['constructor_kwargs'][key])
        pass

        ######################################################
        ### GENERAL PARAMETERS OF THE M-PART OF THE CONV-LIKE LAYERS: some are fixed in VGG
        ### NOTE: 'phi_activation' is now customisable (a keyword argument with default), \
        ### but for "continuity" it is assigned here.
        ######################################################

        basic_sm_layer_kwargs = {
            'phi_activation': phi_activation, 'm_padding': 'same',
            'm_kernel_size': self._extra_state_dict['constructor_kwargs']['m_kernel_size']
        }
        for key in ['m_padding']:
            if key in kwargs:
                print(f"Warning: key '{key}' does not allow for user defined values in VGGxClassifier: " +
                      f"its value is always defined as {basic_sm_layer_kwargs[key]}. " +
                      f"The provided value {kwargs[key]} is thus ignored.")
                kwargs.pop(key, None)
            pass
        pass

        # The basic parameters for the 'inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn' include the 'w_kernel_size'
        basic_conv_like_layer_kwargs = copy.deepcopy(basic_sm_layer_kwargs)
        if self._extra_state_dict['constructor_kwargs']['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal',
                                                                              'ibnn']:
            basic_conv_like_layer_kwargs['w_kernel_size'] = self._extra_state_dict['constructor_kwargs'][
                'w_kernel_size']
        pass

        ######################################################
        ### SEPARATE THE PARAMETERS OF THE "KWARGS" CORRESPONDING TO THE SELECTED CONV-LIKE LAYER, IF NOT 'SM',
        ### AND OF 'SM' (IN CASE THERE ARE 'SM' LAYERS)
        ######################################################

        ### Fill a dictionary with the parameters relevant for the 'sm' layers

        sm_kwargs = {}

        # All args of the SMLayer, with default values or not, minus 'in_channels' and 'out_channels' and 'm_groups'
        sm_potential_args = _dict_conv_like_layers['sm'].constructor_default_values(only_not_none=False)
        # We remove those because they are created per layer
        sm_potential_args.pop('in_channels', None)
        sm_potential_args.pop('out_channels', None)
        sm_potential_args.pop('m_groups', None)

        # All args of the SMLayer with default values
        sm_kwargs_default_kwargs = _dict_conv_like_layers['sm'].constructor_default_values(only_not_none=True)

        # Hierarchically fill the potential arguments of the SMLayer: "forced", present here in kwargs, or default
        for key in sm_potential_args:
            if key in basic_sm_layer_kwargs:
                sm_kwargs[key] = copy.deepcopy(basic_sm_layer_kwargs[key])
            elif key in kwargs:
                sm_kwargs[key] = copy.deepcopy(kwargs[key])
            elif key in sm_kwargs_default_kwargs:
                sm_kwargs[key] = copy.deepcopy(sm_kwargs_default_kwargs[key])
            pass
        pass

        # And add the argument 'm_independent_channels'
        # sm_kwargs['m_independent_channels'] = copy.deepcopy(m_independent_channels)

        ### Fill a dictionary with the parameters relevant for the selected conv-like layer (which, if 'sm', \
        ###  has already been anyway done; in the other cases, the 'sm' params are a subset) and remove \
        ### them from the 'kwargs'

        conv_like_layer_kwargs = {}

        # All args of the Layer, with default values or not, minus 'in/out_channels' and 'm/w_groups'
        conv_like_layer_potential_args = \
            _dict_conv_like_layers[self._fields_to_log['conv_like_type']].constructor_default_values(
                only_not_none=False
            )
        # We remove the following fields, because they are created per layer
        conv_like_layer_potential_args.pop('in_channels', None)
        conv_like_layer_potential_args.pop('out_channels', None)
        conv_like_layer_potential_args.pop('m_groups', None)
        conv_like_layer_potential_args.pop('w_groups', None)

        # All args of the Layer with default values
        conv_like_layer_default_kwargs = \
            _dict_conv_like_layers[self._fields_to_log['conv_like_type']].constructor_default_values(
                only_not_none=True
            )

        # Hierarchically fill the potential arguments of the Layer: "forced", present here in kwargs, or default
        for key in conv_like_layer_potential_args:
            if key in basic_conv_like_layer_kwargs:
                conv_like_layer_kwargs[key] = copy.deepcopy(basic_conv_like_layer_kwargs[key])
            elif key in kwargs:
                conv_like_layer_kwargs[key] = kwargs.pop(key, None)  # This time we remove them from 'kwargs'
            elif key in conv_like_layer_default_kwargs:
                conv_like_layer_kwargs[key] = copy.deepcopy(conv_like_layer_default_kwargs[key])
            pass
        pass

        # But still there will be "kwargs" left to be used
        for key in kwargs:
            conv_like_layer_kwargs[key] = kwargs[key]
        pass

        # Add the parameters of 'conv_like_layer_kwargs' to the 'self._fields_to_log'!
        for key in conv_like_layer_kwargs:
            if key not in self._fields_to_log:
                self._fields_to_log[key] = copy.deepcopy(conv_like_layer_kwargs[key])
            pass
        pass

        ######################################################
        ### FORCE ALL FIELDS IN "self._fields_to_log" TO BE HASHABLE... by transforming to str if necessary
        ######################################################
        for key in self._fields_to_log:
            if not isinstance(self._fields_to_log[key], typing.Hashable):
                self._fields_to_log[key] = str(self._fields_to_log[key])
            pass
        pass

        ######################################################
        ### POPULATE THE NETWORK
        ######################################################

        # Create backbone from convolutional blocks
        backbone_blocks = []

        # Exploring sequentially blocks of (n*conv + maxpool) layers...
        # Each block will be a torch.nn.Sequential object

        n_out_channels_block_i_minus_1 = self._extra_state_dict['constructor_kwargs']['in_channels']
        out_size_block_i_minus_1 = self._extra_state_dict['constructor_kwargs']['in_size']
        #
        for i, n_conv_layers_block_i in enumerate(
                self._extra_state_dict['constructor_kwargs']['conv_block_specification']
        ):  # i-th block
            #
            ### BLOCK (i): a SEQUENTIAL
            # sequential_block_i = None
            list_conv_like_block_i = []

            ### CONV-LIKE LAYERS of BLOCK (i)

            # Number of channels in the current block
            n_channels_block_i = min(base_channels * 2 ** (i), base_channels * 2 ** 3)
            #
            ### CONV-LIKE LAYERS (j) OF THE CURRENT BLOCK (i)
            for j in range(n_conv_layers_block_i):  # j-th conv layer of the i-th block
                #
                # Use the type of convolutional layer according to the indications in If the current layer is the first one of the first block, always use the 'conv_like_type'. \
                # Otherwise, use the 'conv_like_type' if 'conv_like_type_position' is True.
                conv_like_layer_ij_type = 'sm'
                if ((self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'everywhere')) or \
                        ((self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'first') and \
                         (i == 0 and j == 0)) or \
                        ((self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'last') and \
                         (i == len(self._extra_state_dict['constructor_kwargs']['conv_block_specification']) - 1 and \
                          j == n_conv_layers_block_i - 1)):
                    conv_like_layer_ij_type = self._extra_state_dict['constructor_kwargs']['conv_like_type']
                pass
                #
                # In- and out- number of channels and image size
                # Only the first conv layer of each block changes the number of channels. And no conv-like layer \
                # changes the spatial size (that is the responsibility of the maxpool layers).
                #
                n_in_channels_ij = n_out_channels_block_i_minus_1 if j == 0 else n_channels_block_i
                n_out_channels_ij = n_channels_block_i
                #
                in_size_ij = out_size_block_i_minus_1
                out_size_ij = out_size_block_i_minus_1

                # Conv-like layer: extra params to be added regarding 'w_kernel_size' and 'w_groups' if INRF or ibnn_internal
                #
                kwargs_layer_ij = copy.deepcopy(sm_kwargs) if conv_like_layer_ij_type == 'sm' \
                    else copy.deepcopy(conv_like_layer_kwargs)
                #
                # No need for resolving before creating the layer anymore: the layer will resolve it...
                # if we provide the 'in_size' for this layer
                kwargs_layer_ij['in_size'] = in_size_ij
                # kwargs_layer_ij['m_kernel_size'] = resolve_kernel_size_for_im_size(
                #     kernel_size=self._extra_state_dict['constructor_kwargs']['m_kernel_size'],
                #     im_size=in_size_ij,
                #     make_odd=True
                # )
                kwargs_layer_ij['m_groups'] = n_out_channels_ij \
                    if self._extra_state_dict['constructor_kwargs']['m_independent_channels'] else 1
                # kwargs_layer_ij['m_padding_mode'] = conv_like_layer_default_kwargs['m_padding_mode']
                #
                if conv_like_layer_ij_type in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
                    # If the stored 'self._extra_state_dict['w_kernel_size']' for the whole network has floats,
                    # then it is relative to the size of the input image. If it is ints, it is absolute.
                    # No need for resolving before creating the layer anymore: the layer will resolve it...
                    # if we provide the 'in_size' for this layer
                    # kwargs_layer_ij['w_kernel_size'] = resolve_kernel_size_for_im_size(
                    #     kernel_size=self._extra_state_dict['constructor_kwargs']['w_kernel_size'],
                    #     im_size=in_size_ij,
                    #     make_odd=True
                    # )
                    kwargs_layer_ij['w_groups'] = n_out_channels_ij \
                        if self._extra_state_dict['constructor_kwargs']['w_independent_channels'] else 1
                    # kwargs_layer_ij['w_padding_mode'] = conv_like_layer_default_kwargs['w_padding_mode']
                pass
                #
                conv_like_layer_ij = _dict_conv_like_layers[conv_like_layer_ij_type](
                    in_channels=n_in_channels_ij, out_channels=n_out_channels_ij,
                    **kwargs_layer_ij
                )
                #
                # Append the conv-like layer to the list of the current block, including name
                list_conv_like_block_i.append((f'conv_like_layer_{j}', conv_like_layer_ij))

                # Add the batch normalization layer if required, as a separate "post-" layer
                conv_like_layer_ij_post = None
                if batch_normalization:
                    conv_like_layer_ij_post = nn.BatchNorm2d(
                        num_features=n_out_channels_ij, affine=True, track_running_stats=True
                    )
                else:
                    conv_like_layer_ij_post = nn.Identity()
                pass
                list_conv_like_block_i.append((f'batch_norm_layer_{j}', conv_like_layer_ij_post))
                #
            pass
            #
            ### MAXPOOL LAYER OF THE CURRENT BLOCK (i):
            ######################################### REMOVED BLOCK #########################################
            # # ONLY IF IT IS NOT THE LAST BLOCK!!! And update the spatial size for the next block
            # if i < len(self._extra_state_dict['conv_block_specification']) - 1:
            #     # The maxpool layer halves the spatial size of the input, and does not change the number of channels.
            #     maxpool_layer_i = nn.MaxPool2d(kernel_size=2)
            #     # Append the conv-like layer to the list of the current block, including name
            #     list_conv_like_block_i.append((f'maxpool', maxpool_layer_i))
            #     # Update the image size for the next block
            #     out_size_block_i_minus_1 = tuple([elem // 2 for elem in out_size_block_i_minus_1])
            # else:
            #     pass
            # pass
            ######################################### ADDED BLOCK #########################################
            # The maxpool layer halves the spatial size of the input, and does not change the number of channels.
            maxpool_layer_i = nn.MaxPool2d(
                kernel_size=self._extra_state_dict['constructor_kwargs']['maxpool_reduction']
            )
            # Append the conv-like layer to the list of the current block, including name
            list_conv_like_block_i.append((f'maxpool_layer', maxpool_layer_i))
            # Update the image size for the next block
            out_size_block_i_minus_1 = tuple(
                [elem // self._extra_state_dict['constructor_kwargs']['maxpool_reduction']
                 for elem in out_size_block_i_minus_1]
            )
            ###############################################################################################
            n_out_channels_block_i_minus_1 = n_channels_block_i

            # Pack the block using Sequential
            sequential_block_i = nn.Sequential(OrderedDict(list_conv_like_block_i))

            # Append the block to the list of blocks
            backbone_blocks.append((f'conv_like_block_{i}', sequential_block_i))
            #
        pass  # end of the blocks

        # Add final adaptive pooling to backbone
        backbone_blocks.append(('adaptive_pool', nn.AdaptiveAvgPool2d((7, 7))))

        # Create the backbone as Sequential
        backbone_module = nn.Sequential(OrderedDict(backbone_blocks))

        # Create head with classifier parts
        head_blocks = []

        # Flatten layer
        head_blocks.append(('flatten', nn.Flatten()))

        ### FULLY CONNECTED LAYERS
        # There will be 'fc_num_layers' fully connected layers:
        #   1st) takes an input dependent on the size of the images and \
        #       the previous blocks, and has 'fc_num_units_intermediate_layers' output units; \
        #   intermediate) have 'fc_num_units_intermediate_layers' input and 'fc_num_units_intermediate_layers' output
        #   last) has 'fc_num_units_intermediate_layers' input and 'out_classes' output

        for i in range(fc_num_layers):
            ### BLOCK (i):
            # a SEQUENTIAL of FC + ReLu + BatchNorm1d, unless it is the last FC block,
            # wherein the ReLu + BatchNorm1d are omitted
            list_fc_block_i = []

            # In- and out- sizes per FC layer
            if i == 0:
                in_features = 7 * 7 * n_out_channels_block_i_minus_1
            else:
                in_features = np.prod(out_size_block_i_minus_1) * n_out_channels_block_i_minus_1 if i == 0 \
                    else fc_num_units_intermediate_layers
            out_features = fc_num_units_intermediate_layers if i < fc_num_layers - 1 else out_classes
            #
            # Create the layer and append it to the list of layers
            fc_layer_i = nn.Linear(in_features, out_features)
            list_fc_block_i.append((f'fc_layer', fc_layer_i))
            #
            if i != (fc_num_layers - 1):  # No ReLU or BatchNorm1d for the very last FC layer
                # Create also a ReLU for it and append it:
                relu_layer_i = nn.ReLU()
                list_fc_block_i.append((f"relu_layer", relu_layer_i))
                #
                # Add the batch normalization layer if required, as a separate "post-" layer...
                ######################################### REMOVED BLOCK #######################################
                # # Do not include it  for the last one
                # fc_layer_i_post = None
                # if batch_normalization and (i < fc_num_layers - 1):
                #     fc_layer_i_post = nn.BatchNorm1d(
                #         num_features=out_features, affine=True, track_running_stats=True
                #     )
                # else:
                #     fc_layer_i_post = nn.Identity()
                # pass
                ######################################### ADDED BLOCK #########################################
                # Include it for all layers
                fc_layer_i_post = nn.Dropout(p=fc_dropout)
                ###############################################################################################
                list_fc_block_i.append((f'dropout', fc_layer_i_post))
            pass
            #
            # Pack the block using Sequential
            sequential_block_i = nn.Sequential(OrderedDict(list_fc_block_i))
            #
            # Append the block to the list of blocks
            head_blocks.append((f'fc_block_{i}', sequential_block_i))
            #
        pass

        ### EXIT LAYER: SOFTMAX?
        # Create the exit layer, which can be a softmax layer or an identity layer, and append it to the list of layers
        exit_layer = None
        if softmax_output:
            exit_layer = nn.Softmax(dim=-1)
        else:
            exit_layer = nn.Identity()
        pass
        head_blocks.append(('exit', exit_layer))

        # Create the head as Sequential
        head_module = nn.Sequential(OrderedDict(head_blocks))

        # Create the full network by combining backbone and head
        self._nn = nn.Sequential(OrderedDict([
            ('backbone', backbone_module),
            ('head', head_module)
        ]))

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        It randomizes the trainable parameters of the network, using the specified distribution and gain, \
        by calling the static method :py:meth:`.random_initialization_subnetwork`.

        Parameters
        ----------
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        VGGxClassifier.random_initialization_subnetwork(self._nn,
                                                        distribution=distribution, gain=gain, additive=additive)

    @staticmethod
    def random_initialization_subnetwork(subnetwork,
                                         distribution='normal', gain=1e-3, additive=True):
        """
        Function that iteratively looks for the method ``.random_initialization(...)`` of the children of the provided \
        subnetwork, if it is a :py:class:`torch.nn.Sequential` object, and applies either it or an analogous \
        randomization of its trainable parameters.

        Parameters
        ----------
        subnetwork : torch.nn.Module or torch.nn.Sequential
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """
        if isinstance(subnetwork, nn.Sequential):
            for layer in subnetwork:
                VGGxClassifier.random_initialization_subnetwork(layer,
                                                                distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer)):
            subnetwork.random_initialization(distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, nn.Module):
            for param_tensor in subnetwork.parameters():
                current_average = param_tensor.flatten().mean().item() if additive else 0.0
                if distribution == 'normal':
                    nn.init.normal_(param_tensor,
                                    mean=current_average, std=0.5 * gain)
                elif distribution == 'uniform':
                    nn.init.uniform_(param_tensor,
                                     a=current_average - 0.5 * gain, b=current_average + 0.5 * gain)
                pass
            pass
        pass

    # def forward(self, x):
    #     """
    #     Forward pass of the network which, since the network is a Sequential object, is simply the forward pass \
    #     of the Sequential object.
    #
    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #     """
    #     result_forward = None
    #     if self.training and self._extra_state_dict['batch_normalization'] and x.ndim > 3 and x.size(-4) == 1:
    #         # print((f"\t\tWARNING: Input batch has only 1 im, so BatchNorm1d would cause an error in training: " +
    #         #        f"EVAL used instead!!!"))
    #         self._nn.eval()
    #         result_forward = self._nn(x)
    #         self._nn.train()
    #     else:
    #         result_forward = self._nn(x)
    #     pass
    #
    #     return result_forward

    # @property
    # def dict_fields_to_log(self):
    #     """
    #     Identical, for now, to :py:meth:`.get_extra_state`.
    #
    #     Returns
    #     -------
    #     dict
    #     """
    #
    #     return self.get_extra_state()

    # def get_extra_state(self):
    #     """
    #     Store all the attributes of the network that are not part of the trainable parameters of the network.\
    #     :py:class:`torch.nn.parameter.Parameter` of the network, since those will be stored \
    #     by :py:class:`torch.nn.Module` itself.
    #
    #     The function returns a dictionary with the arguments used to construct this object \
    #     (see the constructor :py:class:`VGGxLayer`).
    #
    #     See also :py:meth:`.set_extra_state` for the counterpart of this method.
    #
    #     Returns
    #     -------
    #     dict
    #     """
    #
    #     return copy.deepcopy(self._extra_state_dict)

    # def set_extra_state(self, extra_state_dict):
    #     """
    #     Although the methods *set_extra_state* usual load stored parameters, we will mostly use it to make sure that \
    #     the parameters stored in ``extra_state_dict``, and corresponding to a previous network, are compatible with \
    #     the parameters of this current network.
    #     We will allow, though, if required, the change of type of the convolutional layers of the network, ruled by \
    #     both the 'self._conv_like_type' and 'self._conv_like_type_position' parameters, simply giving a warning.
    #     """
    #
    #     ##############################################################################
    #     # REMEMBER: the 'self._extra_state_dict' is a dict that contains:
    #     #   - ``self._extra_state_dict['net']`` (:py:class:`str`): the type of network used
    #     #   - ``self._extra_state_dict['batch_normalization']`` (:py:class:`bool`): whether batch normalization is used
    #     #   - ``self._extra_state_dict['softmax_output']`` (:py:class:`bool`): whether batch normalization is used
    #     #   - ``self._extra_state_dict['constructor_kwargs']`` (:py:class:`dict`):
    #     #       the kwargs that would directly construct an object identical (but for the trainable parameters)
    #     #       to the current one.
    #     ##############################################################################
    #
    #     # Check the keys 'net', 'batch_normalization', and 'softmax_output'
    #     for key in extra_state_dict.keys():
    #         if key in ['net', 'batch_normalization', 'softmax_output']:
    #             if self._extra_state_dict[key] != extra_state_dict[key]:
    #                 raise Exception(
    #                     f"Provided extra_state_dict['{key}'] does not match current self._extra_state_dict['{key}']: " +
    #                     f"{extra_state_dict[key]} != {self._extra_state_dict[key]}; match of these fields is required."
    #                 )
    #             pass
    #         pass
    #     pass
    #
    #     # Check 'constructor_kwargs': does it exist, does it fit?
    #     if 'constructor_kwargs' not in extra_state_dict.keys():
    #         raise Exception(
    #             f"Provided extra_state_dict['constructor_kwargs'] does not exist, no checks can be performed!"
    #         )
    #     else:
    #         for key in extra_state_dict['constructor_kwargs'].keys():
    #             if key not in self._extra_state_dict['constructor_kwargs'].keys():
    #                 print(f"Warning: key '{key}' of the argument 'extra_state_dict['constructor_kwargs']'" +
    #                       f"not present in the current network.")
    #             elif key in ['conv_like_type', 'conv_like_type_position']:
    #                 # Warning only
    #                 if self._extra_state_dict['constructor_kwargs'][key] != extra_state_dict['constructor_kwargs'][key]:
    #                     print((f"Warning: '{key}' was {self._extra_state_dict['constructor_kwargs'][key]} in the loaded dict; " +
    #                            f"the current network has, however, {extra_state_dict['constructor_kwargs'][key]}; " +
    #                            f"the value will not be transferred into the receiving network."))
    #                 pass
    #             elif key in ['m_padding_mode', 'w_padding_mode']:
    #                 # Warning and transfer
    #                 if self._extra_state_dict['constructor_kwargs'][key] != extra_state_dict['constructor_kwargs'][key]:
    #                     print((f"Warning: '{key}' was {self._extra_state_dict['constructor_kwargs'][key]} in the loaded dict; " +
    #                            f"the current network has, however, {extra_state_dict['constructor_kwargs'][key]}; " +
    #                            f"the value will not be transferred into the receiving network."))
    #                     self._extra_state_dict['constructor_kwargs'][key] = \
    #                         copy.deepcopy(extra_state_dict['constructor_kwargs'][key])
    #                 pass
    #             else:
    #                 assert self._extra_state_dict['constructor_kwargs'][key] == extra_state_dict['constructor_kwargs'][key], \
    #                     (f"Provided extra_state_dict['constructor_kwargs']['{key}'] " +
    #                      f"does not match current self._extra_state_dict['constructor_kwargs']['{key}']: " +
    #                      f"{extra_state_dict['constructor_kwargs'][key]} != {self._extra_state_dict['constructor_kwargs'][key]}.")
    #             pass
    #         pass
    #     pass




#########################################################################################
#########################################################################################
# CLASSES: AlexNetClassifierLoose model
#########################################################################################
#########################################################################################


class AlexNetClassifierLoose(MultilayerClassifier):
    """
    Customizable classifier inspired by the family of neural network architectures AlexNet [Krizhevsky2012]_ or, \
    better, on [Krizhevsky2014]_.
    The present class :py:class:`.AlexNetClassifier` allows the creation of a classifier containing the number and \
    type of convolutional-like layers indicated in the constructor, as well as a number of final fully-connected layer \
    also specified therein.
    AlexNet architecture according to [Krizhevsky2014]_ (and corresponding too to the implementation included in \
    Pytorch and more precisely in :py:mod:`torchvision.models.alexnet`) \
    comprises five convolutional layers followed by three fully connected \
    layers. The first two convolutional layers are followed by local response normalization (LRN) and max-pooling \
    layers, while the fifth convolutional layer is followed by max-pooling only. The first convolutional layer uses \
    11x11 filters with stride 4, the second layer uses 5x5 filters, and the last three layers use 3x3 filters. \
    The convolutional layers have progressively more filters: 96 in the first layer, 256 in the second, and 384, 384, \
    and 256 in the last three layers respectively. After the convolutional layers, the original \
    network [Krizhevsky2012]_ has 3 fully connected layers, where their respective (flattened) \
    input and output sizes are:

    $$\\textrm{length} \\rightarrow \\boxed{FC1} \\rightarrow 4096 \
    \\rightarrow \\boxed{FC2} \\rightarrow 4096 \\rightarrow \\boxed{FC3} \
    \\rightarrow 1000$$

    with 1000 being the number of classes considered by [Krizhevsky2012]_ (i.e. from the ImageNet dataset), \
    followed by a softmax output.

    The present class :py:class:`.AlexNetClassifier` allows the definition of a network following \
    the general principles of the original AlexNet structure described above but providing the following \
    potential variations:

    - The option to replace CNN layers with other convolutional-like layers, specifically INRF or ibnn_internal layers; \
      this option includes replacing all CNN layers or just the first or last one.
    - The constructor allows the join specification of the number of convolutional blocks and the number of \
      convolutional-like layers present in each block through the argument `conv_block_specification`, which is a list \
      of $B$ integers where each integer indicates the number of layers within the block: max pooling is performed \
      after each block. (Let $L$ be the total amount of convolutional-like layers, that is, the sum of the $B$ \
      elements of `conv_block_specification`.)
    - The number of output channels resulting from each convolution-like layer is not specified per block but \
      per layer, that is, `channels_per_conv_layer` is a list with as many entries as $L$ (the sum of all the elements of
      `conv_block_specification`): each element of `channels_per_conv_layer` indicates \
      the number of channels of all the convolutions of each block.
    - It allows kernels (m) of different sizes than the original sizes (11x11, 5x5, 3x3) assumed \
      by the original AlexNet for the convolutional layers. \
      The size of the kernels is not specified per block but \
      per layer, that is, `m_kernel_size_per_conv_layer` is a list with as many entries as $L$ (the sum of all the elements of
      `conv_block_specification`): each element of `channels_per_conv_layer` indicates \
      the number of channels of all the convolutions of each block.
    - It allows max-pooling with a custom maxpool_reduction factor at the end of each block.
    - It allows the specification of the number of intermediate units and the number of FC layers.
    - Unlike the original post [1], we leave the softmax output as an option.
    - The class allows batch normalization layers, which, if applied, are applied after each convolutional and FC layer.

    References
    ----------
    .. [Krizhevsky2012]  *Krizhevsky, A., Sutskever, I., & Hinton, G. E.*. ImageNet Classification with Deep \
        Convolutional Neural Networks, 2012 (in NIPS 2012). \
        `link <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_
    .. [Krizhevsky2014]  *Krizhevsky, A.*. One weird trick for parallelizing convolutional neural networks, 2014 \
        (in CoRR). \
        `link <https://https://arxiv.org/abs/1404.5997>`_

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, with ``w_kernel_size`` = ``()`` \
    (empty tuple), **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``)
    conv_block_specification : tuple|list[int]
        List of integers whose length indicates the number of convolutional blocks of the network, and whose \
        $n$-th entry, $> 0$, indicates the number of convolutional sub-layer in said block.
        Default: ``(1, 1, 3)``
    channels_per_conv_layer : tuple|list[int], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements), \
        wherein each entry indicates the number \
        of output channels of each convolutional-like layer of said block.
        Default: ``(64, 192, 384, 256, 256)`` (as in the original AlexNet, as interpreted by its Pytorch \
        implementation `torchvision.models.alexnet <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet>`_)
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements) \
        and as many as `channels_per_conv_layer`. Each element of the list/tuple follows the convention of the \
        convolutional-like layers (:py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer`) for their argument `m_kernel_size`.
        Default: ``(11, 5, 3, 3, 3)`` (as in the original Alexnet)
    phi_activation_per_conv_layer : tuple|list[str]
        Activation function used in the convolutional-like layers
    batch_normalization_per_conv_layer : tuple|list[bool]
        Whether batch normalization is to be used, indicated for the position after each convolutional-like layer
    maxpool_reduction_per_conv_block : tuple|list[int], optional
        Default: ``[2, 2, 2]`` (no reduction in any block)
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs). Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image. The argument is set with a default ``None``, but the \
        constructor **raises an Exception if no value, or** ``None``, **is provided** and the type of indicated \
        ``conv_like_type`` is among ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs). Default: ``True``
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$.
        Default: ``3`` (as in the original AlexNet)
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued.
        Default: ``4096`` (default for AlexNet according to its Pytorch implementation)
    fc_batch_normalization : bool, optional
        Whether batch normalization is to be used in the intermediate fully connected layers
    fc_dropout : float, optional
        fc_dropout probability for the fully connected layers.
        Default: ``0.5`` (default for AlexNet according to its Pytorch implementation)
    penciled_decision : bool, optional
        If ``True``, the (first, if more than one) FC layer is parallel to the vector (1,...,1).
        Default: ``False``
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer.
        Default: ``False``
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: ``True``
    device : str, optional
        Value among ``'cuda'`` and ``'cpu'``, device where the classifier is created. Error if ``'cuda'`` but \
        no GPU is available.
        Default: ``'cuda'`` if GPU available, ``'cpu'`` otherwise
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_like_type, conv_like_type_position,
                 conv_block_specification=(1, 1, 3), channels_per_conv_layer=(64, 192, 384, 256, 256),
                 m_kernel_size_per_conv_layer=(11, 5, 3, 3, 3),
                 phi_activation_per_conv_layer=None, batch_normalization_per_conv_layer=None,
                 maxpool_reduction_per_conv_block=None,
                 m_independent_channels=False, w_kernel_size=None, w_independent_channels=True,
                 fc_num_layers=3, fc_num_units_intermediate_layers=4096, fc_batch_normalization=True,
                 fc_dropout=0.5, penciled_decision=False, softmax_output=False, prenormalization=False,
                 device=None,
                 **kwargs):

        # Default values for 'phi_activation_per_conv_layer', 'maxpool_reduction_per_conv_block'
        # and 'batch_normalization_per_conv_layer':
        if isinstance(conv_block_specification, (list, tuple)) and all([isinstance(x, int) and x > 0 for x in conv_block_specification]):
            num_blocks = len(conv_block_specification)
            num_conv_layers = int(np.array(conv_block_specification).sum())
            if maxpool_reduction_per_conv_block is None:
                maxpool_reduction_per_conv_block = [3]*num_blocks
            if phi_activation_per_conv_layer is None:
                phi_activation_per_conv_layer = ['relu']*num_conv_layers
            if batch_normalization_per_conv_layer is None:
                batch_normalization_per_conv_layer = [True]*num_conv_layers
            pass
        else:
            raise Exception(f"Invalid conv_block_specification: {conv_block_specification}: " +
                            f"list/tuple of positive integers expected, {conv_block_specification} found!")
        pass

        ######################################################
        # RUN THE CONSTRUCTOR OF THE PARENT CLASS MultilayerClassifier SO:
        # - THE STRUCTURES OF 'nn.Module' ARE CREATED
        # - THE SPECIFIC CHECKS OF 'MultilayerClassifier' ARE RUN
        ######################################################

        MultilayerClassifier.__init__(
            self,
            in_size=in_size, in_channels=in_channels, out_classes=out_classes,
            conv_like_type=conv_like_type, conv_like_type_position=conv_like_type_position,
            conv_block_specification=conv_block_specification, channels_per_conv_layer=channels_per_conv_layer,
            m_kernel_size_per_conv_layer=m_kernel_size_per_conv_layer,
            phi_activation_per_conv_layer=phi_activation_per_conv_layer,
            batch_normalization_per_conv_layer=batch_normalization_per_conv_layer,
            maxpool_reduction_per_conv_block=maxpool_reduction_per_conv_block,
            m_independent_channels=m_independent_channels,
            w_kernel_size=w_kernel_size, w_independent_channels=w_independent_channels,
            fc_num_layers=fc_num_layers, fc_num_units_intermediate_layers=fc_num_units_intermediate_layers,
            fc_batch_normalization=fc_batch_normalization, fc_dropout=fc_dropout,
            penciled_decision=penciled_decision, softmax_output=softmax_output, prenormalization=prenormalization,
            device=None,
            **kwargs
        )

        ######################################################
        # INITIALIZE THE STRUCTURES REQUIRED BY 'ClassifierBaseModel' WHICH WILL BE FILLED IN THE CONSTRUCTOR
        ######################################################

        prenormalization_module = None
        backbone_module = None
        head_module = None

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: BACKBONE
        ######################################################
        ######################################################

        # Make sure that there are no repeated args and kwargs in the function by creating a dict that we peel out
        kwargs_for_create_standard_backbone_module = copy.deepcopy(self._extra_state_dict['constructor_kwargs'])
        # We remove first arguments that we know we will not use
        for key in ['out_classes', 'fc_num_layers', 'fc_num_units_intermediate_layers',
                    'fc_batch_normalization', 'fc_dropout', 'penciled_decision', 'softmax_output',
                    'prenormalization']:
            if key in kwargs_for_create_standard_backbone_module.keys():
                kwargs_for_create_standard_backbone_module.pop(key)
            pass
        pass
        # And call the function
        backbone_module, backbone_out_size, backbone_out_channels = _create_standard_backbone_module(
            in_size=kwargs_for_create_standard_backbone_module.pop('in_size', None),
            in_channels=kwargs_for_create_standard_backbone_module.pop('in_channels', None),
            conv_like_type=kwargs_for_create_standard_backbone_module.pop('conv_like_type', None),
            conv_like_type_position=kwargs_for_create_standard_backbone_module.pop('conv_like_type_position', None),
            conv_block_specification=kwargs_for_create_standard_backbone_module.pop('conv_block_specification', None),
            channels_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('channels_per_conv_layer', None),
            phi_activation_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('phi_activation_per_conv_layer', None),
            m_kernel_size_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('m_kernel_size_per_conv_layer', None),
            batch_normalization_per_layer=kwargs_for_create_standard_backbone_module.pop('batch_normalization_per_conv_layer', None),
            maxpool_reduction_per_conv_block=kwargs_for_create_standard_backbone_module.pop('maxpool_reduction_per_conv_block', None),
            m_independent_channels=kwargs_for_create_standard_backbone_module.pop('m_independent_channels', None),
            w_kernel_size=kwargs_for_create_standard_backbone_module.pop('w_kernel_size', None),
            w_independent_channels=kwargs_for_create_standard_backbone_module.pop('w_independent_channels', None),
            **kwargs_for_create_standard_backbone_module
        )
        # Add in the backbone AdaptiveAvgPool2d((6, 6)) to follow the original AlexNet structure
        # backbone_out_size = np.prod(backbone_out_size) * 6 * 6
        # backbone_out_channels = backbone_out_channels
        # backbone_module.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((6, 6)))

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: HEAD
        ######################################################
        ######################################################

        num_features_after_flatten = \
            int(backbone_out_channels * np.prod(backbone_out_size))

        head_module = _create_standard_head_module(
            num_features_in=num_features_after_flatten,
            num_features_out=self._extra_state_dict['out_classes'],
            num_layers=self._extra_state_dict['fc_num_layers'],
            num_features_intermediate_layers=self._extra_state_dict['fc_num_units_intermediate_layers'],
            batch_normalization=self._extra_state_dict['fc_batch_normalization'],
            dropout=self._extra_state_dict['fc_dropout'],
            penciled_decision=self._extra_state_dict['penciled_decision'],
            softmax_output=self._extra_state_dict['softmax_output']
        )

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK:
        ### CREATE sef.prenormalization and integrate everything into self._nn
        ### AND MOVE IT TO THE DESIRED DEVICE
        ######################################################
        ######################################################

        # Prenormalization
        prenormalization_module = nn.Identity() if not self._extra_state_dict['prenormalization'] else \
            nn.BatchNorm2d(num_features=in_channels, affine=False, track_running_stats=True)

        # Create the full network by combining backbone and head
        self._nn = nn.Sequential(OrderedDict([
            ('prenormalization', prenormalization_module),
            ('backbone', backbone_module),
            ('head', head_module)
        ]))

        # Computation device
        computation_device = None
        if device is None:  # default
            computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu'], \
                f"Invalid device: 'cuda' or 'cpu' expected, {device} found!."
            if device == 'cuda' and not torch.cuda.is_available():
                raise Exception(f"Invalid device: 'cuda' selected but no GPU available.")
            computation_device = device
        pass
        self.to_device(computation_device)

        ######################################################
        ######################################################

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        It randomizes the trainable parameters of the network, using the specified distribution and gain, \
        by calling the static method :py:meth:`.random_initialization_subnetwork`.

        Parameters
        ----------
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        AlexNetClassifierLoose.random_initialization_subnetwork(self._nn,
                                                           distribution=distribution, gain=gain, additive=additive)

    @staticmethod
    def random_initialization_subnetwork(subnetwork,
                                         distribution='normal', gain=1e-3, additive=True):
        """
        Function that iteratively looks for the method ``.random_initialization(...)`` of the children of the provided \
        subnetwork, if it is a :py:class:`torch.nn.Sequential` object, and applies either it or an analogous \
        randomization of its trainable parameters.

        Parameters
        ----------
        subnetwork : torch.nn.Module or torch.nn.Sequential
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """
        if isinstance(subnetwork, nn.Sequential):
            for layer in subnetwork:
                AlexNetClassifierLoose.random_initialization_subnetwork(layer,
                                                                   distribution=distribution, gain=gain,
                                                                   additive=additive)
        elif isinstance(subnetwork, (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer)):
            subnetwork.random_initialization(distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, nn.Module):
            for param_tensor in subnetwork.parameters():
                current_average = param_tensor.flatten().mean().item() if additive else 0.0
                if distribution == 'normal':
                    nn.init.normal_(param_tensor,
                                    mean=current_average, std=0.5 * gain)
                elif distribution == 'uniform':
                    nn.init.uniform_(param_tensor,
                                     a=current_average - 0.5 * gain, b=current_average + 0.5 * gain)
                pass
            pass
        pass



#########################################################################################
#########################################################################################
# CLASSES: AlexNetClassifierLoose model
#########################################################################################
#########################################################################################


class AlexNetClassifier(MultilayerClassifier):
    """
    *Strict* AlexNet according to [Krizhevsky2014]_ (which mimics PyTorch architecture).
    The present class :py:class:`.AlexNetClassifier` allows the creation of a classifier containing the number and \
    type of convolutional-like layers indicated in the constructor, as well as a number of final fully-connected layer \
    also specified therein.
    AlexNet architecture according to [Krizhevsky2014]_ (and corresponding too to the implementation included in \
    Pytorch and more precisely in :py:mod:`torchvision.models.alexnet`) \
    comprises five convolutional layers followed by three fully connected \
    layers. The first two convolutional layers are followed by local response normalization (LRN) and max-pooling \
    layers, while the fifth convolutional layer is followed by max-pooling only. The first convolutional layer uses \
    11x11 filters with stride 4, the second layer uses 5x5 filters, and the last three layers use 3x3 filters. \
    The convolutional layers have progressively more filters: 96 in the first layer, 256 in the second, and 384, 384, \
    and 256 in the last three layers respectively. After the convolutional layers, the original \
    network [Krizhevsky2012]_ has 3 fully connected layers, where their respective (flattened) \
    input and output sizes are:

    $$\\textrm{length} \\rightarrow \\boxed{FC1} \\rightarrow 4096 \
    \\rightarrow \\boxed{FC2} \\rightarrow 4096 \\rightarrow \\boxed{FC3} \
    \\rightarrow 1000$$

    with 1000 being the number of classes considered by [Krizhevsky2012]_ (i.e. from the ImageNet dataset), \
    followed by a softmax output.

    The present class :py:class:`.AlexNetClassifier` allows the definition of a network following \
    the general principles of the original AlexNet structure described above but providing the following \
    potential variations:

    - The option to replace CNN layers with other convolutional-like layers, specifically INRF or ibnn_internal layers; \
      this option includes replacing all CNN layers or just the first or last one.
    - The constructor allows the join specification of the number of convolutional blocks and the number of \
      convolutional-like layers present in each block through the argument `conv_block_specification`, which is a list \
      of $B$ integers where each integer indicates the number of layers within the block: max pooling is performed \
      after each block. (Let $L$ be the total amount of convolutional-like layers, that is, the sum of the $B$ \
      elements of `conv_block_specification`.)
    - The number of output channels resulting from each convolution-like layer is not specified per block but \
      per layer, that is, `channels_per_conv_layer` is a list with as many entries as $L$ (the sum of all the elements of
      `conv_block_specification`): each element of `channels_per_conv_layer` indicates \
      the number of channels of all the convolutions of each block.
    - It allows kernels (m) of different sizes than the original sizes (11x11, 5x5, 3x3) assumed \
      by the original AlexNet for the convolutional layers. \
      The size of the kernels is not specified per block but \
      per layer, that is, `m_kernel_size_per_conv_layer` is a list with as many entries as $L$ (the sum of all the elements of
      `conv_block_specification`): each element of `channels_per_conv_layer` indicates \
      the number of channels of all the convolutions of each block.
    - It allows max-pooling with a custom maxpool_reduction factor at the end of each block.
    - It allows the specification of the number of intermediate units and the number of FC layers.
    - Unlike the original post [1], we leave the softmax output as an option.
    - The class allows batch normalization layers, which, if applied, are applied after each convolutional and FC layer.

    References
    ----------
    .. [Krizhevsky2012]  *Krizhevsky, A., Sutskever, I., & Hinton, G. E.*. ImageNet Classification with Deep \
        Convolutional Neural Networks, 2012 (in NIPS 2012). \
        `link <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_
    .. [Krizhevsky2014]  *Krizhevsky, A.*. One weird trick for parallelizing convolutional neural networks, 2014 \
        (in CoRR). \
        `link <https://https://arxiv.org/abs/1404.5997>`_

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, with ``w_kernel_size`` = ``()`` \
    (empty tuple), **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``)
    conv_block_specification : tuple|list[int]
        List of integers whose length indicates the number of convolutional blocks of the network, and whose \
        $n$-th entry, $> 0$, indicates the number of convolutional sub-layer in said block.
        Default: ``(1, 1, 3)``
    channels_per_conv_layer : tuple|list[int], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements), \
        wherein each entry indicates the number \
        of output channels of each convolutional-like layer of said block.
        Default: ``(64, 192, 384, 256, 256)`` (as in the original AlexNet, as interpreted by its Pytorch \
        implementation `torchvision.models.alexnet <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet>`_)
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements) \
        and as many as `channels_per_conv_layer`. Each element of the list/tuple follows the convention of the \
        convolutional-like layers (:py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer`) for their argument `m_kernel_size`.
        Default: ``(11, 5, 3, 3, 3)`` (as in the original Alexnet)
    phi_activation_per_conv_layer : tuple|list[str]
        Activation function used in the convolutional-like layers
    batch_normalization_per_conv_layer : tuple|list[bool]
        Whether batch normalization is to be used, indicated for the position after each convolutional-like layer
    maxpool_reduction_per_conv_block : tuple|list[int], optional
        Default: ``[2, 2, 2]`` (no reduction in any block)
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs). Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image. The argument is set with a default ``None``, but the \
        constructor **raises an Exception if no value, or** ``None``, **is provided** and the type of indicated \
        ``conv_like_type`` is among ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs). Default: ``True``
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$.
        Default: ``3`` (as in the original AlexNet)
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued.
        Default: ``4096`` (default for AlexNet according to its Pytorch implementation)
    fc_batch_normalization : bool, optional
        Whether batch normalization is to be used in the intermediate fully connected layers
    fc_dropout : float, optional
        fc_dropout probability for the fully connected layers.
        Default: ``0.5`` (default for AlexNet according to its Pytorch implementation)
    penciled_decision : bool, optional
        If ``True``, the (first, if more than one) FC layer is parallel to the vector (1,...,1).
        Default: ``False``
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer.
        Default: ``False``
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: ``True``
    device : str, optional
        Value among ``'cuda'`` and ``'cpu'``, device where the classifier is created. Error if ``'cuda'`` but \
        no GPU is available.
        Default: ``'cuda'`` if GPU available, ``'cpu'`` otherwise
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_like_type, conv_like_type_position,
                 conv_block_specification=(1, 1, 3), channels_per_conv_layer=(64, 192, 384, 256, 256),
                 m_kernel_size_per_conv_layer=(11, 5, 3, 3, 3),
                 phi_activation_per_conv_layer=None, batch_normalization_per_conv_layer=None,
                 maxpool_reduction_per_conv_block=None,
                 m_independent_channels=False, w_kernel_size=None, w_independent_channels=True,
                 fc_num_layers=3, fc_num_units_intermediate_layers=4096, fc_batch_normalization=True,
                 fc_dropout=0.5, penciled_decision=False, softmax_output=False, prenormalization=False,
                 device=None,
                 **kwargs):

        # Default values for 'phi_activation_per_conv_layer', 'maxpool_reduction_per_conv_block'
        # and 'batch_normalization_per_conv_layer':
        if isinstance(conv_block_specification, (list, tuple)) and all([isinstance(x, int) and x > 0 for x in conv_block_specification]):
            num_blocks = len(conv_block_specification)
            num_conv_layers = int(np.array(conv_block_specification).sum())
            # if maxpool_reduction_per_conv_block is None:
            #     maxpool_reduction_per_conv_block = [2]*num_blocks
            if phi_activation_per_conv_layer is None:
                phi_activation_per_conv_layer = ['relu']*num_conv_layers
            elif isinstance(phi_activation_per_conv_layer, str):
                phi_activation_per_conv_layer = [phi_activation_per_conv_layer]*num_conv_layers
            pass
            #
            if batch_normalization_per_conv_layer is None:
                batch_normalization_per_conv_layer = [True]*num_conv_layers
            elif isinstance(batch_normalization_per_conv_layer, bool):
                batch_normalization_per_conv_layer = [batch_normalization_per_conv_layer]*num_conv_layers
            pass
        else:
            raise Exception(f"Invalid conv_block_specification: {conv_block_specification}: " +
                            f"list/tuple of positive integers expected, {conv_block_specification} found!")
        pass

        ######################################################
        # WE ASSESS THAT THE PROVIDED PARAMETERS ARE EXACTLY WHAT IS ALLOWED BY THE ALEXNET ARCHITECTURE (STRICT)
        ######################################################

        ###############################################
        # NOTE(1):  WE ARE IGNORING THE PADDING, WHICH THERE IS (A BIT, IN THE FIRST LAYER MAINLY),
        #           BECAUSE OUR ibnn_internal and ibnn DO NOT ALLOW FOR NUMERIC PADDING (ONLY 'valid', 'same'...).
        ###############################################
        # NOTE(2):  SINCE OUR LAYERS DO NOT ALLOW FOR "STRIDE", WE WILL "SIMULATE" THE STRIDE OF THE \
        #           FIRST LAYER OF ALEXNET USING, INSTEAD, A DILATION IN THE MAXPOOL2D IMMEDIATELY FOLLOWING.
        #           FROM PyTorch-AlexNet:
        #               ...
        #               nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #               nn.ReLU(inplace=True),
        #               nn.MaxPool2d(kernel_size=3, stride=2)
        #               ...
        #           WE WILL DO, INSTEAD (, THE ANALOGOUS TO):
        #               ...
        #               nn.Conv2d(3, 64, kernel_size=11, STRIDE=1, padding=2),
        #               nn.ReLU(inplace=True),
        #               nn.MaxPool2d(kernel_size=3, STRIDE=2*4, DILATION=4)
        #               ...
        ###############################################

        kernel_maxpool_reduction_per_conv_block = (3, 3, 3)
        stride_maxpool_reduction_per_conv_block = (2*4, 2, 2)
        dilation_maxpool_reduction_per_conv_block = (4, 1, 1)

        dict_admissible_arguments = {}
        dict_received_arguments = {}
        #####
        dict_admissible_arguments['in_size'] = (224, 224)
        dict_received_arguments['in_size'] = in_size
        #
        dict_admissible_arguments['in_channels'] = 3
        dict_received_arguments['in_channels'] = in_channels
        #
        dict_admissible_arguments['conv_block_specification'] = (1, 1, 3)
        dict_received_arguments['conv_block_specification'] = tuple(conv_block_specification)
        #
        dict_admissible_arguments['maxpool_reduction_per_conv_block'] = stride_maxpool_reduction_per_conv_block
        dict_received_arguments['maxpool_reduction_per_conv_block'] = \
            stride_maxpool_reduction_per_conv_block if maxpool_reduction_per_conv_block is None \
            else maxpool_reduction_per_conv_block
        #
        dict_admissible_arguments['channels_per_conv_layer'] = (64, 192, 384, 256, 256)
        dict_received_arguments['channels_per_conv_layer'] = tuple(channels_per_conv_layer)
        #
        dict_admissible_arguments['m_kernel_size_per_conv_layer'] = ((11, 11), (5, 5), (3, 3), (3, 3), (3, 3))
        dict_received_arguments['m_kernel_size_per_conv_layer'] = tuple(
            [kernel_size_check_and_reformat_into_tuple(elem) for elem in m_kernel_size_per_conv_layer]
        )
        #
        dict_admissible_arguments['phi_activation_per_conv_layer'] = ('relu',)*num_conv_layers
        dict_received_arguments['phi_activation_per_conv_layer'] = tuple(phi_activation_per_conv_layer)
        #
        dict_admissible_arguments['fc_num_layers'] = 3
        dict_received_arguments['fc_num_layers'] = fc_num_layers
        #
        dict_admissible_arguments['fc_num_units_intermediate_layers'] = 4096
        dict_received_arguments['fc_num_units_intermediate_layers'] = fc_num_units_intermediate_layers
        #
        dict_admissible_arguments['fc_num_layers'] = 3
        dict_received_arguments['fc_num_layers'] = fc_num_layers

        # Run the checks
        for key in dict_admissible_arguments:
            assert dict_received_arguments[key]==dict_admissible_arguments[key], \
                f"AlexNetClassifier (strict) only admits {dict_admissible_arguments[key]} as '{key}': " + \
                f"{dict_received_arguments[key]} found!"
        pass

        # And make eventually 'm_kernel_size_per_conv_layer' be 'stride_maxpool_reduction_per_conv_block' in any case
        maxpool_reduction_per_conv_block = stride_maxpool_reduction_per_conv_block

        ######################################################
        # RUN THE CONSTRUCTOR OF THE PARENT CLASS MultilayerClassifier SO:
        # - THE STRUCTURES OF 'nn.Module' ARE CREATED
        # - THE SPECIFIC CHECKS OF 'MultilayerClassifier' ARE RUN
        ######################################################

        MultilayerClassifier.__init__(
            self,
            in_size=in_size, in_channels=in_channels, out_classes=out_classes,
            conv_like_type=conv_like_type, conv_like_type_position=conv_like_type_position,
            conv_block_specification=conv_block_specification, channels_per_conv_layer=channels_per_conv_layer,
            m_kernel_size_per_conv_layer=m_kernel_size_per_conv_layer,
            phi_activation_per_conv_layer=phi_activation_per_conv_layer,
            batch_normalization_per_conv_layer=batch_normalization_per_conv_layer,
            maxpool_reduction_per_conv_block=maxpool_reduction_per_conv_block,
            m_independent_channels=m_independent_channels,
            w_kernel_size=w_kernel_size, w_independent_channels=w_independent_channels,
            fc_num_layers=fc_num_layers, fc_num_units_intermediate_layers=fc_num_units_intermediate_layers,
            fc_batch_normalization=fc_batch_normalization, fc_dropout=fc_dropout,
            penciled_decision=penciled_decision, softmax_output=softmax_output, prenormalization=prenormalization,
            device=None,
            **kwargs
        )

        ######################################################
        # INITIALIZE THE STRUCTURES REQUIRED BY 'ClassifierBaseModel' WHICH WILL BE FILLED IN THE CONSTRUCTOR
        ######################################################

        prenormalization_module = None
        backbone_module = None
        head_module = None

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: BACKBONE
        ######################################################
        ######################################################

        # Make sure that there are no repeated args and kwargs in the function by creating a dict that we peel out
        kwargs_for_create_standard_backbone_module = copy.deepcopy(self._extra_state_dict['constructor_kwargs'])
        # We remove first arguments that we know we will not use
        for key in ['out_classes', 'fc_num_layers', 'fc_num_units_intermediate_layers',
                    'fc_batch_normalization', 'fc_dropout', 'penciled_decision', 'softmax_output',
                    'prenormalization', 'maxpool_reduction_per_conv_block']:
            if key in kwargs_for_create_standard_backbone_module.keys():
                kwargs_for_create_standard_backbone_module.pop(key)
            pass
        pass

        # We create a backbone specific for the AlexNetClassifier in strict sense
        backbone_module, backbone_out_size, backbone_out_channels = _create_standard_backbone_module(
            in_size=kwargs_for_create_standard_backbone_module.pop('in_size', None),
            in_channels=kwargs_for_create_standard_backbone_module.pop('in_channels', None),
            conv_like_type=kwargs_for_create_standard_backbone_module.pop('conv_like_type', None),
            conv_like_type_position=kwargs_for_create_standard_backbone_module.pop('conv_like_type_position', None),
            conv_block_specification=kwargs_for_create_standard_backbone_module.pop('conv_block_specification', None),
            channels_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('channels_per_conv_layer', None),
            phi_activation_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('phi_activation_per_conv_layer', None),
            m_kernel_size_per_conv_layer=kwargs_for_create_standard_backbone_module.pop('m_kernel_size_per_conv_layer', None),
            batch_normalization_per_layer=kwargs_for_create_standard_backbone_module.pop('batch_normalization_per_conv_layer', None),
            maxpool_reduction_per_conv_block=stride_maxpool_reduction_per_conv_block,
            kernel_maxpool_reduction_per_conv_block=kernel_maxpool_reduction_per_conv_block,
            dilation_maxpool_reduction_per_conv_block=dilation_maxpool_reduction_per_conv_block,
            m_independent_channels=kwargs_for_create_standard_backbone_module.pop('m_independent_channels', None),
            w_kernel_size=kwargs_for_create_standard_backbone_module.pop('w_kernel_size', None),
            w_independent_channels=kwargs_for_create_standard_backbone_module.pop('w_independent_channels', None),
            **kwargs_for_create_standard_backbone_module
        )

        # Add in the backbone AdaptiveAvgPool2d((6, 6)) to follow the original AlexNet structure
        # backbone_out_size = np.prod(backbone_out_size) * 6 * 6
        # backbone_out_channels = backbone_out_channels
        # backbone_module.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((6, 6)))

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: HEAD
        ######################################################
        ######################################################

        num_features_after_flatten = \
            int(backbone_out_channels * np.prod(backbone_out_size))

        head_module = _create_standard_head_module(
            num_features_in=num_features_after_flatten,
            num_features_out=self._extra_state_dict['out_classes'],
            num_layers=self._extra_state_dict['fc_num_layers'],
            num_features_intermediate_layers=self._extra_state_dict['fc_num_units_intermediate_layers'],
            batch_normalization=self._extra_state_dict['fc_batch_normalization'],
            dropout=self._extra_state_dict['fc_dropout'],
            penciled_decision=self._extra_state_dict['penciled_decision'],
            softmax_output=self._extra_state_dict['softmax_output']
        )

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK:
        ### CREATE sef.prenormalization and integrate everything into self._nn
        ### AND MOVE IT TO THE DESIRED DEVICE
        ######################################################
        ######################################################

        # Prenormalization
        prenormalization_module = nn.Identity() if not self._extra_state_dict['prenormalization'] else \
            nn.BatchNorm2d(num_features=in_channels, affine=False, track_running_stats=True)

        # Create the full network by combining backbone and head
        self._nn = nn.Sequential(OrderedDict([
            ('prenormalization', prenormalization_module),
            ('backbone', backbone_module),
            ('head', head_module)
        ]))

        # Computation device
        computation_device = None
        if device is None:  # default
            computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cuda', 'cpu'], \
                f"Invalid device: 'cuda' or 'cpu' expected, {device} found!."
            if device == 'cuda' and not torch.cuda.is_available():
                raise Exception(f"Invalid device: 'cuda' selected but no GPU available.")
            computation_device = device
        pass
        self.to_device(computation_device)

        ######################################################
        ######################################################

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        It randomizes the trainable parameters of the network, using the specified distribution and gain, \
        by calling the static method :py:meth:`.random_initialization_subnetwork`.

        Parameters
        ----------
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        AlexNetClassifierLoose.random_initialization_subnetwork(self._nn,
                                                           distribution=distribution, gain=gain, additive=additive)

    @staticmethod
    def random_initialization_subnetwork(subnetwork,
                                         distribution='normal', gain=1e-3, additive=True):
        """
        Function that iteratively looks for the method ``.random_initialization(...)`` of the children of the provided \
        subnetwork, if it is a :py:class:`torch.nn.Sequential` object, and applies either it or an analogous \
        randomization of its trainable parameters.

        Parameters
        ----------
        subnetwork : torch.nn.Module or torch.nn.Sequential
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """
        if isinstance(subnetwork, nn.Sequential):
            for layer in subnetwork:
                AlexNetClassifierLoose.random_initialization_subnetwork(layer,
                                                                   distribution=distribution, gain=gain,
                                                                   additive=additive)
        elif isinstance(subnetwork, (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer)):
            subnetwork.random_initialization(distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, nn.Module):
            for param_tensor in subnetwork.parameters():
                current_average = param_tensor.flatten().mean().item() if additive else 0.0
                if distribution == 'normal':
                    nn.init.normal_(param_tensor,
                                    mean=current_average, std=0.5 * gain)
                elif distribution == 'uniform':
                    nn.init.uniform_(param_tensor,
                                     a=current_average - 0.5 * gain, b=current_average + 0.5 * gain)
                pass
            pass
        pass


#########################################################################################
#########################################################################################
# CLASSES: EfficientNetv2sClassifier model
#########################################################################################
#########################################################################################


class EfficientNetv2sClassifier(MultilayerClassifier):
    """
    Partially customizable classifier inspired by the family of neural network architectures and, \
    more particular, based on EfficientNetv2-S [Tan2021]_. \
    The version *v2-S* (for "searched") has  (see [Tan2021]_, Figure 2 and Table 4; interpretation of the figures \
    based on `torchvision.models.efficientnet <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_):

    - an initial convolutional layer of size 3x3 whose output has **24 channels**,
    - followed by a number of blocks of Fused-MBConv and MBConv layers (containing skip connections and \
      Squeeze-and-Excitation blocks [Hu2019]_), referred to as *inverted residual blocks*, \
      and whose first layer accepts 24 channels, and
    - a final stage having a (1x1) convolution that **raises the number of channels to 1280**, followed by \
      a(n adaptive) pooling layer bringing this result to 1-pixel extent and a final FC layer \
      to the number of final classes.

    The present class :py:class:`.EfficientNetv2sClassifier` adapts the above architecture so the first \
    convolutional-like layer and the last convolution + pooling + FC layer are tunable, while keeping the rest \
    of intermediate layers as in the original EfficientNetv2 (see \
    `torchvision.models.efficientnet <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_). \
    The introduced modifications, thus:

    - Allow tuning the initial and final convolutional layers, so their spatial sizes are different than the \
      original, and for different convolutional-like layer types such as INRF or ibnn_internal layers. \
      **Note: the first convolutional like layer always generates 24 output channels.**
    - The resulting number of channels of the last convolution, equal to input dimension of thet final FC layer, \
      is also tunable (`fc_num_units_intermediate_layers`).

    References
    ----------
    .. [Tan2021]  *Tan, M., & Le, V. L.*. EfficientNetv2: Smaller Models and Faster Training, \
            2021 (in ICML 2021). `link <https://arxiv.org/abs/2104.00298>`_
    .. [Hu2019]  *Hu, J., Shen, L., Albanie, S., Sun, G., & Wu, E.*. Squeeze-and-Excitation Networks, \
            2019 (in TPAMI 2019). `link <https://arxiv.org/abs/1709.01507>`_

    **IMPORTANT NOTE: Filter kernels** $\\Omega$ **without spatial extent**, that is, with ``w_kernel_size`` = ``()`` \
    (empty tuple), **represent a uniform value for the whole extent of the image;** \
    **this convention does not apply to** $\\mathbf{M}$: see :py:func:`.conv2d_crossdiff` for more details.

    Parameters
    ----------
    in_size : tuple[int]
        Spatial size of the input images: $(H,W)$
    in_channels : int
    out_classes : int
    conv_like_type : str
        Value among ``'sm'``, ``'inrfv1'``, ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    conv_like_type_position : str, optional
        Value among ``'everywhere'``, ``'first'``, and ``'last'``. If ``'everywhere'`` the convolutional-like layer \
        indicated in `conv_like_type` is used in all convolutional \
        sub-layers of each block of the network; if ``'first'``, only the first layer of the first block will be of \
        type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``); finally, if ``'last'``, only the \
        last layer of the last block will be of type `conv_like_type` and the rest usual CNN layers (*i.e.* ``'sm'``)
    m_kernel_size_per_conv_layer : tuple|list[ tuple|list[int|float] | int | float ], optional
        List/tuple with as many elements as layers indicated by `conv_block_specification' (the sum of its elements) \
        and as many as `channels_per_conv_layer`. Each element of the list/tuple follows the convention of the \
        convolutional-like layers (:py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer`) for their argument `m_kernel_size`.
        Default: ``((3,3), (1,1))`` (as in the original, see \
        `torchvision.models.efficientnet <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_)
    num_hidden_channels_2 : int, optional
        Default: `1280` (as in the original, see \
        `torchvision.models.efficientnet <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_)
    phi_activation : str, optional
        Activation function used in the convolutional-like layers. Default: ``'SiLu'``
    m_independent_channels : bool, optional
        Whether the output of 'm' mixes all channels (that is, 'm_group'=1) or channels are addressed fully \
        independently (that is, 'm_groups' equals the number of inputs). Default: ``False``
    w_kernel_size : tuple[int] or tuple[float] or list[int] or list[float] or int or float
        The kernel size to be checked: if floats 0<x<0 the absolute pixel size of each layer is resolved from its \
        corresponding input image. The argument is set with a default ``None``, but the \
        constructor **raises an Exception if no value, or** ``None``, **is provided** and the type of indicated \
        ``conv_like_type`` is among ``'inrfv2'``, ``'inrfv3'``, ``'ibnn_lite'``, ``'ibnn_internal'``, ``'ibnn'``
    w_independent_channels : bool, optional
        Whether the output of 'w' mixes all channels (that is, 'w_group'=1) or channels are addressed fully \
        independently (that is, 'w_groups' equals the number of inputs). Default: ``True``
    fc_num_layers : int, optional
        Number of fully connected layers, which needs to be $>0$. Default: 1
    fc_num_units_intermediate_layers : int, optional
        Number of intermediate units $>0$ in the intermediate fully connected layers. The first layer has a number \
        of inputs given by the incoming data and the last layer has a number of output classes: therefore \
        this parameter is not used when the argument `fc_num_layers` is exactly $1$; and in such case \
        `fc_num_units_intermediate_layers` should be set to ``None`` to indicate explicit acknowledgment of this fact and, \
        when it is not $-1$, it will be set to $-1$ internally for record and a warning will be issued.
        Default: `-1`
    dropout : float, optional
        Dropout probability for the fully connected layers.
        Default: ``0.5``
    stochastic_depth_prob : float, optional
        Probability of skipping the current layer in the stochastic depth procedure.
        Default: ``0.2``
    prenormalization : bool, optional
        If ``True``, the input to the network is normalized to mean $0$ and standard deviation $1$ (using \
        an initial :py:class:`torch.nn.BatchNorm2d` without affine transform); otherwise, \
        the input is unaltered.
        Default: `True`
    batch_normalization : bool, optional
        Default: `True`
    softmax_output : bool, optional
        If ``True``, the output of the network is passed through a softmax layer. Default: ``False``
    device : str, optional
        Value among ``'cuda'`` and ``'cpu'``, device where the classifier is created. Error if ``'cuda'`` but \
        no GPU is available.
        Default: ``'cuda'`` if GPU available, ``'cpu'`` otherwise
    **kwargs : optional
        These keyword arguments refer to specific arguments of, respectively, :py:class:`.SMLayer`, \
        :py:class:`.INRFv1Layer`, :py:class:`.INRFv2Layer`, :py:class:`.INRFv3Layer`, :py:class:`.IBNNLiteLayer`, \
        and :py:class:`.IBNNInternalLayer` if selected: see their documentation for greater detail.
    """

    def __init__(self, in_size, in_channels, out_classes,
                 conv_like_type, conv_like_type_position,
                 m_kernel_size_per_conv_layer=((3, 3), (1, 1)), num_hidden_channels_2=1280,
                 phi_activation='silu', m_independent_channels=False,
                 w_kernel_size=None, w_independent_channels=True,
                 fc_num_layers=1, fc_num_units_intermediate_layers=-1,
                 dropout=0.5, stochastic_depth_prob=0.2,
                 prenormalization=False, batch_normalization=True, softmax_output=False,
                 device=None,
                 **kwargs):

        ######################################################
        # FIRST: Load the inverted residual block specification of the architecture "efficientnet_v2_s"
        ######################################################

        # Obtain the intermediate layers of the network from the function '_efficientnet_conf' of 'torchvision.models'
        # NOTE: the layers contained in the list 'inverted_residual_setting' returned by the function \
        # '_efficientnet_conf' are instances of the class '_MBConvConfig', which is a named tuple with the following \
        # fields: 'block', 'input_channels', 'out_channels', 'num_layers', 'expand_ratio', 'kernel_size', 'stride'.
        # Therefore we can use the 'block' field to create the actual block, and the fields
        # 'input_channels' and 'out_channels' to set the input and output channels of the block.

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")

        # Checks
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")
        pass

        # Number of input channels to the inverted residual blocks
        in_channels_to_inverted_residual_blocks = inverted_residual_setting[0].input_channels

        # Number of output channels of the inverted residual blocks
        out_channels_from_inverted_residual_blocks = inverted_residual_setting[-1].out_channels

        ######################################################
        # RUN THE CONSTRUCTOR OF THE PARENT CLASS MultilayerClassifier SO:
        # - THE STRUCTURES OF 'nn.Module' ARE CREATED
        # - THE SPECIFIC CHECKS OF 'MultilayerClassifier' ARE RUN
        ######################################################

        conv_block_specification = (1, 1)  # There are two (first and last) tunable convolutional-like layers
        #
        ## The first layer has 'in_channels_to_inverted_residual_blocks'' channels, the last one has num_hidden_channels_2
        channels_per_conv_layer = (in_channels_to_inverted_residual_blocks, num_hidden_channels_2)

        # If 'maxpool_reduction' was provided as a keyword, take it out and check it: None will be made 1 (int)
        maxpool_reduction = kwargs.pop('maxpool_reduction', None)
        maxpool_reduction = 1 if maxpool_reduction is None else maxpool_reduction

        assert maxpool_reduction == 1, \
            (f"'maxpool_reduction' is not used in EfficientNetv2sClassifier so, if present, " +
             f"it should be None or 1, but got {maxpool_reduction} instead")

        MultilayerClassifier.__init__(
            self,
            in_size=in_size, in_channels=in_channels, out_classes=out_classes,
            conv_like_type=conv_like_type, conv_like_type_position=conv_like_type_position,
            conv_block_specification=conv_block_specification,
            channels_per_conv_layer=channels_per_conv_layer,
            m_kernel_size_per_conv_layer=m_kernel_size_per_conv_layer,
            phi_activation=phi_activation, m_independent_channels=m_independent_channels,
            w_kernel_size=w_kernel_size, w_independent_channels=w_independent_channels,
            fc_num_layers=fc_num_layers, fc_num_units_intermediate_layers=fc_num_units_intermediate_layers,
            prenormalization=prenormalization, batch_normalization=batch_normalization,
            maxpool_reduction=maxpool_reduction, softmax_output=softmax_output, device=device
        )

        ######################################################
        # INITIALIZE THE STRUCTURES REQUIRED BY 'ClassifierBaseModel' WHICH WILL BE FILLED IN THE CONSTRUCTOR
        ######################################################

        ######
        # The following has been already "created partly" by the parent class, so we just need to fill it
        ######
        # self._extra_state_dict = {}
        # self._extra_state_dict['constructor_kwargs'] = {}  # TO BE FILLED LATER!!!
        # self._fields_to_log = {}
        # self._nn = None
        # self._extra_state_dict['net'] = type(self).__name__
        # self._fields_to_log['net'] = type(self).__name__
        ######

        prenormalization_module = None
        backbone_module = None
        head_module = None

        ######################################################
        ### INITIAL, GENERAL PARAMETERS: check what is not already checked by the parent class
        ######################################################
        ### ... which is: 'dropout' and 'stochastic_depth_prob',
        ######################################################

        assert isinstance(dropout, (int, float)) and 0 <= dropout <= 1, \
            f"dropout should be a float in [0,1], but got {dropout} instead"
        self._extra_state_dict['constructor_kwargs']['dropout'] = copy.deepcopy(dropout)

        assert isinstance(stochastic_depth_prob, (int, float)) and 0 <= stochastic_depth_prob <= 1, \
            f"stochastic_depth_prob should be a float in [0,1], but got {stochastic_depth_prob} instead"
        self._extra_state_dict['constructor_kwargs']['stochastic_depth_prob'] = copy.deepcopy(stochastic_depth_prob)

        ######################################################
        ### LAYER PARAMETER SEPARATION:
        ### The will generate the required for 'sm' and for the selected conv-like layer
        ### to ease their use for populating the backbone of the network
        ######################################################

        # Fields in the constructor of the layers to leave out: the fields that will be filled individually per layer
        fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'm_kernel_size', 'm_groups', 'w_groups']

        # Dictionary of explicit args in the network that we want to force
        dict_explicit_args_sm = {'phi_activation': self._extra_state_dict['constructor_kwargs']['phi_activation'],
                                 'm_padding': 'same'}
        dict_explicit_args_conv_like_layer = copy.deepcopy(dict_explicit_args_sm)
        if self._extra_state_dict['constructor_kwargs']['conv_like_type'] in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal',
                                                                              'ibnn']:
            dict_explicit_args_conv_like_layer['w_kernel_size'] = self._extra_state_dict['constructor_kwargs'][
                'w_kernel_size']
        pass

        # Dictionary, but for the fields in the list 'fields_leave_out', for the SMLayers
        sm_kwargs = processed_constructor_kwargs_for_conv_like_layer(
            _dict_conv_like_layers['sm'],
            fields_leave_out=fields_leave_out,
            dict_explicit_args=dict_explicit_args_sm, dict_non_explicit_args=kwargs,
            flag_add_all_other_args=False
        )

        # Dictionary, but for the fields in the list 'fields_leave_out', for the selected conv-like layer
        conv_like_layer_kwargs = processed_constructor_kwargs_for_conv_like_layer(
            _dict_conv_like_layers[self._extra_state_dict['constructor_kwargs']['conv_like_type']],
            fields_leave_out=fields_leave_out,
            dict_explicit_args=dict_explicit_args_conv_like_layer, dict_non_explicit_args=kwargs,
            flag_add_all_other_args=True
        )

        ######################################################
        ### Add all the 'conv_like_layer_kwargs' into the constructor kwargs
        ######################################################

        for key in conv_like_layer_kwargs:
            self._extra_state_dict['constructor_kwargs'][key] = copy.deepcopy(conv_like_layer_kwargs[key])
        pass

        ######################################################
        ### Include all the arguments in the 'self._extra_state_dict['constructor_kwargs'] into 'self._fields_to_log'
        ######################################################
        for key in self._extra_state_dict['constructor_kwargs']:
            self._fields_to_log[key] = copy.deepcopy(self._extra_state_dict['constructor_kwargs'][key])
        pass

        ######################################################
        ### FORCE ALL FIELDS IN "self._fields_to_log" TO BE HASHABLE... by transforming to str if necessary
        ######################################################
        for key in self._fields_to_log:
            if not isinstance(self._fields_to_log[key], typing.Hashable):
                self._fields_to_log[key] = str(self._fields_to_log[key])
            pass
        pass

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: BACKBONE
        ######################################################
        ######################################################

        # 'backbone_module' is a sequential to be filled with the blocks of the backbone. In order to give names,
        # we use an OrderedDict, which is then converted to a nn.Sequential
        ordered_dict_backbone_module = OrderedDict()

        ######################################################
        # Building the first conv layer
        ######################################################

        if self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'first' or \
                self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'everywhere':
            conv_like_layer_type = self._extra_state_dict['constructor_kwargs']['conv_like_type']
        else:
            conv_like_layer_type = 'sm'
        pass

        # Conv-like layer: extra params to be added regarding kernel sizes groups (separated m and w)
        # (indicated in fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'm_kernel_size', 'm_groups', 'w_groups'])

        kwargs_layer_i = copy.deepcopy(sm_kwargs) if conv_like_layer_type == 'sm' \
            else copy.deepcopy(conv_like_layer_kwargs)

        kwargs_layer_i['in_size'] = copy.deepcopy(self._extra_state_dict['constructor_kwargs']['in_size'])
        kwargs_layer_i['in_channels'] = copy.deepcopy(self._extra_state_dict['constructor_kwargs']['in_channels'])
        kwargs_layer_i['out_channels'] = self._extra_state_dict['constructor_kwargs']['channels_per_conv_layer'][0]
        kwargs_layer_i['m_kernel_size'] = self._extra_state_dict['constructor_kwargs']['m_kernel_size_per_conv_layer'][0]
        kwargs_layer_i['m_groups'] = kwargs_layer_i['out_channels'] \
            if self._extra_state_dict['constructor_kwargs']['m_independent_channels'] else 1

        if conv_like_layer_type in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            # If the stored 'self._extra_state_dict['w_kernel_size']' for the whole network has floats,
            # then it is relative to the size of the input image. If it is ints, it is absolute.
            kwargs_layer_i['w_groups'] = kwargs_layer_i['out_channels'] \
                if self._extra_state_dict['constructor_kwargs']['w_independent_channels'] else 1
        pass

        # Create the layer
        conv_like_layer = _dict_conv_like_layers[conv_like_layer_type](
            **kwargs_layer_i
        )

        # Append the conv-like layer to the list of the current block, including name
        ordered_dict_backbone_module.update({'conv_0': conv_like_layer})

        # Add the batch normalization layer if required, as a separate "post-" layer
        if batch_normalization:
            conv_like_layer_post = nn.BatchNorm2d(
                num_features=kwargs_layer_i['out_channels'], affine=True, track_running_stats=True
            )
        else:
            conv_like_layer_post = nn.Identity()
        pass

        # Append the post-layer normalization to the list of the current block
        ordered_dict_backbone_module.update({'norm_conv_0': conv_like_layer_post})

        ######################################################
        # Building inverted residual blocks... already loaded from the architecture
        ######################################################

        # "torchvision.models.efficientnet" and its component classes and functions do already address their respective
        # activations and even batch normalization if indicated so (and even something called "stochastic depth"), so
        # we leverage said ability and trust that the layers and their post-layer normalization are correct.

        ordered_dict_inverted_residual_stages = OrderedDict()

        # and fill it iteratively
        # with ints from 0 to 10, giving each number a name 'stage_n'
        inverted_residual_setting = copy.deepcopy(inverted_residual_setting)
        if not isinstance(inverted_residual_setting, Sequence) or \
                not all([isinstance(cnf, _MBConvConfig) for cnf in inverted_residual_setting]):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for stage_i, cnf in enumerate(inverted_residual_setting):
            stage = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                # Add a batch normalization if so indicated
                post_layer_norm_class = None
                if batch_normalization:
                    post_layer_norm_class = nn.BatchNorm2d
                else:
                    post_layer_norm_class = nn.Identity
                pass

                # Append the block, including normalization, to the stage
                stage.append(block_cnf.block(block_cnf, sd_prob, post_layer_norm_class))
                stage_block_id += 1
                #
            pass
            #
            # Append individually each stage of the inverted_residual_blocks into its joint sequential
            ordered_dict_inverted_residual_stages.update(
                {f"stage_{stage_i}": nn.Sequential(*stage)}
            )
            #
        pass

        # Append, as a block, all the layers/modules of the inverted_residual_blocks into the backbone module
        ordered_dict_backbone_module.update(
            {'inverted_residual_stages': nn.Sequential(ordered_dict_inverted_residual_stages)}
        )

        ######################################################
        # Build the last convolutional layer (fitting the sizes of the previous layers)
        ######################################################

        # NOTE: THE inverted_residual_stages, DUE TO THEIR STRUCTURE OF STRIDES, REDUCE THE SIZE OF THE INPUT BY 2^4
        im_size_after_inverted_residual_stages = tuple([
            math.ceil(elem / (2 ** 4)) for elem in self._extra_state_dict['constructor_kwargs']['in_size']
        ])

        if self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'last' or \
                self._extra_state_dict['constructor_kwargs']['conv_like_type_position'] == 'everywhere':
            conv_like_layer_type = self._extra_state_dict['constructor_kwargs']['conv_like_type']
        else:
            conv_like_layer_type = 'sm'
        pass

        # Conv-like layer: extra params to be added regarding kernel sizes groups (separated m and w)
        # (indicated in fields_leave_out = ['in_size', 'in_channels', 'out_channels', 'm_kernel_size', 'm_groups', 'w_groups'])

        kwargs_layer_i = copy.deepcopy(sm_kwargs) if conv_like_layer_type == 'sm' \
            else copy.deepcopy(conv_like_layer_kwargs)

        kwargs_layer_i['in_size'] = im_size_after_inverted_residual_stages
        kwargs_layer_i['in_channels'] = out_channels_from_inverted_residual_blocks
        kwargs_layer_i['out_channels'] = self._extra_state_dict['constructor_kwargs']['channels_per_conv_layer'][-1]
        kwargs_layer_i['m_kernel_size'] = self._extra_state_dict['constructor_kwargs']['m_kernel_size_per_conv_layer'][-1]
        kwargs_layer_i['m_groups'] = kwargs_layer_i['out_channels'] \
            if self._extra_state_dict['constructor_kwargs']['m_independent_channels'] else 1

        if conv_like_layer_type in ['inrfv2', 'inrfv3', 'ibnn_lite', 'ibnn_internal', 'ibnn']:
            # If the stored 'self._extra_state_dict['w_kernel_size']' for the whole network has floats,
            # then it is relative to the size of the input image. If it is ints, it is absolute.
            kwargs_layer_i['w_groups'] = kwargs_layer_i['out_channels'] \
                if self._extra_state_dict['constructor_kwargs']['w_independent_channels'] else 1
        pass

        # Create the layer
        conv_like_layer = _dict_conv_like_layers[conv_like_layer_type](
            **kwargs_layer_i
        )

        # Append the conv-like layer to the list of the current block, including name
        ordered_dict_backbone_module.update({'conv_1': conv_like_layer})

        # Add the batch normalization layer if required, as a separate "post-" layer
        if batch_normalization:
            conv_like_layer_post = nn.BatchNorm2d(
                num_features=kwargs_layer_i['out_channels'], affine=True, track_running_stats=True
            )
        else:
            conv_like_layer_post = nn.Identity()
        pass

        # Append the post-layer normalization to the list of the current block
        ordered_dict_backbone_module.update({'norm_conv_1': conv_like_layer_post})

        ######################################################
        # Pack the ordered dict into the backbone module
        ######################################################

        backbone_module = nn.Sequential(ordered_dict_backbone_module)

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK: HEAD
        ######################################################
        ######################################################

        ######################################################
        # Append an AdaptiveAvgPool2d layer to get to H=W=1
        ######################################################

        head_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self._extra_state_dict['constructor_kwargs']['channels_per_conv_layer'][-1], out_classes)
        )

        # Append the conv-like layer to the list of the current block, including name
        ### EXIT LAYER: SOFTMAX?
        # Create the exit layer, which can be a softmax layer or an identity layer, and append it to the list of layers
        exit_layer = None
        if self._extra_state_dict['constructor_kwargs']['softmax_output']:
            exit_layer = nn.Softmax(dim=-1)
        else:
            exit_layer = nn.Identity()
        pass
        head_module.append(exit_layer)

        ######################################################
        ######################################################
        ### POPULATE THE NETWORK:
        ### CREATE sef.prenormalization and integrate everything into self._nn
        ### AND MOVE IT TO THE DESIRED DEVICE
        ######################################################
        ######################################################

        # Prenormalization
        prenormalization_module = nn.Identity() \
            if not self._extra_state_dict['constructor_kwargs']['prenormalization'] \
            else nn.BatchNorm2d(num_features=self._extra_state_dict['constructor_kwargs']['in_channels'],
                                affine=False, track_running_stats=True)

        # Create the full network by combining backbone and head
        self._nn = nn.Sequential(OrderedDict([
            ('prenormalization', prenormalization_module),
            ('backbone', backbone_module),
            ('head', head_module)
        ]))

        ######################################################
        ######################################################
        ### Move/register into the computation device
        ######################################################
        ######################################################

        self.to_device(self.chosen_device)

        ######################################################
        ######################################################

    def random_initialization(self, distribution='normal', gain=1e-3, additive=True):
        """
        It randomizes the trainable parameters of the network, using the specified distribution and gain, \
        by calling the static method :py:meth:`.random_initialization_subnetwork`.

        Parameters
        ----------
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """

        EfficientNetv2sClassifier.random_initialization_subnetwork(self._nn,
                                                                   distribution=distribution, gain=gain,
                                                                   additive=additive)

    @staticmethod
    def random_initialization_subnetwork(subnetwork,
                                         distribution='normal', gain=1e-3, additive=True):
        """
        Function that iteratively looks for the method ``.random_initialization(...)`` of the children of the provided \
        subnetwork, if it is a :py:class:`torch.nn.Sequential` object, and applies either it or an analogous \
        randomization of its trainable parameters.

        Parameters
        ----------
        subnetwork : torch.nn.Module or torch.nn.Sequential
        distribution : str, optional
            Value among ``'normal'`` and ``'uniform'``, whereas the latter is U[-0.5,0.5]. \
            Default: ``'normal'``
        gain : int or float, optional
            Gain/scale applied to the 'standard', normalized distribution indicated by ``distribution``. \
            Default: ``1e-3``
        additive : bool, optional
            Randomness added (``True``) around the current value or around zero (``False``).
            Default: ``True``
        """
        if isinstance(subnetwork, nn.Sequential):
            for layer in subnetwork:
                EfficientNetv2sClassifier.random_initialization_subnetwork(layer,
                                                                           distribution=distribution, gain=gain,
                                                                           additive=additive)
        elif isinstance(subnetwork, (SMLayer, INRFv1Layer, INRFv2Layer, INRFv3Layer, IBNNLiteLayer, IBNNInternalLayer)):
            subnetwork.random_initialization(distribution=distribution, gain=gain, additive=additive)
        elif isinstance(subnetwork, nn.Module):
            for param_tensor in subnetwork.parameters():
                current_average = param_tensor.flatten().mean().item() if additive else 0.0
                if distribution == 'normal':
                    nn.init.normal_(param_tensor,
                                    mean=current_average, std=0.5 * gain)
                elif distribution == 'uniform':
                    nn.init.uniform_(param_tensor,
                                     a=current_average - 0.5 * gain, b=current_average + 0.5 * gain)
                pass
            pass
        pass
