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

# __init__.py

__version__ = '0.5'
from torch import optim, nn


#############################################################################################
# DATA STRUCTURES FOR LOGGING EXPERIMENTS
#############################################################################################

_dict_optimizer_classes = {'sgd': optim.SGD,
                           'asgd': optim.ASGD,
                           'adam': optim.Adam
                           }

_dict_scheduler_classes = {'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
                           'ExponentialLR': optim.lr_scheduler.ExponentialLR,
                           'linear_warmup': optim.lr_scheduler.LinearLR, # It is in fact an alias for LinearLR
                           'LinearLR': optim.lr_scheduler.LinearLR, # It is in fact an alias for ConstantLR
                           'warmup': optim.lr_scheduler.ConstantLR, # It is in fact an alias for ConstantLR
                           'ConstantLR': optim.lr_scheduler.ConstantLR,
                           'CyclicLR': optim.lr_scheduler.CyclicLR
                           }

_dict_loss_functions = {'cross_entropy': nn.functional.cross_entropy,
                        'mse_loss': nn.functional.mse_loss,
                        'nll_loss': nn.functional.nll_loss,
                        'l1_loss': nn.functional.l1_loss,
                        'smooth_l1_loss': nn.functional.smooth_l1_loss,
                        'kl_div': nn.functional.kl_div
                        }
