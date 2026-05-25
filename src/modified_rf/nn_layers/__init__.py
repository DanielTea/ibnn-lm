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

from .core import *
from .utils import *
from .functional import *

_dict_conv_like_layers = {
    'sm': SMLayer,
    'inrfv1': INRFv1Layer,
    'inrfv2': INRFv2Layer,
    'inrfv3': INRFv3Layer,
    'ibnn_lite': IBNNLiteLayer,
    'ibnn_internal': IBNNInternalLayer,
    'ibnn': IBNNLayer,
}