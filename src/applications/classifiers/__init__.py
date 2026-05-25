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

_dict_classifiers = {
    'VGGxClassifier': VGGxClassifier,
    # 'SingleHiddenLayerClassifier': SingleHiddenLayerClassifier,
    # 'DoubleHiddenLayerClassifier': DoubleHiddenLayerClassifier,
    'MultiHiddenLayerClassifier': MultiHiddenLayerClassifier,
    'AlexNetClassifier': AlexNetClassifier,
    'EfficientNetv2sClassifier': EfficientNetv2sClassifier,
 }

_dict_classifiers_as_in_conf_file = {
    # 'single_layer': SingleHiddenLayerClassifier,
    # 'double_layer': DoubleHiddenLayerClassifier,
    'multi_layer': MultiHiddenLayerClassifier,
    'vggx': VGGxClassifier,
    'alexnet': AlexNetClassifier,
    'efficientnetv2s': EfficientNetv2sClassifier,
 }
