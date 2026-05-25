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

from itertools import chain

import numpy as np
import torch
from torch.nn.functional import softmax

from torchattacks.attack import Attack
from torchattacks import Pixle

class PixleFix(Pixle):
    """
    This implementation of Pixle (PixleFix) takes almost exatcly the implementation of Pixle in TorchAttacks \
    and only rewrites few components of the original that do not match the purposes of our evaluation \
    (e.g. the method '_perturb' of the original implementation appears to work only for 3-channel images, \
    while we intend Pixle attacks also on 1- and N-channel images).

    For reference about the original implementation of Pixle, please refer to
    Pixle: a fast and effective black-box attack based on rearranging pixels'
    [https://arxiv.org/abs/2202.02236]

    As info regarding image/label shapes:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, \
          `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, iteratsion=50)
        >>> adv_images = attack(images, labels)

    Parameters
    ----------
    model (nn.Module) : nn.Module
            Model to attack
    x_dimensions : int or float, or a tuple containing a combination of those
        Size of the sampled patch along ther x side for each iteration. \
        The integers are considered as fixed number of size, while the float as parcentage of the size. \
        A tuple is used to specify both under and upper bound of the size.
        Default: ``(2, 10)``
    y_dimensions : int or float or a tuple containing a combination of those
        Size of the sampled patch along ther y side for each iteration. \
        The integers are considered as fixed number of size, while the float as parcentage of the size. \
        A tuple is used to specify both under and upper bound of the size.
        Default: (2, 10)
    pixel_mapping : str
        The type of mapping used to move the pixels. Can be: \
        'random', 'similarity', 'similarity_random', 'distance', 'distance_random'.
        Default: 'random'
    restarts : int
        The number of restarts that the algortihm performs.
        Default: 20
    max_iterations : int)
        Number of iterations to perform for each restart
        Default: 10
    update_each_iteration : bool
        If the attacked images must be modified after each iteration (``True``) or after each restart (``False``)
        Default: ``False``
    """

    def __init__(self, model, x_dimensions=(2, 10), y_dimensions=(2, 10),
                 pixel_mapping='random', restarts=20,
                 max_iterations=10, update_each_iteration=False):
        super().__init__(model, x_dimensions, y_dimensions,
                 pixel_mapping, restarts,
                 max_iterations, update_each_iteration)

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        s = source[0][source_pixels].view(c, -1)

        destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination
