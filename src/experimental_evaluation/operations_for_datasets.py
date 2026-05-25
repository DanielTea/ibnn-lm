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
import copy

import numpy as np

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import autocontrast
from torchvision.io import decode_image, ImageReadMode

import PIL

from scipy import ndimage
import sklearn.datasets
from skimage import color

import urllib.request
import zipfile
import shutil
from tqdm import tqdm


#############################################################################################
# The definition of transforms of "torchvision.transforms" behave like "torch.nn.Module" objects:
# see https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py
#############################################################################################


class RGB2CIELuv(v2.Transform):
    """
    Convert sRGB image to CIELuv color space, if the input is RGB. If the input is monochromatic (1 channel), \
    it is converted to 3 channels by replicating the single channel 3 times. The range of the output channels \
    corresponds to the usual ranges of the CIELuv color space, which are, respectively and approximately, \
     [0, 100], [-134, 220], and [-140, 122].

    The behavior of the function differs slightly depending on the format of the data passed as input:

    - Regarding whether the input is only an image or batch of images or a list/tuple wherein \
        the first element is an image:
        - The image/batch of images is transformed
        - and the result is returned as it was input: if list/tuple, a list with the non-image \
          elements untouched.

    - Regarding the format of the image:
        - If the input is a numpy array, the image is returned as a numpy array.
        - If the input is a torch.Tensor, the image is returned as a torch.Tensor.
        - In both cases: **the input image is always interpreted as a float in the range [0, 1]**, even \
         if its data type is an integer. If its value is greater than 1.0 or smaller than 0.0 there will be an error.
        - In all cases: if the input is monochromatic (1 channel), it is converted to 3 channels by replicating the \
            single channel 3 times.

    Additionally, different normalization types can be applied to **each** single image, \
    if desired (*Note*: the channels of the \
    CIELuv color space are located, respectively and approximately, in the ranges [0,100], [-134,220], and [-140,122]):
        - ``None`` or ``'none'``: the image is left untouched
        - ``'L'``: each image is standardized so its L channel is centered at 50, the theoretical average of the \
            L channel, and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping.
        - ``'Luv'``: each image is standardized so its L channel is centered at 50 (general average of the L channel) \
            and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping; the u and v channels \
            are centered at 0 and have a standard deviation such that 3 standard deviations both sides of 0 fit inside \
            the theoretical range of the channel.
    The normalization is performed by an object of the class :py:class:`.NormalizeCIELuv` contained as an attribute.
    """

    def __init__(self, normalization_type=None):
        #
        super().__init__()
        self.cieluv_normalizer = NormalizeCIELuv(normalization_type=normalization_type)

    def forward(self, sample):
        """
        Convert RGB image to CIELuv color space, if the input is RGB. If the input is monochromatic (1 channel), \
        it is converted to 3 channels by replicating the single channel 3 times.

        Parameters
        ----------
        sample : np.ndarray or torch.Tensor or list or tuple
            The image to be transformed, or a list/tuple wherein the first element is the image to be transformed

        Returns
        -------
        np.ndarray or torch.Tensor or list or tuple
            The transformed image, or a list/tuple wherein the first element is the transformed image
        """

        ####################################
        # List/tuple or single image? Extract the image
        ####################################

        img = None
        flag_sample_is_list_tuple = False
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            img = sample
            flag_sample_is_list_tuple = False
        elif isinstance(sample, (list, tuple)):
            img = sample[0]
            flag_sample_is_list_tuple = True
        else:
            raise ValueError(
                f"Sample type {type(sample)} not supported: only np.ndarray, torch.Tensor or list/tuple are supported")
        pass

        ####################################
        # Check if img is a tensor: the operation is always performed using np.arrays, so in such case transform.
        # WARNING: THE CHANNELS ARE, IN NUMPY, IN THE VERY LAST DIMENSION -> ADAPT IT WHEN NECESSARY
        ####################################

        flag_original_image_is_tensor = False
        img_numpy = None

        if isinstance(img, torch.Tensor):
            flag_original_image_is_tensor = True
            img_numpy = img.movedim(source=-3, destination=-1).numpy()
        elif isinstance(img, np.ndarray):
            img_numpy = img
        else:
            raise ValueError(f"Image type {type(img)} not supported: only torch.Tensor images are supported")
        pass

        ####################################
        # Check tensor of 3 channels image; if 1 channel, replicate it 3 times; otherwise raise an error
        ####################################

        if img_numpy.shape[-1] == 1:
            img_numpy = np.repeat(img_numpy, 3, axis=-1)
        elif img_numpy.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"Image shape {img_numpy.shape} not supported: only 3 channels images are supported")
        pass

        ####################################
        # Convert RGB image to CIELuv color space
        ####################################

        # Check that the input image is in the range [0, 1]?
        if np.max(img_numpy) > 1.0 or np.min(img_numpy) < 0.0:
            raise ValueError(f"Image values {img_numpy} are not in the range [0, 1]")
        pass

        img_numpy_output = color.rgb2luv(img_numpy, channel_axis=-1)

        ####################################
        # Normalize the image, if necessary
        ####################################

        img_numpy_output = self.cieluv_normalizer(img_numpy_output)

        ####################################
        # Back to tensor, if necessary, and back to list/tuple, if necessary
        ####################################

        img_output = img_numpy_output if not flag_original_image_is_tensor else \
            torch.from_numpy(img_numpy_output).movedim(source=-1, destination=-3).to(torch.float32)

        sample_output = None
        if not flag_sample_is_list_tuple:
            sample_output = img_output
        else:
            sample_output = [img_output] + sample[1:]
            if isinstance(sample, tuple):
                sample_output = tuple(sample_output)
            pass
        pass

        return sample_output


class NormalizeCIELuv(v2.Transform):
    """
    Perform an image-wise normalization of an image batch or image expressed CIELuv color space: the image/batch to \
    normalize is expected to be in CIELuv color space and checks about it will be performed.

    The behavior of the function differs slightly depending on the format of the data passed as input:

    - Regarding whether the input is only an image or batch of images or a list/tuple wherein \
        the first element is an image:
        - The image/batch of images is transformed
        - and the result is returned as it was input: if list/tuple, a list with the non-image \
          elements untouched.

    - Regarding the format of the image:
        - If the input is a numpy array, the image is normalized and returned as a numpy array.
        - If the input is a torch.Tensor, the image is normalized and returned as a torch.Tensor.
        - In both cases: it is assumed that the image, initially float or int, is formatted in the CIELuv having \
            3 channels, respectively L, u, and v, with their respective ranges as indicated below.

    Additionally, different normalization types can be applied to **each** single image, \
    if desired (*Note*: the channels of the \
    CIELuv color space are located, respectively and approximately, in the ranges [0,100], [-134,220], and [-140,122]):
        - ``None`` or ``'none'``: the image is left untouched
        - ``'L'``: each image is standardized so its L channel is centered at 50, the theoretical average of the \
            L channel, and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping.
        - ``'Luv'``: each image is standardized so its L channel is centered at 50 (general average of the L channel) \
            and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping; the u and v channels \
            are centered at 0 and have a standard deviation such that 3 standard deviations both sides of 0 fit inside \
            the theoretical range of the channel.
    """

    def __init__(self, normalization_type=None):
        #
        super().__init__()
        #
        self.normalization_type = normalization_type if normalization_type is not None else 'none'
        if isinstance(self.normalization_type, str):
            self.normalization_type = self.normalization_type.lower()
            if self.normalization_type not in ['none', 'l', 'luv']:
                raise ValueError(
                    f"Normalization type {self.normalization_type}: only 'none', 'L' or 'Luv' are supported")
            pass
        else:
            raise ValueError(
                f"Normalization type {self.normalization_type} not supported: only str ('none', 'L' or 'Luv') or None are supported")
        pass

        # Prepare the values for normalization in case they are needed

        self.theoretical_range_channel = {
            'L': torch.Tensor([0., 100.]),
            'u': torch.Tensor([-134., 220.]),
            'v': torch.Tensor([-140., 122.])
        }

        self.mean_channel = {
            'L': 50.,
            'u': 0.,
            'v': 0.
        }

        self.std_channel = {}
        for channel in self.theoretical_range_channel:
            self.std_channel[channel] = \
                ((self.theoretical_range_channel[channel] - self.mean_channel[channel]).abs().min() / 3).item()
        pass

    def forward(self, sample):
        """
        Normalize CIELuv image(s) according to the stated 'normalization_type' of the constructor.

        Parameters
        ----------
        sample : np.ndarray or torch.Tensor or list or tuple
            The image to be transformed, or a list/tuple wherein the first element is the image to be transformed

        Returns
        -------
        np.ndarray or torch.Tensor or list or tuple
            The transformed image, or a list/tuple wherein the first element is the transformed image
        """

        ####################################
        # List/tuple or single image? Extract the image
        ####################################

        img = None
        flag_sample_is_list_tuple = False
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            img = sample
            flag_sample_is_list_tuple = False
        elif isinstance(sample, (list, tuple)):
            img = sample[0]
            flag_sample_is_list_tuple = True
        else:
            raise ValueError(
                f"Sample type {type(sample)} not supported: only np.ndarray, torch.Tensor or list/tuple are supported")
        pass

        ####################################
        # Check if img is a tensor: the operation is always performed using np.arrays, so in such case transform.
        # WARNING: THE CHANNELS ARE, IN NUMPY, IN THE VERY LAST DIMENSION -> ADAPT IT WHEN NECESSARY
        ####################################

        flag_original_image_is_tensor = False
        img_numpy = None

        if isinstance(img, torch.Tensor):
            flag_original_image_is_tensor = True
            img_numpy = img.movedim(source=-3, destination=-1).numpy()
        elif isinstance(img, np.ndarray):
            img_numpy = img
        else:
            raise ValueError(f"Image type {type(img)} not supported: only torch.Tensor images are supported")
        pass

        ####################################
        # Check that there are exactly 3 channels; otherwise raise an error
        ####################################

        if img_numpy.shape[-1] == 3:
            pass
        else:
            raise ValueError(
                f"Image shape {img_numpy.shape} not supported: 3 channels (CIELuv) images are compulsory.SSS")
        pass

        ####################################
        # Normalize the image, if necessary
        ####################################

        img_numpy_output = img_numpy.copy()

        if self.normalization_type == 'none':
            pass
        else:  # If either 'l' or 'luv'
            # Mean and std of each image and channel (for all)
            mean_img_numpy_output = np.mean(img_numpy_output, axis=(-3, -2), keepdims=True)
            std_img_numpy_output = np.std(img_numpy_output, axis=(-3, -2), keepdims=True)
            # Normalize the requested channels
            for ch_num, ch_name in enumerate(self.mean_channel):
                if ch_num == 0 or self.normalization_type == 'luv':
                    # Normalize the channel
                    img_numpy_output[..., ch_num] = \
                        (img_numpy_output[..., ch_num] - mean_img_numpy_output[..., ch_num]) / \
                        std_img_numpy_output[..., ch_num] * self.std_channel[ch_name] + self.mean_channel[ch_name]
                    pass
                pass
            pass
            # Clip the resulting channels to their theoretical ranges
            np.clip(img_numpy_output, out=img_numpy_output,
                    a_min=[self.theoretical_range_channel[ch_name][0] for ch_name in self.theoretical_range_channel],
                    a_max=[self.theoretical_range_channel[ch_name][1] for ch_name in self.theoretical_range_channel])
        pass

        ####################################
        # Back to tensor, if necessary, and back to list/tuple, if necessary
        ####################################

        img_output = img_numpy_output if not flag_original_image_is_tensor else \
            torch.from_numpy(img_numpy_output).movedim(source=-1, destination=-3).to(torch.float32)

        sample_output = None
        if not flag_sample_is_list_tuple:
            sample_output = img_output
        else:
            sample_output = [img_output] + sample[1:]
            if isinstance(sample, tuple):
                sample_output = tuple(sample_output)
            pass
        pass

        return sample_output


class CIELuv2RGB(v2.Transform):
    """
    Convert a CIELuv image to RGB; the output will be in floats 0.0->1.0 (default) unless it is indicated by the flag \
     ``int_output``, which causes ints 0->255, if explicitly indicated so with said flag \
    (set as ``True``).

    The behavior of the function differs slightly depending on the format of the data passed as input:

    - Regarding whether the input is only an image or batch of images or a list/tuple wherein \
        the first element is an image:
        - The image/batch of images is transformed
        - and the result is returned as it was input: if list/tuple, a list with the non-image \
          elements untouched.

    - Regarding the format of the image:
        - If the input is a numpy array, the image is returned as a numpy array.
        - If the input is a torch.Tensor, the image is returned as a torch.Tensor.
        - In both cases: if the input is monochromatic (1 channel), it is converted to 3 channels by replicating the \
            single channel 3 times.

    Additionally, different CIELuv normalization types can be applied to **each** single image, \
    if desired, which will have an effect on the resulting RGB images (*Note*: the channels of the \
    CIELuv color space are located, respectively, in the ranges [0,100], [-134,220], and [-140,122]):
        - ``None`` or ``'none'``: the image is left untouched
        - ``'L'``: each image is standardized so its L channel is centered at 50, the theoretical average of the \
            L channel, and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping.
        - ``'Luv'``: each image is standardized so its L channel is centered at 50 (general average of the L channel) \
            and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping; the u and v channels \
            are centered at 0 and have a standard deviation such that 3 standard deviations both sides of 0 fit inside \
            the theoretical range of the channel.
    The normalization is performed by an object of the class :py:class:`.NormalizeCIELuv` contained as an attribute.
    """

    def __init__(self, normalization_type=None, int_output=False):
        #
        super().__init__()
        self.cieluv_normalizer = NormalizeCIELuv(normalization_type=normalization_type)
        if not isinstance(int_output, bool):
            raise ValueError(f"Argument 'int_output' of type {type(int_output)}; only bool is supported")
        pass
        self.int_output = int_output

    def forward(self, sample):
        """
        Transform CIELuv image(s) according to the stated 'normalization_type' and 'int_output' flag \
        of the constructor.

        Parameters
        ----------
        sample : np.ndarray or torch.Tensor or list or tuple
            The image to be transformed, or a list/tuple wherein the first element is the image to be transformed

        Returns
        -------
        np.ndarray or torch.Tensor or list or tuple
            The transformed image, or a list/tuple wherein the first element is the transformed image
        """

        ####################################
        # List/tuple or single image? Extract the image
        ####################################

        img = None
        flag_sample_is_list_tuple = False
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            img = sample
            flag_sample_is_list_tuple = False
        elif isinstance(sample, (list, tuple)):
            img = sample[0]
            flag_sample_is_list_tuple = True
        else:
            raise ValueError(
                f"Sample type {type(sample)} not supported: only np.ndarray, torch.Tensor or list/tuple are supported")
        pass

        ####################################
        # Check if img is a tensor: the operation is always performed using np.arrays, so in such case transform.
        # WARNING: THE CHANNELS ARE, IN NUMPY, IN THE VERY LAST DIMENSION -> ADAPT IT WHEN NECESSARY
        ####################################

        flag_original_image_is_tensor = False
        img_numpy = None

        if isinstance(img, torch.Tensor):
            flag_original_image_is_tensor = True
            img_numpy = img.movedim(source=-3, destination=-1).numpy()
        elif isinstance(img, np.ndarray):
            img_numpy = img
        else:
            raise ValueError(f"Image type {type(img)} not supported: only torch.Tensor images are supported")
        pass

        ####################################
        # Normalize the image, if necessary
        ####################################

        img_numpy_normalized = self.cieluv_normalizer(img_numpy)

        ####################################
        # Transform to the requested RGB type
        ####################################

        img_numpy_output = color.luv2rgb(img_numpy_normalized, channel_axis=-1)
        if self.int_output:
            img_numpy_output = (img_numpy_output * 255.0).astype(np.uint8)
        pass

        ####################################
        # Back to tensor, if necessary, and back to list/tuple, if necessary
        ####################################

        img_output = img_numpy_output if not flag_original_image_is_tensor else \
            torch.from_numpy(img_numpy_output).movedim(source=-1, destination=-3).to(torch.float32)

        sample_output = None
        if not flag_sample_is_list_tuple:
            sample_output = img_output
        else:
            sample_output = [img_output] + sample[1:]
            if isinstance(sample, tuple):
                sample_output = tuple(sample_output)
            pass
        pass

        return sample_output


class NormalizeRGB(v2.Transform):
    """
    Normalize **according to the CIELuv** color space an image in RGB format but keep the result in RGB: the procedure \
    is therefore RGB->CIELuv->Normalization->RGB.

    The behavior of the function differs slightly depending on the format of the data passed as input:

    - Regarding whether the input is only an image or batch of images or a list/tuple wherein \
        the first element is an image:
        - The image/batch of images is transformed
        - and the result is returned as it was input: if list/tuple, a list with the non-image \
          elements untouched.

    - Regarding the format of the image:
        - If the input is a numpy array, the image is returned as a numpy array.
        - If the input is a torch.Tensor, the image is returned as a torch.Tensor.
        - In both cases: **the input image is always interpreted as a float in the range [0, 1]**, even \
            if its data type is an integer. If its value is greater than 1.0 or smaller than 0.0 there will be an error.
        - In both cases: if the input is monochromatic (1 channel), it is converted to 3 channels by replicating the \
            single channel 3 times.

    Additionally, different CIELuv normalization types can be applied to **each** single image, \
    if desired, which will have an effect on the resulting RGB images (*Note*: the channels of the \
    CIELuv color space are located, respectively, in the ranges [0,100], [-134,220], and [-140,122]):
        - ``None`` or ``'none'``: the image is left untouched
        - ``'L'``: each image is standardized so its L channel is centered at 50, the theoretical average of the \
            L channel, and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping.
        - ``'Luv'``: each image is standardized so its L channel is centered at 50 (general average of the L channel) \
            and has a standard deviation of 50/3, so 99.7% of the samples will not need clipping; the u and v channels \
            are centered at 0 and have a standard deviation such that 3 standard deviations both sides of 0 fit inside \
            the theoretical range of the channel.
    The normalization is performed by an object of the class :py:class:`.NormalizeCIELuv` contained as an attribute.
    """

    def __init__(self, normalization_type=None):
        #
        super().__init__()
        #
        self.rbg2cieluv_with_normalizer = RGB2CIELuv(normalization_type=normalization_type)
        self.cieluv2rgb_backtransform = CIELuv2RGB(normalization_type=None)

    def forward(self, sample):
        """
        Normalize RGB image(s) according to the stated 'normalization_type'.

        Parameters
        ----------
        sample : np.ndarray or torch.Tensor or list or tuple
            The image to be transformed, or a list/tuple wherein the first element is the image to be transformed

        Returns
        -------
        np.ndarray or torch.Tensor or list or tuple
            The transformed image, or a list/tuple wherein the first element is the transformed image
        """

        ####################################
        # List/tuple or single image? Extract the image
        ####################################

        img = None
        flag_sample_is_list_tuple = False
        if isinstance(sample, (np.ndarray, torch.Tensor)):
            img = sample
            flag_sample_is_list_tuple = False
        elif isinstance(sample, (list, tuple)):
            img = sample[0]
            flag_sample_is_list_tuple = True
        else:
            raise ValueError(
                f"Sample type {type(sample)} not supported: only np.ndarray, torch.Tensor or list/tuple are supported")
        pass

        ####################################
        # Convert the images using the already created transforms
        ####################################

        img_output = self.cieluv2rgb_backtransform(self.rbg2cieluv_with_normalizer(img))

        ####################################
        # Back to list/tuple, if necessary
        ####################################

        sample_output = None
        if not flag_sample_is_list_tuple:
            sample_output = img_output
        else:
            sample_output = [img_output] + sample[1:]
            if isinstance(sample, tuple):
                sample_output = tuple(sample_output)
            pass
        pass

        return sample_output


#############################################################################################
# DICTIONARY WITH GENERAL INFORMATION AND LOADING FUNCTION FOR THE CONSIDERED DATASETS FOR UNIFIED PROCESSING
#############################################################################################

_dict_dataset_info_and_constructor = {
    'mnist': {
        'channels': 1, 'default_im_size': (28, 28), 'classes': 10,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.MNIST(root=root_folder, train=True, download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.MNIST(root=root_folder, train=False, download=True, transform=im_transform),
        'label_denomination': 'targets',
    },
    'fashion-mnist': {
        'channels': 1, 'default_im_size': (28, 28), 'classes': 10,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.FashionMNIST(root=root_folder, train=True, download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.FashionMNIST(root=root_folder, train=False, download=True, transform=im_transform),
        'label_denomination': 'targets',
    },
    'svhn': {
        'channels': 3, 'default_im_size': (32, 32), 'classes': 10,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.SVHN(root=root_folder, split='train', download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.SVHN(root=root_folder, split='test', download=True, transform=im_transform),
        'label_denomination': 'labels',
    },
    'cifar10': {
        'channels': 3, 'default_im_size': (32, 32), 'classes': 10,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.CIFAR10(root=root_folder, train=True, download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.CIFAR10(root=root_folder, train=False, download=True, transform=im_transform),
        'label_denomination': 'targets',
    },
    'cifar100': {
        'channels': 3, 'default_im_size': (32, 32), 'classes': 100,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.CIFAR100(root=root_folder, train=True, download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.CIFAR100(root=root_folder, train=False, download=True, transform=im_transform),
        'label_denomination': 'targets',
    },
    'food101': {
        'channels': 3, 'default_im_size': (512, 512), 'classes': 101,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.Food101(root=root_folder, split='train', download=True, transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.Food101(root=root_folder, split='test', download=True, transform=im_transform),
        'label_denomination': None,
    },
    'places365_small': {
        'channels': 3, 'default_im_size': (256, 256), 'classes': 434,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.Places365(root=root_folder, split='train-standard', small=True,
                               download=not os.path.exists(os.path.join(root_folder, 'data_256_standard')),
                               transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.Places365(root=root_folder, split='val', small=True,
                               download=not os.path.exists(os.path.join(root_folder, 'data_256_standard')),
                               transform=im_transform),
        'label_denomination': None
    },
    'tiny-imagenet': {
        'channels': 3, 'default_im_size': (64, 64), 'classes': 200,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            (prepare_tiny_imagenet(root_folder),
             datasets.ImageFolder(os.path.join(root_folder, 'tiny-imagenet-200/train'), transform=im_transform))[1],
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.ImageFolder(os.path.join(root_folder, 'tiny-imagenet-200/val'), transform=im_transform),
        'label_denomination': 'targets',
    },
    'imagenet': {
        'channels': 3, 'default_im_size': (224, 224), 'classes': 1000,
        'simplified_train_data_constructor': lambda root_folder, im_transform: \
            datasets.ImageNet(root=root_folder, split='train', transform=im_transform),
        'simplified_test_data_constructor': lambda root_folder, im_transform: \
            datasets.ImageNet(root=root_folder, split='val', transform=im_transform),
        'label_denomination': 'targets',
    },
}


#############################################################################################
# DICTIONARY WITH GENERAL INFORMATION AND LOADING FUNCTION FOR THE CONSIDERED DATASETS FOR UNIFIED PROCESSING
#############################################################################################

#############################################################################################
# ACCESSORY FUNCTIONS FOR DATA LOADING
#############################################################################################

def _get_min_max_within_image_from_dataloader(dataloader, device=None):
    """
    Calculates the per-channel worst-case min and max across images in the dataloader.
    For each channel:
        - min = maximum of per-image minimums
        - max = minimum of per-image maximums

    Parameters
    ----------
    dataloader : DataLoader
        A PyTorch dataloader yielding batches of images.
    device : torch.device or str, optional
        Device to use for computation. Defaults to CUDA if available.

    Returns
    -------
    worst_min, worst_max : torch.Tensor (on CPU)
        1D tensors containing the worst-case min and max for each channel.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    worst_min, worst_max = None, None

    if dataloader is not None and len(dataloader) > 0:
        worst_min, worst_max = None, None
        for data, _ in dataloader:
            data = data.to(device)

            channels_min, _ = data.movedim(-3, 0).flatten(1).min(dim=1)
            channels_max, _ = data.movedim(-3, 0).flatten(1).max(dim=1)
            worst_min = channels_min if worst_min is None else torch.maximum(worst_min, channels_min)
            worst_max = channels_max if worst_max is None else torch.minimum(worst_max, channels_max)

    # Move to CPU for return
    if worst_min is not None and worst_max is not None:
        worst_min = worst_min.cpu()
        worst_max = worst_max.cpu()

    return worst_min, worst_max


def _get_mean_std_from_dataloader(dataloader, device=None):
    """
    This function calculates the mean and std for each of the channels of the images in the dataloader, using GPU if available.

    Parameters
    ----------
    dataloader : DataLoader
    device : torch.device or str (optional)
        The device on which to perform computations. If None, uses CUDA if available.

    Returns
    -------
    mean, std : 1D Tensor with as many elements as channels in the dataset of the dataloader
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = None, None
    if dataloader is not None and len(dataloader) > 0:
        sum_image_avg_sep_channels = 0
        sum_squared_image_avg_sep_channels = 0
        num_images = 0

        for data, _ in dataloader:
            data = data.to(device)
            num_images_batch = data.size(0)  # Batch size (number of images)

            # Mean over height and width (H, W), per image and per channel
            per_image_mean = torch.mean(data, dim=[2, 3])  # shape: [B, C]
            per_image_squared_mean = torch.mean(data ** 2, dim=[2, 3])  # shape: [B, C]

            sum_image_avg_sep_channels += per_image_mean.sum(dim=0)  # sum across batch
            sum_squared_image_avg_sep_channels += per_image_squared_mean.sum(dim=0)
            num_images += num_images_batch

        mean = sum_image_avg_sep_channels / num_images
        std = (sum_squared_image_avg_sep_channels / num_images - mean ** 2).sqrt()

        # Move to CPU for return
        if mean is not None and std is not None:
            mean = mean.cpu()
            std = std.cpu()

    return mean, std


#############################################################################################
# DATA-LOADING FUNCTIONS
#############################################################################################

class LoadedDatasetDict(dict):
    """
    Subclass of :py:class:`dict` intended to contain all the information regarding a loaded dataset. \
    It contains exactly the following fields (with `None` if not provided with a correct value):

    - `'dataset_name'`: :py:class:`str`

    - `'proportion_train'`, `'proportion_val'`: :py:class:`float`

    - `'proportion_mislabel'`: :py:class:`float`

    - `'generator_seed'`: :py:class:`int`

    - `'num_workers'`: :py:class:`int`

    - `'batch_size'`: :py:class:`int`

    - `'num_batches_train'`, `'num_batches_val'`, `'num_batches_test'`: :py:class:`int`

    - `'dataloader_train'`, `'dataloader_val'`, `'dataloader_test'`: :py:class:`torch.utils.data.DataLoader`

    - `'im_height'`: :py:class:`int`

    - `'im_width'`: :py:class:`int`

    - `'channels'`, `'classes'`: :py:class:`int`

    - `'original_mean_train'`, `'original_std_train'`: :py:class:`torch.Tensor`

    - `'final_mean_train'`, `'final_std_train'`: :py:class:`torch.Tensor`

    - `'original_min_train'`, `'original_max_train'`: :py:class:`torch.Tensor`

    - `'final_min_train'`, `'final_max_train'`: :py:class:`torch.Tensor`

    - `'forced_bn'`: :py:class:`bool`

    - `'colorspace'`, `'source_normalization'`: :py:class:`str`

    - `'normalized_dataset'`: :py:class:`str`

    - `'tuple_of_pairs_other_kwargs'`: ideally it would be :py:class:`dict`, but for hashability, \
       it is a :py:class:`tuple` (composed of 2D tuples `(key, value)`) containing any other keyword arguments
    """
    _keys_and_types = {
        'dataset_name': str,
        'proportion_train': float, 'proportion_val': float,
        'proportion_mislabel': float,
        'generator_seed': int,
        'num_workers': int,
        'batch_size': int, 'num_batches_train': int, 'num_batches_val': int, 'num_batches_test': int,
        'dataloader_train': torch.utils.data.DataLoader,
        'dataloader_val': torch.utils.data.DataLoader,
        'dataloader_test': torch.utils.data.DataLoader,
        'channels': int, 'im_height': int, 'im_width': int, 'classes': int,
        'original_mean_train': torch.Tensor, 'original_std_train': torch.Tensor,
        'final_mean_train': torch.Tensor, 'final_std_train': torch.Tensor,
        'original_min_train': torch.Tensor, 'original_max_train': torch.Tensor,
        'final_min_train': torch.Tensor, 'final_max_train': torch.Tensor,
        'forced_bn': bool,  # forced black-and-white
        'colorspace': str, 'source_normalization': str,
        'normalized_dataset': str,
        'tuple_of_pairs_other_kwargs': tuple
    }

    def __init__(self):
        super().__init__()
        for key in self._keys_and_types:
            self[key] = None

    def __setitem__(self, key, val):
        if key not in self._keys_and_types:
            raise KeyError(f"Attempted key '{key}' is not part of class LoadedDatasetDict !")
        elif (val is not None) and (not isinstance(val, self._keys_and_types[key])):
            raise TypeError(f"Value of type {type(val)}: key '{key}' requires {self._keys_and_types[key]} instead!")
        pass
        dict.__setitem__(self, key, val)

    def dict_to_metrics_fields(self):
        """
        Returns a copy of the current :py:class:`.LoadedDatasetDict` instance but containing only those parameters \
        that can be logged as a metric, that is, float or int.

        Returns
        -------
        dict
        """

        metric_able_dict = {}
        for key in self.keys():
            value = self[key]
            if isinstance(value, (float, int)):
                metric_able_dict[key] = copy.deepcopy(value)
        pass
        #
        return metric_able_dict

    def dict_wo_dataloaders(self):
        """
        Returns a copy of the current :py:class:`.LoadedDatasetDict` instance but without the dataloaders, \
        so that it can be hashed or compared.

        Returns
        -------
        dict
        """

        simplified_dataset_dict = {}
        for key in self.keys():
            if not 'dataloader_' in key:
                simplified_dataset_dict[key] = copy.deepcopy(self[key])
            pass
        pass
        #
        return simplified_dataset_dict



def obtain_classification_dataset_loaders(dataset_name, desired_im_size=None,
                                          loaded_im_colorspace='rgb', loaded_im_normalization=None,
                                          normalization=None, batch_size:int=100,
                                          train_proportion:float=0.9, val_proportion:float=0.1,
                                          mislabeled_proportion:float=None,
                                          num_workers=None, shuffle=True, generator_seed=None,
                                          root_folder='../data', verbose='medium'):
    """
    This function loads the dataset indicated by the argument `dataset_name` using the options indicated in the rest \
    arguments, returning a dictionary containing, among other informative fields of the dataset, \
    :py:class:`torch.utils.data.DataLoader` for subsets *train*, *val(idation)*, and *test*.

    Both *train* and *val(idation)* subsets are obtained by sampling from the same training set of the selected \
    dataset, according to the respective proportions `train_proportion`and `val_proportion`, which must be, \
    added together, lower than `1.0`. The sampling is performed \
    using the function :py:func:`torch.utils.data.random_split` which, for the same proportions and the same \
    `generator_seed`, generates consistently the same separation between *train* and *val(idation)*.

    Parameters
    ----------
    dataset_name : str
        Value among the considered `'mnist'` (:py:class:`~torchvision.datasets.MNIST`), \
        `'fashion-mnist'` (:py:class:`~torchvision.datasets.FashionMNIST`), \
        `'svhn'` (:py:class:`~torchvision.datasets.SVHN`), \
        `'cifar10'` (:py:class:`~torchvision.datasets.CIFAR10`), \
        `'cifar100'` (:py:class:`~torchvision.datasets.CIFAR100`), \
        `'food101'` (:py:class:`~torchvision.datasets.Food101`), \
        `'places365_small'` (:py:class:`~torchvision.datasets.Places365`, option `small`=`True)
    desired_im_size : tuple[int], optional
        Default: `None`
    loaded_im_colorspace : str, optional
        Requested colorspace, value among ``'gray'``, ``'rgb'`` and ``'cieluv'``. \
        If the input dataset has 1 channel and the requested colorspace has 3 (``'RGB'`` and ``'CIELuv'``) the input \
        will be interpreted as gray and transformed into RGB values corresponding to gray (i.e. $g\\to\\g(1,1,1)$). \
        If the input dataset has 3 channels it would be interpreted as RGB: from it \
        the images are transformed to the indicated colorspace.
        Default: ``rgb``
    loaded_im_normalization : str, optional
        Value among ``None``, ``'l'``, and ``'luv'``, indicating whether per-image normalization is performed \
        in the loaded ``loaded_im_colorspace`` (``'rgb'`` or ``'cieluv'``) images, or only the ``'l'`` channel \
        or all L, u, and v (for ``'luv'``) channels where normalized at each image.
        Default: ``None``
    normalization : str, optional
        Value among ``'min_max_01_per_im'``, ``'N(0;1)'``, or ``None``. \
        It indicates whether each channel is (separately) normalized to zero mean and unitary standard deviation \
        **across the complete dataset**, in the case of `'N(0;1)'``, or normalized to 0 and 1 as their min-max values \
        **independently per image**, in the case of ``'min_max_01_per_im'``, \
        based on the statistics of the *train* subset and applied to all three *train*, *val(idation)*, and *test*.
        Default: `None`
    batch_size : int, optional
        Default: `100`
    train_proportion : float, optional
        Default: `0.9`
    val_proportion : float, optional
        Default: `0.1`
    mislabeled_proportion : float, optional
        Proportion of the train subdataset that is mislabeled: instead of the correct labels, *any* of the other \
        labels is assigned to it.
        Default: ``None`` (= ``0.0``)
    num_workers : int, optional
        Number of workers to be used in the loaders. Default: ``None`` (which means ``4``)
    shuffle : bool, optional
        If `True`, the multiple obtained :py:class:`~torch.utils.data.DataLoader` in the returned \
        :py:class:`.LoadedDatasetDict` are reshuffled at every epoch (that is, every time they are accessed from the \
        beginning as an iterable). In order to keep the batch and image order at each epoch access this \
        argument should be set to `False`. \
        Default: `True`
    generator_seed : int, optional
        Default: `None`
    root_folder : str, optional
        Folder of download of the dataset. Default: `'../../data'`
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of printing the training progress. \
        Default: ``'medium'``

    Returns
    -------
    LoadedDatasetDict

        Dictionary subclass :py:class:`.LoadedDatasetDict`
    """

    # Default values if not provided or none
    default_num_workers = 4

    # print(f"\n\n")
    # print(f"root_folder =       {root_folder}")
    # print(f"root_folder (ABS) = {os.path.abspath(root_folder)}")
    # print(f"\n\n")

    ################################################################
    # Initial checks
    ################################################################
    #
    assert loaded_im_colorspace is not None and isinstance(loaded_im_colorspace, str), \
        f"Expected 'loaded_im_colorspace' to be a string, received: {type(loaded_im_colorspace)}"
    loaded_im_colorspace = loaded_im_colorspace.lower()
    #
    assert loaded_im_colorspace in ['gray', 'rgb', 'cieluv'], \
        f"Expected 'loaded_im_colorspace' to be 'gray', 'rgb', or 'cieluv', received: {loaded_im_colorspace}"
    #
    assert loaded_im_normalization is None or isinstance(loaded_im_normalization, str), \
        f"Expected 'loaded_im_normalization' to be None or a string, received: {type(loaded_im_normalization)}"
    loaded_im_normalization = loaded_im_normalization.lower() if isinstance(loaded_im_normalization, str) else None
    #
    assert loaded_im_normalization in [None, 'l', 'luv'], \
        f"Expected 'loaded_im_normalization' to be None, 'l', or 'luv', received: {loaded_im_normalization}"
    ################################################################
    assert isinstance(train_proportion, (float, int)) and 0.0 <= train_proportion <= 1.0, \
        f"Expected 'train_proportion' to be a float in [0.0, 1.0], received: {train_proportion}"
    train_proportion = float(train_proportion)
    #
    if val_proportion is None:
        val_proportion = 1.0 - train_proportion
    else:
        assert isinstance(val_proportion, (float, int)) and 0.0 <= val_proportion <= 1.0, \
            f"Expected 'val_proportion' to be a float in [0.0, 1.0], received: {val_proportion}"
    pass
    val_proportion = float(val_proportion)
    #
    assert (train_proportion + val_proportion) <= 1.0, \
        f"Expected 'train_proportion' + 'val_proportion' <= 1.0, received: {train_proportion + val_proportion}"
    ################################################################
    mislabeled_proportion = 0.0 if mislabeled_proportion is None else mislabeled_proportion
    assert isinstance(mislabeled_proportion, (float,int)) and 0.0 <= mislabeled_proportion <= 1.0, \
        f"Expected 'mislabeled_proportion' to be a float/int in [0.0, 1.0], received: {mislabeled_proportion}"
    mislabeled_proportion = float(mislabeled_proportion)
    ################################################################
    if num_workers is None:
        num_workers = default_num_workers
    assert isinstance(num_workers, int) and num_workers >= 0, \
        f"Expected 'num_workers' to be a non-negative integer, received: {num_workers}"
    ################################################################
    assert isinstance(shuffle, bool), f"Expected 'shuffle' to be a boolean, received: {type(shuffle)}"
    ################################################################

    # Parameters of the dataset as default in Pytorch and the desired parameters

    dataset_info_and_constructor = _dict_dataset_info_and_constructor[dataset_name]
    #
    default_channels = dataset_info_and_constructor['channels']
    default_im_colorspace = 'gray' if default_channels == 1 else 'rgb'
    loaded_channels = 1 if loaded_im_colorspace == "gray" else 3
    #
    im_size = dataset_info_and_constructor['default_im_size'] if desired_im_size is None else desired_im_size
    loaded_im_size = im_size
    #
    loaded_classes = dataset_info_and_constructor['classes']

    # Create the required transformations for the dataset

    im_transform = None  # If 'normalization' is selected 2 different runs with different transforms will be issued
    original_mean_train, original_std_train = None, None
    final_mean_train, final_std_train = None, None
    original_min_train, original_max_train = None, None
    final_min_train, final_max_train = None, None

    # Create an object of the dict-subclass LoadedDatasetDict
    dataset_dict = LoadedDatasetDict()

    im_transform = None
    for i in ['before_normalization', 'with_normalization']:
        #
        if i == 'before_normalization':
            #
            ##############################
            # FIRST PASS, before normalization: the image transform performs color conversion, and maybe within-image \
            # normalization, but not across the dataset normalization. The image transform will be reused/complemented \
            # if a second pass is performed with normalization (adding, only the normalization).
            ##############################
            #
            if verbose in ['medium', 'high']:
                print(f"Dataset loading and transformation before normalization ...")
            pass
            #
            if loaded_im_normalization is None:
                if loaded_im_colorspace == 'gray':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.Grayscale(), v2.Resize(size=im_size)])
                elif loaded_im_colorspace == 'rgb':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.RGB(), v2.Resize(size=im_size)])
                elif loaded_im_colorspace == 'cieluv':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.RGB(), RGB2CIELuv(), v2.Resize(size=im_size)])
                else:
                    raise Exception(
                        f"Expected 'loaded_im_colorspace': 'gray', 'rgb', or 'cieluv', received: {loaded_im_colorspace}"
                    )
                pass
            elif loaded_im_normalization in ['l', 'luv']:
                if loaded_im_colorspace == 'gray':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.RGB(), NormalizeRGB(loaded_im_normalization),
                                               v2.Grayscale(), v2.Resize(size=im_size)])
                elif loaded_im_colorspace == 'rgb':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.RGB(), NormalizeRGB(loaded_im_normalization),
                                               v2.Resize(size=im_size)])
                elif loaded_im_colorspace == 'cieluv':
                    im_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                               v2.RGB(), RGB2CIELuv(loaded_im_normalization),
                                               v2.Resize(size=im_size)])
                else:
                    raise Exception(
                        f"Expected 'loaded_im_colorspace': 'gray', 'rgb', or 'cieluv', received: {loaded_im_colorspace}"
                    )
                pass
            pass
            #
        else:  # if 'with_normalization'
            #
            ##############################
            # SECOND PASS, if requested
            ##############################
            #
            if normalization is None:
                break
            else:
                if verbose in ['medium', 'high']:
                    print(f"Dataset loading and transformation after {normalization} normalization ...")
                pass
                if normalization == 'N(0;1)':
                    im_transform = v2.Compose([im_transform, v2.Normalize(original_mean_train, original_std_train)])
                elif normalization == 'min_max_01_per_im':
                    im_transform = v2.Compose([im_transform, autocontrast])
                else:
                    raise Exception((f"Expected 'normalization': 'N(0;1)', 'min_max_01_per_im', or None, " +
                                     f"received: {normalization}"))
                pass
            pass
        pass

        if verbose == 'high':
            print(f"\tDownloading the training (train+val) set ...  ", end='')
        pass
        total_train_data = dataset_info_and_constructor['simplified_train_data_constructor'](root_folder,
                                                                                             im_transform)
        if verbose == 'high':
            print("")
            print(f"\tDownloading the test subset ...  ", end='')
        pass
        test_data = dataset_info_and_constructor['simplified_test_data_constructor'](root_folder, im_transform)

        # Separation of 'total_tra in_data' into 'train_data' and 'val_data'
        added_proportion = train_proportion + val_proportion
        if added_proportion > 1.0:
            train_proportion = train_proportion / added_proportion
            val_proportion = val_proportion / added_proportion
        pass
        remaining_proportion = max(1.0-train_proportion-val_proportion, 0.0)

        manual_generator = None if generator_seed is None else torch.Generator().manual_seed(int(generator_seed))
        if verbose == 'high':
            print("")
            print(f"\tSplit of the training (train+val) set into train and val ...  ", end='')
        pass

        train_data, val_data, _ = random_split(
            total_train_data, [train_proportion, val_proportion, remaining_proportion],
            generator=manual_generator
        )

        ##################
        ####################################
        ######################################################
        ########################################################################
        # Mislabel, if so requested, a number of labels of the dataset!
        ########################################################################

        # Extract the (true) numeric labels of the 'train_data' subdataset where labels are contained in this dataset
        key_name_for_labels_in_dataset = dataset_info_and_constructor['label_denomination']
        labels = getattr(train_data.dataset, key_name_for_labels_in_dataset)
        # Extract the indices of the "list_of_true_labels" which really correspond to the train subdataset
        # (others belong to the val subdataset (and to the dropped data)
        indices_belonging_to_train_data = train_data.indices

        # Choose a subset of this 'indices_belonging_to_train_data' indicating the labels to mislabel!
        indices_for_mislabeling = torch.tensor(indices_belonging_to_train_data)[
            torch.randperm(
                len(indices_belonging_to_train_data), generator=manual_generator
            )[0:round(mislabeled_proportion*len(indices_belonging_to_train_data))]
        ].tolist()

        # Create a "copy" of labels (in the needed format) for modification...
        new_labels = labels.detach().clone() if isinstance(labels, torch.Tensor) else copy.deepcopy(labels)
        # ... and mislabel them!
        for index_for_mislabeling in indices_for_mislabeling:
            # Extract the true numeric label for the index
            true_label = labels[index_for_mislabeling]
            # Get ANY other
            new_wrong_label = \
                (true_label + torch.randint(1, loaded_classes, size=(1,), generator=manual_generator).item()) \
                % loaded_classes
            # Modify the label
            new_labels[index_for_mislabeling] = new_wrong_label
            # # Check
            # checked_label = new_labels[index_for_mislabeling]
            # print("")
        pass

        # Force the new version of the labels into the attribute of the object "train_data.dataset"
        setattr(train_data.dataset, key_name_for_labels_in_dataset, new_labels)

        # CHECKS TO DELETE
        #
        # print(f"index_for_mislabeling = {index_for_mislabeling}", flush=True)
        # print(f"labels[index_for_mislabeling] = {labels[index_for_mislabeling]}", flush=True)
        # print(f"Written labels[index_for_mislabeling] = {getattr(train_data.dataset, key_name_for_labels_in_dataset)[index_for_mislabeling]}", flush=True)
        #
        # print(f"NO index_for_mislabeling = {index_for_mislabeling+1}", flush=True)
        # print(f"labels[index_for_mislabeling] = {labels[index_for_mislabeling+1]}", flush=True)
        # print(f"Written labels[index_for_mislabeling] = {getattr(train_data.dataset, key_name_for_labels_in_dataset)[index_for_mislabeling+1]}", flush=True)


        ########################################################################
        ######################################################
        ####################################
        ##################

        # Pack the resulting dataloaders and info
        bs = batch_size
        if verbose == 'high':
            print("")
            print(f"\tCreation of the Dataloaders for train, val, and test ...")
        pass
        #
        dataset_dict['dataset_name'] = dataset_name
        dataset_dict['proportion_train'] = train_proportion
        dataset_dict['proportion_val'] = val_proportion
        dataset_dict['proportion_mislabel'] = mislabeled_proportion
        dataset_dict['generator_seed'] = generator_seed
        #
        dataset_dict['batch_size'] = batch_size
        #
        dataset_dict['num_workers'] = num_workers
        #
        dataset_dict['dataloader_train'] = \
            DataLoader(train_data, batch_size=bs, shuffle=shuffle, num_workers=num_workers) if len(train_data) else None
        dataset_dict['num_batches_train'] = len(dataset_dict['dataloader_train']) \
            if dataset_dict['dataloader_train'] is not None else 0
        dataset_dict['dataloader_val'] = \
            DataLoader(val_data, batch_size=bs, shuffle=shuffle, num_workers=num_workers) if len(val_data) else None
        dataset_dict['num_batches_val'] = len(dataset_dict['dataloader_val']) \
            if dataset_dict['dataloader_val'] is not None else 0
        dataset_dict['dataloader_test'] = \
            DataLoader(test_data, batch_size=bs, shuffle=shuffle, num_workers=num_workers) if len(test_data) else None
        dataset_dict['num_batches_test'] = len(dataset_dict['dataloader_test']) \
            if dataset_dict['dataloader_test'] is not None else 0
        #
        dataset_dict['channels'] = loaded_channels
        dataset_dict['colorspace'] = loaded_im_colorspace
        dataset_dict['im_height'] = loaded_im_size[-2]
        dataset_dict['im_width'] = loaded_im_size[-1]
        dataset_dict['classes'] = loaded_classes
        #
        # Calculate the mean and std for normalization
        if i == 'before_normalization':
            if verbose in ['medium', 'high']:
                print(f"Calculation of dataset statistics ...")
            pass
            # Calculation of the mean and std of the 'train' subset:
            original_mean_train, original_std_train = _get_mean_std_from_dataloader(
                dataset_dict['dataloader_train'])
            # Calculation of the min and max of the 'train' subset:
            original_min_train, original_max_train = _get_min_max_within_image_from_dataloader(
                dataset_dict['dataloader_train'])
        pass
    pass

    # Calculation of the mean and std of the 'train' subset after normalization
    final_mean_train, final_std_train = original_mean_train, original_std_train
    final_min_train, final_max_train = original_min_train, original_max_train
    if normalization is None:
        pass
        # final_mean_train, final_std_train = _get_mean_std_from_dataloader(dataset_dict['dataloader_train'])
        # final_min_train, final_max_train = _get_min_max_within_image_from_dataloader(
        #     dataset_dict['dataloader_train'])
    else:
        if verbose in ['medium', 'high']:
            print(f"Calculation of dataset statistics ...")
            final_mean_train, final_std_train = _get_mean_std_from_dataloader(dataset_dict['dataloader_train'])
            final_min_train, final_max_train = _get_min_max_within_image_from_dataloader(
                dataset_dict['dataloader_train'])
        pass
    pass

    dataset_dict['original_mean_train'] = original_mean_train
    dataset_dict['original_std_train'] = original_std_train
    dataset_dict['final_mean_train'] = final_mean_train
    dataset_dict['final_std_train'] = final_std_train
    dataset_dict['original_min_train'] = original_min_train
    dataset_dict['original_max_train'] = original_max_train
    dataset_dict['final_min_train'] = final_min_train
    dataset_dict['final_max_train'] = final_max_train
    dataset_dict['colorspace'] = loaded_im_colorspace if loaded_im_colorspace is not None else 'unknown'
    dataset_dict['source_normalization'] = loaded_im_normalization if loaded_im_normalization is not None else 'no'
    dataset_dict['normalized_dataset'] = normalization if normalization is not None else 'no'
    dataset_dict['tuple_of_pairs_other_kwargs'] = ()

    return dataset_dict



def prepare_tiny_imagenet(root_folder):
    """
    Check if Tiny-ImageNet is already downloaded and download it if necessary.
    Also reorganizes the validation directory to make it compatible with ImageFolder.

    Parameters
    ----------
    root_folder : str
        Path where the dataset will be downloaded/verified
    """


    tiny_imagenet_dir = os.path.join(root_folder, 'tiny-imagenet-200')

    # If directory exists, assume dataset is downloaded
    if os.path.exists(tiny_imagenet_dir):
        print("Tiny-ImageNet already downloaded.")
        return

    # Download URL for Tiny-ImageNet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file = os.path.join(root_folder, "tiny-imagenet-200.zip")

    # Download the file with progress bar
    print("Downloading Tiny-ImageNet...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Tiny-ImageNet") as t:
        urllib.request.urlretrieve(url, zip_file, reporthook=t.update_to)

    # Extract the file
    print("Extracting archive...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(root_folder)

    # Reorganize validation directory to work with ImageFolder
    val_dir = os.path.join(tiny_imagenet_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    # If annotations file exists, reorganize validation images
    if os.path.exists(val_annotations_file):
        print("Reorganizing validation set...")

        # Create directories for each class
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.split('\t')
                img_name, class_id = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_id)

                # Create directory for class if it doesn't exist
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Move image to its class directory
                src_path = os.path.join(val_images_dir, img_name)
                dst_path = os.path.join(class_dir, img_name)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)

        # Remove images directory and annotations file after reorganizing
        if os.path.exists(val_images_dir):
            shutil.rmtree(val_images_dir)
        if os.path.exists(val_annotations_file):
            os.remove(val_annotations_file)

    # Remove the zip file to save space
    os.remove(zip_file)
    print("Tiny-ImageNet downloaded and organized successfully.")


###################################################################################################
# DataLoader setting and LoadedDatasetDict creation
###################################################################################################


_list_dataset_names_for_create_sklearn_dataset_for_2D_analysis = [
    'moons', 'circles', 'blobs', 'classification'
]


#############################################################################################


def _create_sklearn_dataset_for_2D_analysis(
        dataset_name, num_points_training, num_points_test, generator_seed=None,
        **kwargs_for_point_generation
):
    """
    Create a classification dataset from the indicated point data from Sklearn.

    Parameters
    ----------
    dataset_name : str
        Type of dataset to create. Options are: 'moons', 'circles', 'blobs', 'classification'.
    num_points_training : int
        Total number of points to generate, corresponding to the "training + validation" set
    num_points_test : int
        Number of points to generate for the test set
    generator_seed : int, optional
        Default: `None`
    kwargs_for_point_generation : dict, optional
        Additional keyword arguments for the point generation function from `sklearn.datasets`. E.g.: for the \
        ``'moons'`` type, one can provide the `noise` argument to add Gaussian noise to the data.

    Returns
    -------
    dict

    The dictionary will contain the following entries:
    'channels': int, 'default_im_size': 2-tuple, 'classes': int,
    'simplified_train_dataset' and 'simplified_test_dataset': torch.utils.data.Dataset
    """

    # Check the validity of common parameters
    assert isinstance(num_points_training, int) and num_points_training > 0, \
        f"'num_points_training' must be a positive integer; got {num_points_training}."
    assert isinstance(num_points_test, int) and num_points_test > 0, \
        f"'num_points_test' must be a positive integer; got {num_points_test}."
    assert generator_seed is None or isinstance(generator_seed, int), \
        f"'generator_seed' must be None or an integer; got {generator_seed}."

    ###############################################
    # Dictionary to return: EMPTY DEFINITION
    ###############################################
    dataset_info_and_constructor = {'channels': 1, 'default_im_size': None, 'classes': None,
                                    'simplified_train_dataset': None, 'simplified_test_dataset': None,
                                    'other_kwargs': {}}
    ###############################################

    # Generate the points
    make_function_from_sklearn = {}
    kwargs_for_skilearn = {}

    # Check type and the provided options
    if dataset_name == 'moons':
        #
        dataset_info_and_constructor['default_im_size'] = (2, 1)
        #
        dataset_info_and_constructor['classes'] = 2
        #
        make_function_from_sklearn = sklearn.datasets.make_moons
        kwargs_for_skilearn = {
            'noise': kwargs_for_point_generation.get('noise', 0.0)
        }
        #
    elif dataset_name == 'circles':
        #
        dataset_info_and_constructor['default_im_size'] = (2, 1)
        #
        dataset_info_and_constructor['classes'] = 2
        #
        make_function_from_sklearn = sklearn.datasets.make_circles
        kwargs_for_skilearn = {
            'noise': kwargs_for_point_generation.pop('noise', 0.0),
            'factor': kwargs_for_point_generation.pop('factor', 0.5)
        }
        #
    elif dataset_name == 'blobs':
        #
        n_features = kwargs_for_point_generation.pop('n_features', 2)
        dataset_info_and_constructor['default_im_size'] = (n_features, 1)
        #
        centers = kwargs_for_point_generation.pop('centers', 3)
        dataset_info_and_constructor['classes'] = centers
        #
        make_function_from_sklearn = sklearn.datasets.make_blobs
        kwargs_for_skilearn = {
            'n_features': n_features,
            'centers': centers,
            'cluster_std': kwargs_for_point_generation.pop('cluster_std', 1.0)
        }
        # SPECIAL FOR 'blobs': we have to add the option 'return_centers' because for 'test' we have to reuse the
        # same centers used in training so we can use a different seed without altering them!!!
        kwargs_for_skilearn['return_centers'] = True
    elif dataset_name == 'classification':
        #
        n_features = kwargs_for_point_generation.pop('n_features', 2)
        dataset_info_and_constructor['default_im_size'] = (n_features, 1)
        #
        n_classes = kwargs_for_point_generation.pop('n_classes', 2)
        dataset_info_and_constructor['classes'] = n_classes
        #
        make_function_from_sklearn = sklearn.datasets.make_classification
        kwargs_for_skilearn = {'n_features': n_features, 'n_classes': n_classes}
        for key in kwargs_for_point_generation:
            if key not in kwargs_for_skilearn:
                kwargs_for_skilearn[key] = kwargs_for_point_generation[key]
            pass
        pass
    else:
        raise ValueError(f"Data type {dataset_name} not recognized. Options are: 'moons', 'circles', 'blobs', 'classification'.")
    pass

    # Generate the points
    dict_num_points = {'training': num_points_training, 'test': num_points_test}
    dict_points = {}
    dict_labels = {}
    dict_tensor_points = {}
    dict_tensor_labels = {}

    for key in dict_num_points:
        returned_info = make_function_from_sklearn(
            n_samples=dict_num_points[key],
            random_state=generator_seed if generator_seed is None else generator_seed + (0 if key == 'training' else 1000),
            **kwargs_for_skilearn
        )
        dict_points[key] = returned_info[0]
        dict_labels[key] = returned_info[1]
        if dataset_name == 'blobs':
            # returned_info is a tuple (X, y, centers)
            kwargs_for_skilearn['centers'] = returned_info[2]
        pass
        # Transform to tensors
        dict_tensor_points[key] = torch.tensor(dict_points[key], dtype=torch.float32).unsqueeze(-1).unsqueeze(-3)
        dict_tensor_labels[key] = torch.tensor(dict_labels[key], dtype=torch.int64).flatten()
    pass

    # Create the classes to return
    dict_dataset_classes = {}
    for key in dict_num_points:
        dict_dataset_classes[key] = torch.utils.data.TensorDataset(
            dict_tensor_points[key],
            dict_tensor_labels[key]
        )
    pass

    ### Incorporate them into the dictionary to return
    dataset_info_and_constructor['simplified_train_dataset'] = dict_dataset_classes['training']
    dataset_info_and_constructor['simplified_test_dataset'] = dict_dataset_classes['test']

    ### And important: we store, in 'other_kwargs', not only the "extra" parameters for the functions of
    ### sklearn, but also the numbers of points generated:
    ### we get the number of points in this format because this is an attribute not shared by all the
    ### datasets (e.g. 'mnists' and so) and is only present in the "point" datasets.
    other_kwargs = {'num_points_training': num_points_training, 'num_points_test': num_points_test}
    other_kwargs.update(kwargs_for_skilearn)
    dataset_info_and_constructor['other_kwargs'] = other_kwargs

    # Return the dictionary
    return dataset_info_and_constructor


#############################################################################################


def _multiclass_mask_from_image(image_path):
    """
    Create a multiclass mask, which is a Torch tensor, from a grayscale image.

    Parameters
    ----------
    image_path : str
        Path to the grayscale image.

    Returns
    -------
    num_classes : int
        Number of classes detected (not counting the background, 0)
    class_mask : torch.Tensor
        A 2D tensor where each pixel value corresponds to a class label (an int; and 0, background)
    """

    ### Read image into Grayscale, make it float 0-1, and make the white go to 0 and the black go to 1
    im_mask = 1 - 1/255*decode_image(image_path, mode=ImageReadMode.GRAY).to(float)[0]

    ### Try to infer how many classes are there.
    ### Procedudre: we calculate the histogram, search for remarkable bumps we have. And then we set
    ### respective thresholds for binarization

    # We calculate the histogram of the values
    num_bins_histogram = 100
    histogram_result = torch.histogram(im_mask, bins=num_bins_histogram)
    freqs = histogram_result.hist
    bin_boundaries = histogram_result.bin_edges
    bin_centers = bin_boundaries[1:] - 0.5 * (bin_boundaries[1] - bin_boundaries[0])

    # We remove the very ends of the histogram, corresponding to the saturated black and whites, and normalize \
    # the resulting histogram
    norm_freqs_no_01 = freqs[1:-1]
    norm_freqs_no_01 = norm_freqs_no_01 / torch.max(norm_freqs_no_01)
    bins_no_01 = bin_centers[1:-1]

    # We calculate the 2nd derivative of the histogram, looking for a negative 2nd derivative of enough intensity
    second_der = norm_freqs_no_01[0:-2] - 2 * norm_freqs_no_01[1:-1] + norm_freqs_no_01[2:]
    second_der_bins = bins_no_01[1:-1]

    # We take those "negative bumps" of enough intensity as "intermediate classes"
    centers_other_classes = second_der_bins[second_der < -0.25]

    # And add the potential background class (at 0) and calculate the boundaries for decision
    aux = torch.cat([torch.zeros(1), centers_other_classes, torch.ones(1)])
    inf_lims_classes = 0.5 * (aux[1:] + aux[0:-1])

    # And obtain the final mask and the number of classes
    class_mask = torch.zeros(im_mask.size(), dtype=int)
    for ind in range(len(inf_lims_classes)):
        c = ind + 1
        class_mask[im_mask > inf_lims_classes[ind]] = c
    class_mask = class_mask.type(dtype=torch.uint8)
    #
    num_classes = len(class_mask.unique())-1

    # # Finally: we refine the mask by appying an opening. The function for the opening is in Scipy, though
    # np_class_mask = class_mask.detach().numpy()
    # clean_np_class_mask = ndimage.grey_opening(np_class_mask, size=(1,1))
    # # Back to a Torch tensor of type dtype=int
    # class_mask = torch.from_numpy(clean_np_class_mask)
    # class_mask = class_mask.type(dtype=torch.int8)

    # Finally: we refine the mask by appying an opening. The function for the opening is in PIL, though
    pil_class_mask = torchvision.transforms.v2.functional.to_pil_image(class_mask)
    clean_pil_class_mask = pil_class_mask.filter(PIL.ImageFilter.ModeFilter(size=5))
    # Back to a Torch tensor of type dtype=int
    class_mask = torchvision.transforms.functional.pil_to_tensor(clean_pil_class_mask).squeeze()
    class_mask = class_mask.type(dtype=torch.uint8)

    return num_classes, class_mask


#############################################################################################


# def _bordering_mask_from_multiclass_mask(class_mask:torch.Tensor,
#                                          max_pixel_dist:int|float=None, min_pixel_dist:int|float=0.0):
#     """
#     Create a mask, aimed at (later) generating samples very close to the class marked with 1 in `class_mask` but \
#     with a different identity/label. The region will be as follows:
#     - (Reminder: class 0 in `class_mask` means background.)
#     - All classes above class 1 are collapsed into some "virtual" class 2, meaning "the other class"
#     - The distance transform between class 1 and "virtual" class 2 is calculated
#     - The resulting mask has:
#         - A distance transform d1 with respect to the class 1 which is `min_pixel_dist` < d1 < `max_pixel_dist`.
#         - A distance transform d2 with respect to the "virtual" class 2 that is within the lowest 10% of the values \
#           d2 inside the mask of class 1 (so the mask is generated "close" to the "virtual" class 2).
#
#     Parameters
#     ----------
#     class_mask : torch.Tensor
#         A 2D tensor where each pixel value corresponds to a class label (an int; and 0, background)
#     max_pixel_dist : float or int
#         Maximum distance to class 1 for the resulting mask, in pixels!
#     min_pixel_dist : float or int, optional
#         Minimum distance, in pixels, to class 1 for the resulting mask, that is, a "guarding" distance \
#         between the generated mask and class 1.
#         Default: ``0.0``
#
#     Returns
#     -------
#     torch.Tensor
#         A 2D tensor where each pixel at 1 corresponds to the bordering mask
#     """
#
#     ######
#     # Initial checks
#     ######
#
#     try:
#         max_pixel_dist = float(max_pixel_dist)
#         min_pixel_dist = float(min_pixel_dist)
#     except Exception as err:
#         print((f"Both 'max_pixel_dist' and 'min_pixel_dist' must be floats or castable to float: " +
#                f"instead, {max_pixel_dist} and {min_pixel_dist} found!"))
#         raise
#     pass
#
#     assert isinstance(class_mask, torch.Tensor), f"'class_mask' must be a tensor: a {type(class_mask)} found!"
#     assert not torch.is_floating_point(class_mask), f"'class_mask' must be an int tensor, but a float tensor found!"
#
#     ######
#     # Collapse the classes into 0 (background), 1, and 2 (the rest)
#     ######
#
#     binary_mask = torch.where(class_mask > 1, 2, class_mask)
#
#     ######
#     # Calculate the distance transform
#     ######
#
#     INCOMPLETE!
#
#     return binary_mask


#############################################################################################


def _random_sampling_from_multiclass_mask(class_mask, num_points,
                                          generator_seed=None, normalized=True, balanced=True, noise=0.0):
    """
    Create a set of random points sampled from a multiclass mask.

    Parameters
    ----------
    class_mask : torch.Tensor
        2D tensor where each pixel value corresponds to a class label (an int; and 0, background)
    num_points : int
    generator_seed : int, optional
        Default: `None`
    normalized : bool, optional
        If `True`, the coordinates of the sampled points are normalized in the region [-1.0, +1.0]^2 (before \
        potentially adding noise, so the noise corresponds to the final units, pixel or normalized, used).
        If `False`, the coordinates are in pixel units.
        Default: `True`
    balanced: bool, optional
        Whether (roughly) the same amount of samples are generated per class (`True`), independently from their \
        relative spatial extent, or whether sampling is done purely randomly.
        Default: `True`
    noise : float, optional
        Standard deviation of Gaussian noise added to the coordinates of the sampled points, in pixel units.
        Default: `0.0`

    Returns
    -------
    tensor_2D_points : torch.Tensor
        Tensor of shape (num_points, 1, 2, 1) with the coordinates of the sampled points
    tensor_labels : torch.Tensor
        Tensor of shape (num_points,) with the labels of the sampled points
    """

    # Check some basic things about the parameters
    assert isinstance(num_points, int) and num_points > 0, \
        f"'num_points' must be a positive integer; got {num_points}."
    assert generator_seed is None or isinstance(generator_seed, int), \
        f"'generator_seed' must be None or an integer; got {generator_seed}."
    assert isinstance(normalized, bool), f"'balanced' must be a boolean; got {normalized} of type {type(normalized)}."
    assert isinstance(balanced, bool), f"'balanced' must be a boolean; got {balanced} of type {type(balanced)}."
    # Check some things about the mask
    assert isinstance(class_mask, torch.Tensor) and class_mask.dim() == 2, \
        f"'class_mask' must be a 2D torch.Tensor; got {type(class_mask)} with dim {class_mask.dim()}."
    assert class_mask.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64], \
        f"'class_mask' must be an integer-type torch.Tensor; got {class_mask.dtype}."
    unique_classes = class_mask.unique()
    num_classes = len(unique_classes) - 1
    assert len(unique_classes) > 1 or (len(unique_classes) == 1 and unique_classes[0] != 0), \
        f"'class_mask' must contain at least one class different from background (0); got only {unique_classes}."
    # If noise is provided, check it
    assert isinstance(noise, (float, int)) and noise >= 0.0, \
        f"'noise' must be a non-negative float; got {noise}."
    noise = float(noise)

    #####
    # Prepare the random generator
    #####
    # We sample classes independently, in case we want to balance them. For that purpose we estimate the "area" \
    # of each class and decide how many samples we will generate per class.
    #####

    # Set the proportions to sample per class

    im_height, im_width = class_mask.size(-2), class_mask.size(-1)

    total_area_im = class_mask.numel()
    area_with_classes = torch.sum(class_mask != 0)
    area_per_class = [torch.sum(class_mask == i+1) for i in range(num_classes)]
    rel_area_to_total_per_class =[e/float(total_area_im) for e in area_per_class]
    #
    proportion_per_class = [1.0/num_classes]*num_classes if balanced \
        else [e/float(area_with_classes) for e in area_per_class]
    #
    num_points_per_class = [int(round(p*num_points)) for p in proportion_per_class[:-1]]
    num_points_per_class.append(num_points-sum(num_points_per_class))

    #####
    # And we sample per class
    #####

    manual_generator = None if generator_seed is None else torch.Generator().manual_seed(int(generator_seed))

    tensor_2D_points = None
    tensor_labels = None

    for i in range(num_classes):
        #####
        id_class_i_in_mask = i+1
        area_class_i = area_per_class[i]
        rel_area_to_total_class_i = rel_area_to_total_per_class[i]
        proportion_class_i = proportion_per_class[i]
        num_points_class_i = num_points_per_class[i]
        #####

        # We generate a number of points 100 times larger than the number that would be \
        # the expected number equal to the desired number of points, to "make sure" that we get enough points \
        # after discarding those that fall outside the class of interest
        num_points_with_excess = int(num_points_class_i / rel_area_to_total_class_i * 100)

        # So: randomly generating float points in the range [0, width) and [0, height)
        tensor_2D_points_01_with_excess = torch.rand((num_points_with_excess, 1, 2, 1), generator=manual_generator)
        tensor_2D_points_pixel_with_excess = torch.zeros_like(tensor_2D_points_01_with_excess)
        tensor_2D_points_pixel_with_excess[:, 0, 0, 0] = \
            torch.clamp(tensor_2D_points_01_with_excess[:, 0, 0, 0] * (im_width-1), 0, im_width-1)
        tensor_2D_points_pixel_with_excess[:, 0, 1, 0] = \
            torch.clamp(tensor_2D_points_01_with_excess[:, 0, 1, 0] * (im_height-1), 0, im_height-1)

        # And quantify them... by turning them into ints
        tensor_2D_points_quantized_pixel_with_excess = tensor_2D_points_pixel_with_excess.round().int()
        # Transform them (2 indices) into 1D indices
        tensor_1D_indices_with_excess = \
            tensor_2D_points_quantized_pixel_with_excess[:, 0, 1, 0] * im_width + \
            tensor_2D_points_quantized_pixel_with_excess[:, 0, 0, 0]
        # Get the labels corresponding to those points
        tensor_labels_with_excess = class_mask.flatten()[tensor_1D_indices_with_excess]

        # Retain only those points whose class is the desired one
        mask_foreground_points = tensor_labels_with_excess==id_class_i_in_mask
        tensor_points_with_excess_class_i = tensor_2D_points_pixel_with_excess[mask_foreground_points]
        tensor_labels_with_excess_class_i = tensor_labels_with_excess[mask_foreground_points]

        # If not enough points... repeat as many points from the beginning as necessary
        while tensor_labels_with_excess_class_i.size(0) < num_points_class_i:
            # Generate a warning about the situation
            print(f"Warning: not enough foreground points sampled ({tensor_labels_with_excess_class_i.size(0)})" +
                  f" to reach the desired number ({num_points_class_i}) for class {id_class_i_in_mask}." +
                  f" Repeating points circularly to reach the number.")
            num_points_needed = num_points_class_i - tensor_labels_with_excess_class_i.size(0)
            num_points_to_add_in_iteration = min(num_points_needed, tensor_labels_with_excess_class_i.size(0))
            #
            tensor_points_with_excess_class_i = torch.cat(
                [tensor_points_with_excess_class_i,
                 tensor_points_with_excess_class_i[0:num_points_to_add_in_iteration]], dim=0
            )
            tensor_labels_with_excess_class_i = torch.cat(
                [tensor_labels_with_excess_class_i,
                 tensor_labels_with_excess_class_i[0:num_points_to_add_in_iteration]], dim=0
            )
        pass

        # Finally, take only the required number of points
        tensor_2D_points_class_i = tensor_points_with_excess_class_i[0:num_points_class_i]
        tensor_labels_class_i = tensor_labels_with_excess_class_i[0:num_points_class_i]
        # And pack the points and labels into the complete set of points and labels
        tensor_2D_points = tensor_2D_points_class_i if tensor_2D_points is None else \
            torch.cat([tensor_2D_points, tensor_2D_points_class_i], dim=0)
        tensor_labels = tensor_labels_class_i if tensor_labels is None else \
            torch.cat([tensor_labels, tensor_labels_class_i], dim=0)
        #
    pass

    # If required, normalize the points to the range [-1.0, 1.0]
    max_value_normalization = 1.0 # To have points in the range [-max_value_normalization, max_value_normalization]
    if normalized:
        offset_pixel_dimensions = torch.tensor([im_width / 2.0, im_height / 2.0], dtype=torch.float32)
        scale_factor = max_value_normalization * 2.0 / (max(im_height, im_width) - 1)
        tensor_2D_points[:, 0, 0, 0] = tensor_2D_points[:, 0, 0, 0] - offset_pixel_dimensions[0]
        tensor_2D_points[:, 0, 1, 0] = tensor_2D_points[:, 0, 1, 0] - offset_pixel_dimensions[1]
        tensor_2D_points = tensor_2D_points * scale_factor
    pass

    # Add noise
    if noise > 0.0:
        # The "manual_generator" has been already generated
        tensor_2D_noise = noise * torch.randn(tensor_2D_points.size(), generator=manual_generator,
                                              dtype=tensor_2D_points.dtype, device=tensor_2D_points.device)
        tensor_2D_points = tensor_2D_points + tensor_2D_noise
    pass

    # And finally: the labels kept, corresponding to the foreground, go 1,...,C: make them go 0,...,C-1
    tensor_labels = tensor_labels - 1

    return tensor_2D_points, tensor_labels


#############################################################################################


def _find_image_custom_dataset(dataset_name):
    """
    It looks for the image of name `dataset_name`.jpg in the folder \\
    `src/experimental_evaluation/masks_for_2D_datasets`. If such image is not found, \
    an error will be raised.

    Parameters
    ----------
    dataset_name : str
        Type of dataset to create, to be based on the image `dataset_name`.jpg

    Returns
    -------
    str : The absolute path to the image
    """

    # Check that the image corresponding to the requested dataset exists
    image_filename = f"{dataset_name}.jpg"
    image_folder = os.path.join(os.path.dirname(__file__), 'masks_for_2D_datasets')
    image_path = os.path.join(image_folder, image_filename)
    assert os.path.exists(image_path), \
        f"Image file for dataset '{dataset_name}' not found at path: {image_path}"

    return image_path


#############################################################################################


def _create_custom_2D_classification_dataset_from_image(
        dataset_name, num_points_training, num_points_test, generator_seed=None,
        **kwargs_for_random_sampling_from_multiclass_mask
):
    """
    Create a 2D-point classification dataset from a grayscale image named `dataset_name`.jpg, \
    which must represent a white background with a number of classes (each one connected or not) represented, each one, \
    by a (fairly) uniform gray level distinguishable for the level of the rest of classes. The dataset is simply \
    generated from said image using Monte Carlo sampling and discarding points that fall in the background (white).

    The image is, by default, presumed as centered at the origin (0,0) and having its longest side of length 1.0, \
    so the coordinates of the generated points will be in the range [-0.5, 0.5].

    Some of the values for `dataset_name` already included are: 'crossed_moons', 'stars', 'three_moons'. \
    However, other dataset names can be used by adding a corresponding schematic image with name \
    `dataset_name`.jpg in the folder `src/experimental_evaluation/masks_for_2D_datasets`. If such image is not found, \
    an error will be raised.

    Parameters
    ----------
    dataset_name : str
        Type of dataset to create, to be based on the image `dataset_name`.jpg
    num_points_training : int
        Total number of points to generate, corresponding to the "training + validation" set
    num_points_test : int
        Number of points to generate for the test set
    generator_seed : int, optional
        Default: `None`

    Returns
    -------
    dict

    The dictionary will contain the following entries:
    'channels': int, 'default_im_size': 2-tuple, 'classes': int,
    'simplified_train_dataset' and 'simplified_test_dataset': torch.utils.data.Dataset
    """

    # Check the validity of common parameters
    assert isinstance(num_points_training, int) and num_points_training > 0, \
        f"'num_points_training' must be a positive integer; got {num_points_training}."
    assert isinstance(num_points_test, int) and num_points_test > 0, \
        f"'num_points_test' must be a positive integer; got {num_points_test}."
    assert generator_seed is None or isinstance(generator_seed, int), \
        f"'generator_seed' must be None or an integer; got {generator_seed}."

    # Take the "kwargs_for_random_sampling_from_multiclass_mask" and extract only those that are "useful",
    # and give a default, and prepare the rest to be stored in "other_kwargs"
    effective_kwargs_for_random_sampling_from_multiclass_mask = \
        {'noise': kwargs_for_random_sampling_from_multiclass_mask.pop('noise', 0.0)}

    # Check that the image corresponding to the requested dataset exists
    image_path = _find_image_custom_dataset(dataset_name)

    # Load the mask corresponding to the image corresponding to the requested dataset
    num_classes, class_mask = _multiclass_mask_from_image(image_path)

    ###############################################
    # Dictionary to return: EMPTY DEFINITION AND SUBSEQUENT FILLING
    ###############################################
    dataset_info_and_constructor = {'channels': 1, 'default_im_size': None, 'classes': None,
                                    'simplified_train_dataset': None, 'simplified_test_dataset': None,
                                    'other_kwargs': {}}
    ###############################################

    dataset_info_and_constructor['default_im_size'] = (2, 1)
    dataset_info_and_constructor['classes'] = num_classes

    # Generate the points
    dict_num_points = {'training': num_points_training, 'test': num_points_test}
    # dict_tensor_points = {}
    # dict_tensor_labels = {}
    dict_dataset_classes = {}

    # Generate the points and labels for both training and test and pack them in respective TensorDataset structures
    for key in dict_num_points:
        tensor_points, tensor_labels = _random_sampling_from_multiclass_mask(
            class_mask, num_points=num_points_training,
            generator_seed=generator_seed if generator_seed is None else generator_seed + (0 if key == 'training' else 1000),
            normalized=True, **effective_kwargs_for_random_sampling_from_multiclass_mask)
        dict_dataset_classes[key] = torch.utils.data.TensorDataset(tensor_points, tensor_labels)
    pass

    ### Incorporate them into the dictionary to return
    dataset_info_and_constructor['simplified_train_dataset'] = dict_dataset_classes['training']
    dataset_info_and_constructor['simplified_test_dataset'] = dict_dataset_classes['test']

    ### And important: we store, in 'other_kwargs', not only the "extra" parameters for the functions
    ### "_random_sampling_from_multiclass_mask" but also the numbers of points generated:
    ### we get the number of points in this format because this is an attribute not shared by all the
    ### datasets (e.g. 'mnists' and so) and is only present in the "point" datasets.
    other_kwargs = {'num_points_training': num_points_training, 'num_points_test': num_points_test}
    other_kwargs.update(effective_kwargs_for_random_sampling_from_multiclass_mask)
    dataset_info_and_constructor['other_kwargs'] = other_kwargs

    # Return the dictionary
    return dataset_info_and_constructor


#############################################################################################


def obtain_classification_dataset_loaders_from_point_data(
        dataset_name, num_points_training, num_points_test=0,
        normalization=None, batch_size=100,
        train_proportion=0.9, val_proportion=0.1,
        num_workers=None, shuffle=True, generator_seed=None,
        **kwargs_for_point_generation
):
    """
    Create a classification dataset from point data, and return the loaders for training, validation and test.

    Different dataset are considered, some of them generated directly from corresponding functions of the library \
    Sklearn, and some others generated using custom distributions which can be included by adding schematic \
    images with the extension '.jpg' in the folder `src/experimental_evaluation/masks_for_2D_datasets`. In particular, \
    so far:

    - The datasets generated from Sklearn correspond to the following options for `dataset_name`: \
      'moons', 'circles', 'blobs', and 'classification'.

    - The `dataset_name` values different from the above search for a corresponding file of name `dataset_name`.jpg \
      and generate random samples accordingly. Values available so far are: 'crossed_moons', 'stars', 'three_moons'.

    Parameters
    ----------
    dataset_name : str
        Type of dataset to create. Options are: 'moons', 'circles', 'blobs', 'classification'.
    num_points_training : int
        Total number of points to generate, corresponding to the "training + validation" set
    num_points_test : int, optional
        Number of points to generate for the test set. Default: 0 (empty test set)
    normalization : str, optional
        **It only accepts ``None`` at the moment!**
        Default: `None`
    batch_size : int, optional
        Batch size to be used in the loaders. Default: 100
    train_proportion : float, optional
        Proportion of the `num_points` non-test points to be used for training (VS for validation).
        Default: 0.9
    val_proportion
        Proportion of the `num_points` non-test points to be used for validation (VS for training).
        Default: 0.1
    num_workers : int, optional
        Number of workers to be used in the loaders. Default: ``None`` (which means ``4``)
    shuffle: bool, optional
        If `True`, the multiple obtained :py:class:`~torch.utils.data.DataLoader` in the returned \
        :py:class:`.LoadedDatasetDict` are reshuffled at every epoch (that is, every time they are accessed from the \
        beginning as an iterable). In order to keep the batch and image order at each epoch access this \
        argument should be set to `False`. \
        Default: `True`
    generator_seed : int, optional
        Default: `None`
    kwargs_for_point_generation : dict, optional
        Additional keyword arguments for the point generation function from `sklearn.datasets`. E.g.: for the \
        ``'moons'`` type, one can provide the `noise` argument to add Gaussian noise to the data.

    Returns
    -------
    LoadedDatasetDict

    Dictionary subclass :py:class:`.LoadedDatasetDict`
    """

    ###############
    # For compatibility with the data loaders of Pytorch and images, we will fill all fields in LoadedDatasetDict:
    ###############
    # `'dataset_name'`
    # `'proportion_train'`, `'proportion_val'`
    # `'generator_seed'`
    # `'batch_size'`
    # `'num_batches_train'`, `'num_batches_val'`, `'num_batches_test'`
    # `'dataloader_train'`, `'dataloader_val'`, `'dataloader_test'` -> we will use an iterable that spits the batches
    # `'im_height'`, `'im_width'`: this will be 2 and 1, respectively (col vectors)
    # `'channels'` -> 1
    # `'classes'` -> in principle 2
    # `'original_mean_train'`, `'original_std_train'`: :py:class:`torch.Tensor`
    # `'final_mean_train'`, `'final_std_train'`: :py:class:`torch.Tensor`
    # `'original_min_train'`, `'original_max_train'`: :py:class:`torch.Tensor`
    # `'final_min_train'`, `'final_max_train'`: :py:class:`torch.Tensor`
    # `'forced_bn'`: :py:class:`bool`
    # `'colorspace'`, `'source_normalization'`: :py:class:`str`
    # `'normalized_dataset'`: :py:class:`str`
    # `'tuple_of_pairs_other_kwargs'`: :py:class:`tuple`
    ###############

    # Default values if not provided or none
    default_num_workers = 4
    default_mislabeled_proportion = 0.0

    # For now 'mislabeled_proportion' is not used, so we use the default, but we prepare it for the future, and we store it in the dictionary to return
    mislabeled_proportion = default_mislabeled_proportion

    # Check the validity of common parameters
    assert isinstance(num_points_training, int) and num_points_training > 0, \
        f"'num_points_training' must be a positive integer; got {num_points_training}."
    assert isinstance(num_points_test, int) and num_points_training >= 0, \
        f"'num_points_test' must be a non-negative integer; got {num_points_test}."
    assert isinstance(batch_size, int) and batch_size > 0, \
        f"'batch_size' must be a positive integer; got {batch_size}."
    assert num_points_training >= batch_size, \
        f"'num_points_training' {num_points_training} must be at least as large as batch_size {batch_size}."
    assert normalization is None, \
        f"'normalization' must be None, no other option currently accepted: Got {normalization}."
    ################################################################
    if num_workers is None:
        num_workers = default_num_workers
    assert isinstance(num_workers, int) and num_workers >= 0, \
        f"Expected 'num_workers' to be a non-negative integer, received: {num_workers}"
    ################################################################

    # If the dataset requested is in the list of datasets generated from Sklearn, to Sklearn we go.
    # Otherwise we explore the custom datasets from images.
    if dataset_name in _list_dataset_names_for_create_sklearn_dataset_for_2D_analysis:
        dataset_info_and_constructor = _create_sklearn_dataset_for_2D_analysis(
            dataset_name, num_points_training, num_points_test,
            generator_seed=generator_seed,
            **kwargs_for_point_generation
        )
    else:
        dataset_info_and_constructor = _create_custom_2D_classification_dataset_from_image(
            dataset_name, num_points_training, num_points_test,
            generator_seed=generator_seed,
            **kwargs_for_point_generation
        )
    pass

    # Total train dataset and test dataset
    total_train_data = dataset_info_and_constructor['simplified_train_dataset']
    test_data = dataset_info_and_constructor['simplified_test_dataset']

    # Separation of 'total_train_data' into 'train_data' and 'val_data'
    added_proportion = train_proportion + val_proportion
    if added_proportion > 1.0:
        train_proportion = train_proportion / added_proportion
        val_proportion = val_proportion / added_proportion
    pass
    remaining_proportion = 1.0 - train_proportion - val_proportion
    manual_generator = None if generator_seed is None else torch.Generator().manual_seed(int(generator_seed))
    train_data, val_data, _ = random_split(
        total_train_data, [train_proportion, val_proportion, remaining_proportion],
        generator=manual_generator
    )

    ############################################################################
    # Create and fill an object of the dict-subclass LoadedDatasetDict
    ############################################################################
    dataset_dict = LoadedDatasetDict()
    ############################################################################

    dataset_dict['dataset_name'] = dataset_name
    dataset_dict['proportion_train'] = train_proportion
    dataset_dict['proportion_val'] = val_proportion
    dataset_dict['proportion_mislabel'] = mislabeled_proportion

    dataset_dict['generator_seed'] = generator_seed

    dataset_dict['batch_size'] = batch_size

    dataset_dict['num_workers'] = num_workers

    dataset_dict['dataloader_train'] = \
        DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) if len(train_data) else None
    dataset_dict['num_batches_train'] = len(dataset_dict['dataloader_train']) \
        if dataset_dict['dataloader_train'] is not None else 0
    dataset_dict['dataloader_val'] = \
        DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) if len(val_data) else None
    dataset_dict['num_batches_val'] = len(dataset_dict['dataloader_val']) \
        if dataset_dict['dataloader_val'] is not None else 0
    dataset_dict['dataloader_test'] = \
        DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) if len(test_data) else None
    dataset_dict['num_batches_test'] = len(dataset_dict['dataloader_test']) \
        if dataset_dict['dataloader_test'] is not None else 0
    #
    dataset_dict['channels'] = 1
    dataset_dict['colorspace'] = 'unknown'
    dataset_dict['im_height'] = dataset_info_and_constructor['default_im_size'][-2]
    dataset_dict['im_width'] = dataset_info_and_constructor['default_im_size'][-1]
    dataset_dict['classes'] = dataset_info_and_constructor['classes']

    dataset_dict['source_normalization'] = 'no'
    dataset_dict['normalized_dataset'] = normalization if normalization is not None else 'no'

    # Obtain the statistics of the dataset
    original_mean_train, original_std_train = _get_mean_std_from_dataloader(
        dataset_dict['dataloader_train'])
    original_min_train, original_max_train = _get_min_max_within_image_from_dataloader(
        dataset_dict['dataloader_train'])
    # No normalization, they are the same
    final_mean_train, final_std_train = original_mean_train, original_std_train
    final_min_train, final_max_train = original_min_train, original_max_train
    #
    dataset_dict['original_mean_train'] = original_mean_train
    dataset_dict['original_std_train'] = original_std_train
    dataset_dict['final_mean_train'] = final_mean_train
    dataset_dict['final_std_train'] = final_std_train
    dataset_dict['original_min_train'] = original_min_train
    dataset_dict['original_max_train'] = original_max_train
    dataset_dict['final_min_train'] = final_min_train
    dataset_dict['final_max_train'] = final_max_train

    dataset_dict['tuple_of_pairs_other_kwargs'] = tuple(sorted(dataset_info_and_constructor['other_kwargs'].items()))
    # print(f"dataset_dict = {dataset_dict}")

    return dataset_dict