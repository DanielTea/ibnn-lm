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

import numpy as np
import itertools

import torch
from torch import nn
import torchattacks

import mlflow
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import random
import math


from experimental_evaluation.interaction_with_mlflow import mlflow_log_dataset
from applications.classifiers import ClassifierBaseModel
from experimental_evaluation.pixle_fix import PixleFix
from modified_rf.nn_layers import SMLayer, IBNNLiteLayer, IBNNInternalLayer, IBNNLayer


def log_random_images(images_batch, batch_num):
    # Select 100 random images
    num_images = min(100, images_batch.size(0))
    random_indices = random.sample(range(images_batch.size(0)), num_images)
    selected_images = images_batch[random_indices]

    # Calculate the number of rows and columns to be as square as possible
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)
    # Create the subplot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(selected_images[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        ax.axis('off')

    # Log the plot to mlflow
    mlflow.log_figure(fig, f"input_batch_{batch_num}.png")

    plt.close(fig)


def log_side_by_side_images(original_images, perturbed_images, indices, batch_num, description,
                            perturbed_images_sm=None, labels=None, predicted_labels=None,
                            predicted_labels_perturbed=None, predicted_labels_perturbed_sm=None):
    # Create the subplot
    num_images = len(indices)
    num_cols = 3 if perturbed_images_sm is not None else 2  # Add a column for perturbed_images_sm if it exists
    num_rows = num_images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Ensure axes is a 2D array
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        # Plot original image
        ax = axes[i, 0]
        original_image = original_images[idx].cpu().numpy()
        if original_image.shape[0] == 1:  # Grayscale image
            original_image = original_image.squeeze(0)
        else:  # RGB image
            original_image = original_image.transpose(1, 2, 0)
        ax.imshow(original_image, cmap='gray')
        ax.axis('off')
        title = 'Original'
        if labels is not None:
            title += f' (Class: {labels[idx].item()})'
        if predicted_labels is not None:
            title += f' (Pred: {predicted_labels[idx].item()})'
        ax.set_title(title)

        # Plot perturbed image
        ax = axes[i, 1]
        perturbed_image = perturbed_images[idx].cpu().numpy()
        if perturbed_image.shape[0] == 1:  # Grayscale image
            perturbed_image = perturbed_image.squeeze(0)
        else:  # RGB image
            perturbed_image = perturbed_image.transpose(1, 2, 0)
        ax.imshow(perturbed_image, cmap='gray')
        ax.axis('off')
        title = 'Perturbed'
        if predicted_labels_perturbed is not None:
            title += f' (Pred: {predicted_labels_perturbed[idx].item()})'
        ax.set_title(title)

        # Plot perturbed image from classifier_sm if it exists
        if perturbed_images_sm is not None:
            ax = axes[i, 2]
            perturbed_image_sm = perturbed_images_sm[idx].cpu().numpy()
            if perturbed_image_sm.shape[0] == 1:  # Grayscale image
                perturbed_image_sm = perturbed_image_sm.squeeze(0)
            else:  # RGB image
                perturbed_image_sm = perturbed_image_sm.transpose(1, 2, 0)
            ax.imshow(perturbed_image_sm, cmap='gray')
            ax.axis('off')
            title = 'Perturbed_SM'
            if predicted_labels_perturbed_sm is not None:
                title += f' (Pred: {predicted_labels_perturbed_sm[idx].item()})'
            ax.set_title(title)

    # Log the plot to mlflow
    mlflow.log_figure(fig, f"{description}_batch_{batch_num}.png")
    plt.close(fig)


def _generate_fgsm_attack_images_from_grad(image, epsilon, data_grad=None, normalization=None):
    """
    The function generates perturbed images using the Fast Gradient Sign Method (FGSM) principle by altering \
    the image `image`, or each image of the 4D `image` if that is the case, for the given scalar value `epsilon`. \
    If the provided `epsilon` is an iterable of scalars all the individual epsilons will be used and the result \
    returned, again, in the form of an iterable.

    The optional argument `data_grad` is the gradient of the loss function with respect to the input image. If it is \
    not provided, which is considered the usual case, the gradient is presumed to be in the corresponding field of the \
    `image` tensor itself; if `data_grad` is provided, it is used instead.

    Parameters
    ----------
    image : torch.Tensor
        The input image, if 3D tensor, or batch of images, if 4D tensor, to be perturbed. The gradient for each image \
        must be included if the gradient is not explicitly provided in the `data_grad` argument
    epsilon : float or list[float] or tuple[float]
        The scalar value $0<\\epsilon<1$ scaling the gradient to perturb the corresponding image(s).
    data_grad : torch.Tensor, optional
        The gradient of the loss function with respect to the input image(s). If not provided, the gradient is \
        presumed to be in the corresponding field of the `image` tensor itself and used. Default: ``None``
    normalization : str, optional
        Value among ``'min_max_01_per_im'`` and ``'N(0;1)'``, or ``None`` (or ``'None'``) \
        (see :py:func:`.obtain_classification_dataset_loaders`), indicating the type of normalization to be applied. \
        Default: ``None``

    Returns
    -------
    torch.Tensor or list[torch.Tensor] or tuple[torch.Tensor]
        The perturbed image(s) after the FGSM attack, of the same size of the input image(s).
    """

    #####
    # Initial checks about sizes and gradients
    #####

    # Setting the grad and the image
    # Grad
    if data_grad is None:
        if image.grad is None:
            raise ValueError("The gradient is not provided and it is not stored in the input image(s)")
        else:
            data_grad = image.grad.data.clone().detach()
        pass
    pass
    # Image
    image = image.clone().detach()
    image.requires_grad = False
    image.grad = None
    # Check their sizes
    if image.size() != data_grad.size():
        raise ValueError("The image and the gradient have different sizes")
    pass

    # Checking the epsilon and making it a list
    flag_initial_epsilon_was_list = False
    list_epsilons = []
    if isinstance(epsilon, (list, tuple)):
        list_epsilons = list(epsilon)
        flag_initial_epsilon_was_list = True
    elif isinstance(epsilon, float):
        list_epsilons = [epsilon]
        flag_initial_epsilon_was_list = False
    else:
        raise ValueError(f"The epsilon must be a float or a list/tuple of floats: type {type(epsilon)} provided " +
                         f"(epsison={epsilon})")
    pass
    for epsilon_i in list_epsilons:
        if (not isinstance(epsilon_i, float)) or (not (0 <= epsilon_i < 1)):
            raise ValueError(f"The epsilon must be a float between 0 and 1: a value epsilon={epsilon_i} " +
                             f"(type {type(epsilon_i)}) provided")
        pass
    pass

    #####
    # Generation of the perturbed images
    #####

    # Collect the element-wise sign of the data gradient
    sign_grad = data_grad.sign()

    # Fill a list of perturbed images with the epsilons of the 'list_epsilons'
    list_perturbed_images = []
    for epsilon_i in list_epsilons:
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image_epsilon_i = image + epsilon_i * sign_grad
        if (normalization is not None) and (normalization == 'min_max_01_per_im'):
            perturbed_image_epsilon_i = torch.clamp(perturbed_image_epsilon_i, 0, 1)
        pass
        # Append the perturbed image to the list
        list_perturbed_images.append(perturbed_image_epsilon_i)
    pass

    # Returning the perturbed images
    return list_perturbed_images if flag_initial_epsilon_was_list else list_perturbed_images[0]


def _generate_pgd_attack_images_from_grad(image, epsilon, alpha, num_iter, data_grad=None):
    """
    The function generates perturbed images using the Projected Gradient Descent (PGD) principle by altering \
    the image `image`, or each image of the 4D `image` if that is the case, for the given scalar value `epsilon`. \
    If the provided `epsilon` is an iterable of scalars all the individual epsilons will be used and the result \
    returned, again, in the form of an iterable.

    The optional argument `data_grad` is the gradient of the loss function with respect to the input image. If it is \
    not provided, which is considered the usual case, the gradient is presumed to be in the corresponding field of the \
    `image` tensor itself; if `data_grad` is provided, it is used instead.

    Parameters
    ----------
    image : torch.Tensor
        The input image, if 3D tensor, or batch of images, if 4D tensor, to be perturbed. The gradient for each image \
        must be included if the gradient is not explicitly provided in the `data_grad` argument
    epsilon : float or list[float] or tuple[float]
        The scalar value $0<\\epsilon<1$ scaling the gradient to perturb the corresponding image(s).
    alpha : float
        The scalar value $0<\\alpha<1$ scaling the gradient to perturb the corresponding image(s).
    num_iter : int
        The number of iterations to perform the PGD attack.
    data_grad : torch.Tensor, optional
        The gradient of the loss function with respect to the input image(s). If not provided, the gradient is \
        presumed to be in the corresponding field of the `image` tensor itself and used. Default: ``None``

    Returns
    -------
    torch.Tensor or list[torch.Tensor] or tuple[torch.Tensor]
        The perturbed image(s) after the PGD attack, of the same size of the input image(s).
    """

    raise Exception("Only a placeholder for now! Not implemented yet!")


def _generate_gn_attack_images(image, params, normalization=None):
    """
        The function generates perturbed images using Gaussian noise to altering \
        the image `image`, or each image of the 4D `image` if that is the case, for the given scalar value `mu` (mean) and `sigma` (std). \
        If the provided `mean` is an iterable of scalars all the individual means will be used and the result \
        returned, again, in the form of an iterable. The same for `std`.

        Parameters
        ----------
        image : torch.Tensor
            The input image, if 3D tensor, or batch of images, if 4D tensor, to be perturbed. The gradient for each image \
            must be included if the gradient is not explicitly provided in the `data_grad` argument
        params : dict of float or dict of list[float] or dict of tuple[float]
            Mean is the scalar value $0<\\mu<1$  of gaussian noise to perturb the corresponding image(s).
            Std is the scalar value $0<\\sigma<1$ of gaussian noise to perturb the corresponding image(s).
        normalization : str, optional
            Value among ``'min_max_01_per_im'`` and ``'N(0;1)'``, or ``None`` (or ``'None'``) \
            (see :py:func:`.obtain_classification_dataset_loaders`), indicating the type of normalization to be applied. \
            Default: ``None``

        Returns
        -------
        torch.Tensor or list[torch.Tensor] or tuple[torch.Tensor]
            The perturbed image(s) after the adding Gaussian noise, of the same size of the input image(s).
        """

    #####
    # Initial checks about sizes
    #####

    # Image
    image = image.clone().detach()

    mean = params['mean']
    std = params['std']

    # Checking the mean and making it a list
    if isinstance(mean, (list, tuple)):
        list_mean = list(mean)
        flag_initial_mean_was_list = True
    elif isinstance(mean, float):
        list_mean = [mean]
        flag_initial_mean_was_list = False
    else:
        raise ValueError(f"The mean must be a float or a list/tuple of floats: type {type(mean)} provided " +
                         f"(mean={mean})")
    pass
    for mean_i in list_mean:
        if (not isinstance(mean_i, float)) or (not (0 <= mean_i <= 1)):
            raise ValueError(f"The mean must be a float between 0 and 1: a value mean={mean_i} " +
                             f"(type {type(mean_i)}) provided")
        pass
    pass

    # Checking the std and making it a list
    if isinstance(std, (list, tuple)):
        list_std = list(std)
        flag_initial_std_was_list = True
    elif isinstance(std, float):
        list_std = [std]
        flag_initial_std_was_list = False
    else:
        raise ValueError(f"The std must be a float or a list/tuple of floats: type {type(std)} provided " +
                         f"(std={std})")
    pass
    for std_i in list_std:
        if (not isinstance(std_i, float)) or (not (0 <= std_i <= 1)):
            raise ValueError(f"The std must be a float between 0 and 1: a value std={std_i} " +
                             f"(type {type(std_i)}) provided")
        pass
    pass

    #####
    # Generation of the perturbed images
    #####

    # Obtain the device of the image
    device = image.device

    # Fill a list of perturbed images with the means and stds of the 'list_mean' and 'list_std'
    list_perturbed_images = []
    for mean_i, std_i in zip(list_mean, list_std):
        # Create the perturbed image by adding Gaussian noise
        perturbed_image = image + torch.normal(mean_i, std_i, size=image.size()).to(device)
        if (normalization is not None) and (normalization == 'min_max_01_per_im'):
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
        pass
        # Append the perturbed image to the list
        list_perturbed_images.append(perturbed_image)

    # Returning the perturbed images
    return list_perturbed_images if flag_initial_mean_was_list or flag_initial_std_was_list else list_perturbed_images[
        0]



####################################################################################################################
####################################################################################################################
####################################################################################################################
#
# FUNCTIONS FOR ATTACKING A MODEL AND ASSESSING THE ATTACKS!
#
####################################################################################################################
####################################################################################################################
####################################################################################################################



def assess_classifier_for_aa(
        classifier_nn, dataset_dict, loss_function, attack_type, attack_params, classifier_sm_surrogate=None,
        validation_on_test_subset=True, mlflow_run_id=None, run_name=None, verbose='medium'):
    """
    The function assesses the performance of a given classifier `classifier_nn` for a given black-box \
    adversarial attack `attack_type` and its parameters `attack_params` using the data provided by the `data_loader`. \
    The loss used for the training should be also provided.

    Parameters
    ----------
    classifier_nn : torch.nn.Module or list[torch.nn.Module] or tuple[torch.nn.Module]
        The classifier neural network to be assessed, if it is a subclass of `nn.Module`, or the family of \
        classifier neural networks to be assessed together, if it is a list/tuple of subclasses of `nn.Module`.
    dataset_dict : LoadedDatasetDict
        The data loader providing the data to be used for the assessment.
    loss_function : ~collections.abc.Callable
        Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
        user-defined functions of *(1)* the output of the network and (2) the target can be used too
    attack_type : str
        The type of the adversarial attack to be used. The possible values are 'FGSM' and 'PGD'
    attack_params : dict
        The dictionary of parameters for the adversarial attack. The parameters depend on the attack type:
        - 'FGSM': {'epsilon': float}
        - 'PGD': {'epsilon': float, 'alpha': float, 'num_iter': int}
        - 'OnePixel': {'pixel_count': int, 'max_iter': int, 'popsize': int}
        - 'DeepFool' : {'steps': int, 'overshoot': float}
        - 'Pixle': {    'x_dimensions': int, 'y_dimensions': int, 'pixel_mapping': str, \
                        'restarts': int, 'max_iter': int, 'update_each_iteration': bool}
        - 'GN' (Gaussian noise) : {'mean': float, 'std': float}
        It is the only full black-box attack considered for now.
    classifier_sm_surrogate: torch.nn.Module, optional
        The classifier neural network Standard Model surrogate version of the classifier `classifier_nn` \
        to be assessed; **it is only used, and necessary in fact, for white-box-based attacks such as ** \
        **'PGD', 'FGSM', 'OnePixel', but not for full black-box attacks such as 'Pixle'**.
        Default: ``None``
    validation_on_test_subset : bool, optional
        The flag indicating whether the validation should be performed on the test subset of the data loader. \
        If `True`, the test subset is used; if `False`, the validation ('val')subset is used. Default: ``True``
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
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of progress printing. \
        Default: ``'medium'``

    Returns
    -------
    float
        The accuracy of the classifier `classifier_nn` for the adversarial attack `attack_type` and its parameters \
        `attack_params` using the data provided by the `data_loader`.
    """

    ###########################
    # Argument checks
    ###########################

    list_of_attacks = {
        'white-box': ['fgsm', 'pgd', 'onepixel'],
        'black-box': ['pixle']
    }

    assert isinstance(classifier_nn, ClassifierBaseModel), \
        f"The 'classifier_nn' must be an instance of the class 'ClassifierBaseModel': " + \
        f"however, {type(classifier_nn)} was provided. "

    attack_type = attack_type.lower()
    if attack_type in list_of_attacks['white-box']:
        assert isinstance(classifier_sm_surrogate, ClassifierBaseModel), \
            f"For the attacks {list(list_of_attacks['white-box'])} the surrogate model is required; " + \
            f"however, {type(classifier_sm_surrogate)} was provided."
    elif attack_type in list_of_attacks['black-box']:
        # All good, simply pass
        pass
    else:
        raise Exception(
            f"The provided 'attack' {attack_type} is not any of the considered attacks " + \
            " or ".join([", ".join(list_of_attacks[key]) + f" ({key})" for key in list_of_attacks])
        )
    pass

    ###########################
    ###########################

    if verbose == 'high':
        print(f"\nClassifier network ...\n")
        print(str(classifier_nn))
        attributes_to_display = classifier_nn.fields_to_log
        print(f"\n ... with fields ...\n")
        for key in attributes_to_display:
            print(f"\t\t{key}: {attributes_to_display[key]}")
        pass
    pass

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
        raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
                        "only one of them can be provided at a time!")
    pass

    ###################################################
    # Log the parameters of the attack (metric, when possible, and always, as param)
    ###################################################

    # Log the dataset info
    if mlflow_run_id is not None:
        mlflow_log_dataset(dataset_dict, mlflow_run_id)
    pass

    if mlflow_run_id is not None:
        mlflow.log_params({'attack_type': attack_type}, run_id=mlflow_run_id)
        for key in attack_params:
            if isinstance(attack_params[key], (int, float)):
                mlflow.log_metrics({f'attack_parameter_{key}': attack_params[key]}, run_id=mlflow_run_id)
            pass
            mlflow.log_params({f'attack_parameter_{key}': str(attack_params[key])}, run_id=mlflow_run_id)
        pass
    pass

    # Field 'dataloader_val' or 'dataloader_test' depending on 'validation_on_test_subset'
    subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'

    # We process the classifiers separately and obtain the following data per classifier
    # - loss for the original images
    # - loss for the adversarial images
    # - accuracy for the original images
    # Finally we additionally combine the result of all classifiers
    list_summary_classifiers = []

    # Check the device of the classifier
    device_classifier_nn = next(iter(classifier_nn.parameters())).data.device.type
    #
    # Auxiliary variables for the assessment of the classifier as accumulated values from each batch
    total_num_images_images = 0
    #
    cum_loss_not_averaged = 0.0
    cum_perturbed_loss_not_averaged = 0.0
    #
    cum_right_classifications_not_averaged = 0.0
    cum_perturbed_right_classifications_not_averaged = 0.0
    #
    cum_non_convergent_for_images = 0.0
    cum_non_convergent_for_perturbed_images = 0.0
    #
    fp_conv_err = 0.0
    fp_perturbed_conv_err = 0.0
    #
    if classifier_sm_surrogate is not None:
        classifier_sm_surrogate = classifier_sm_surrogate.to(device_classifier_nn)
        classifier_sm_surrogate.eval()
        # cum_perturbed_loss_not_averaged_sm = 0.0
        # cum_perturbed_right_classifications_not_averaged_sm = 0.0
        # sm_cum_loss_not_averaged = 0.0
        # sm_cum_right_classifications_not_averaged = 0.0
        # sm_cum_perturbed_loss_not_averaged = 0.0
        # sm_cum_perturbed_right_classifications_not_averaged = 0.0
        # sm_cum_perturbed_loss_not_averaged_sm = 0.0
        # sm_cum_perturbed_right_classifications_not_averaged_sm = 0.0
        # fp_perturbed_conv_err_sm = 0.0
    pass

    #########################################################################################################
    # Set the attack generator (a function that takes a batch of images and returns the perturbed batch
    #########################################################################################################

    attack_generator = None # It will be function which takes (images_batch, labels_batch) as inputs

    classifier_for_attack_calculation = \
        classifier_nn if attack_type.lower() in list_of_attacks['black-box'] \
            else classifier_sm_surrogate

    if attack_type.lower() == 'fgsm':
        attack_generator = torchattacks.FGSM(
            classifier_for_attack_calculation,
            attack_params['epsilon']
        )
    elif attack_type.lower() == 'pgd':
        attack_generator = torchattacks.PGD(
            classifier_for_attack_calculation,
            attack_params['epsilon'], attack_params['alpha'], attack_params['num_iter']
        )
    elif attack_type.lower() == 'onepixel':
        attack_generator = torchattacks.OnePixel(
            classifier_for_attack_calculation,
            attack_params['pixel_count'], attack_params['max_iter'], attack_params['popsize']
        )
    elif attack_type.lower() == 'deepfool':
        attack_generator = torchattacks.DeepFool(
            classifier_for_attack_calculation, attack_params['steps'],
            attack_params['overshoot']
        )
    elif attack_type.lower() == 'pixle':
        class_for_pixle_attack = PixleFix # Original: torchattacks.Pixle; Modification: PixleFix
        attack_generator = class_for_pixle_attack(
            classifier_for_attack_calculation,
            attack_params['x_dimensions'], attack_params['y_dimensions'], attack_params['pixel_mapping'],
            attack_params['restarts'], attack_params['max_iter'], attack_params['update_each_iteration']
        )
    elif attack_type.lower() == 'gn':
        attack_generator = lambda im_batch, lab_batch : \
            _generate_gn_attack_images(im_batch, attack_params, normalization=dataset_dict['normalized_dataset'])
    else:
        raise ValueError(f"The attack type '{attack_type}' is not recognized!")
    pass

    for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
        #
        # We perform everything in the device of the network:
        images_batch = images_batch.to(device_classifier_nn)
        labels_batch = labels_batch.to(device_classifier_nn)
        #
        # We generate the attacked images
        perturbed_images_batch = attack_generator(images_batch, labels_batch)
        #
        # And we evaluate the output of the classifier for the adversarial images
        #
        classifier_nn.eval()
        if classifier_sm_surrogate is not None:
            classifier_sm_surrogate.eval()
        pass
        #
        classifier_nn.zero_grad()
        if classifier_sm_surrogate is not None:
            classifier_sm_surrogate.zero_grad()
        pass

        with (torch.no_grad()):
            #
            ###################################################################################
            # Original (unperturbed) images
            ###################################################################################

            output_images_batch = classifier_nn(images_batch)

            # Statistics regarding convergence
            _, warning_lack_of_convergence, _, _ = classifier_nn.get_last_forward_convergence_info()
            cum_non_convergent_for_images += warning_lack_of_convergence.sum(dim=None).item()
            #
            loss_images_batch = loss_function(output_images_batch, labels_batch)
            #
            # Accumulate the values for out-of-the-calculation, including un-averaging the loss
            total_num_images_images += output_images_batch.size(0)
            cum_loss_not_averaged += loss_images_batch.item() * output_images_batch.size(0)

            output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
            right_classifications_batch = (output_labels_batch.to(device_classifier_nn) == labels_batch.to(
                device_classifier_nn)).sum().item()

            cum_right_classifications_not_averaged += right_classifications_batch
            #
            cum_non_convergent_images = 0.0

            ###################################################################################
            # Perturbed (attacked) images
            ###################################################################################

            output_perturbed_images_batch = classifier_nn(perturbed_images_batch)
            #
            # Statistics regarding convergence
            _, warning_lack_of_convergence, _, _ = classifier_nn.get_last_forward_convergence_info()
            cum_non_convergent_for_perturbed_images += warning_lack_of_convergence.sum(dim=None).item()

            loss_perturbed_images_batch = loss_function(output_perturbed_images_batch, labels_batch)
            cum_perturbed_loss_not_averaged += loss_perturbed_images_batch.item() * output_images_batch.size(0)
            output_perturbed_labels_batch = torch.max(output_perturbed_images_batch, dim=-1)[1]
            right_classifications_perturbed_batch = (
                    output_perturbed_labels_batch.to(device_classifier_nn) == labels_batch.to(device_classifier_nn)
            ).sum().item()
            cum_perturbed_right_classifications_not_averaged += right_classifications_perturbed_batch
            #
            #
            # Log the specific images (correct in perturbed, incorrect in original)
            incorrect_indices = (output_labels_batch != labels_batch).nonzero(as_tuple=True)[0]
            correct_perturbed_indices = (output_perturbed_labels_batch == labels_batch).nonzero(as_tuple=True)[0]
            # Convert tensors to numpy arrays
            incorrect_indices_np = incorrect_indices.cpu().numpy()
            correct_perturbed_indices_np = correct_perturbed_indices.cpu().numpy()

            # Find the intersection using numpy
            specific_indices_np = np.intersect1d(incorrect_indices_np, correct_perturbed_indices_np)

            # Convert the result back to a tensor
            specific_indices = torch.tensor(specific_indices_np, device=incorrect_indices.device)

            if verbose in ['medium', 'high']:
                end_batch = "\n" if verbose == 'high' else "\r"
                print((f"   " +
                       f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
                       f"loss={loss_images_batch:.3f}   " +
                       f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
                       f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}   " +
                       f"perturbed_loss={loss_perturbed_images_batch:.3f}   " +
                       f"perturbed_acc={100 * right_classifications_perturbed_batch / output_images_batch.size(0):.3f} %   " +
                       f"perturbed_conv_err={fp_perturbed_conv_err}"),
                      end=end_batch)
            pass
            #
        pass
        #
    pass  # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[dataloader_subset])

    # Generate the summary about convergence
    proportion_non_convergent_for_unperturbed_images = \
        cum_non_convergent_for_images / total_num_images_images
    proportion_non_convergent_for_perturbed_images = \
        cum_non_convergent_for_perturbed_images / total_num_images_images

    # Calculate the final values for the classifier
    summary_classifier_nn = {
        'total_num_images_images': total_num_images_images,
        'loss': cum_loss_not_averaged / total_num_images_images,
        'acc': cum_right_classifications_not_averaged / total_num_images_images,
        'proportion_conv_err': proportion_non_convergent_for_unperturbed_images,
        'perturbed_loss': cum_perturbed_loss_not_averaged / total_num_images_images,
        'perturbed_acc': cum_perturbed_right_classifications_not_averaged / total_num_images_images,
        'perturbed_proportion_conv_err': proportion_non_convergent_for_perturbed_images
    }
    if mlflow_run_id is not None:
        mlflow.log_metrics(summary_classifier_nn, run_id=mlflow_run_id)
    pass
    #
    if verbose in ['high', 'medium']:
        print(f"Summary:{' ' * 40}")
        print(f"   {'attack':<35}{attack_type}, {str(attack_params)}")
        for key in summary_classifier_nn:
            end_particle = " % \n" if ('acc' in key or 'proportion' in key) else "\n"
            factor = 100.0 if 'acc' in key else 1.0
            print(f"   {key:<35}{factor * summary_classifier_nn[key]:.3f}", end=end_particle)
        pass
    pass
    #
    ####################################################################
    # Log the info of the classifier (for easier retrieval)
    ####################################################################
    if mlflow_run_id is not None:
        dict_params_to_log = classifier_nn.fields_to_log
        for key in dict_params_to_log:
            if isinstance(dict_params_to_log[key], (int, float)):
                mlflow.log_metric(key, dict_params_to_log[key], run_id=mlflow_run_id)
            pass
            mlflow.log_params({key: str(dict_params_to_log[key])}, run_id=mlflow_run_id)
        pass
    pass

    # Return the summary or list of summaries of the classifiers
    return summary_classifier_nn





def assess_classifier_for_white_box_aa(
        classifier_nn, dataset_dict, loss_function, attack_type, attack_params, classifier_sm=None, classifier_L0=None,
        validation_on_test_subset=True, subbatch_size=None, mlflow_run_id=None, run_name=None, verbose='medium'):
    """
    The function assesses the performance of a given classifier `classifier_nn` for a given white-box \
    adversarial attack `attack_type` and its parameters `attack_params` using the data provided by the `data_loader`. \
    The loss used for the training should be also provided.

    Parameters
    ----------
    classifier_nn : torch.nn.Module or list[torch.nn.Module] or tuple[torch.nn.Module]
        The classifier neural network to be assessed, if it is a subclass of `nn.Module`, or the family of \
        classifier neural networks to be assessed together, if it is a list/tuple of subclasses of `nn.Module`.
    dataset_dict : LoadedDatasetDict
        The data loader providing the data to be used for the assessment.
    loss_function : ~collections.abc.Callable
        Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
        user-defined functions of *(1)* the output of the network and (2) the target can be used too
    attack_type : str
        The type of the adversarial attack to be used. The possible values are 'FGSM' and 'PGD'
    attack_params : dict
        The dictionary of parameters for the adversarial attack. The parameters depend on the attack type:
        - 'FGSM': {'epsilon': float}
        - 'PGD': {'epsilon': float, 'alpha': float, 'num_iter': int}
    classifier_sm: torch.nn.Module, optional
        The classifier neural network Standard Model to be assessed.
        Default: ``None``
    classifier_L0: torch.nn.Module, optional
        The classifier neural network ibnn_internal with 'lambda' equal 0 to be assessed.
        Default: ``None``
    validation_on_test_subset : bool, optional
        The flag indicating whether the validation should be performed on the test subset of the data loader. \
        If `True`, the test subset is used; if `False`, the validation ('val')subset is used. Default: ``True``
    subbatch_size : int, optional
        The size of the sub-batches to be used for the gradient calculation. If it is not provided the same \
        batch size used in the dataset `dataset_dict` is used. Default: ``None``
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
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of progress printing. \
        Default: ``'medium'``

    Returns
    -------
    float
        The accuracy of the classifier `classifier_nn` for the adversarial attack `attack_type` and its parameters \
        `attack_params` using the data provided by the `data_loader`.
    """

    if verbose == 'high':
        print(f"Training of the classifier network:\n")
        print(str(classifier_nn))
        attributes_to_display = classifier_nn.fields_to_log
        print(f"\n\tFields to log:\n")
        for key in attributes_to_display:
            print(f"\t\t{key}: {attributes_to_display[key]}")
        pass
    pass

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
        raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
                        "only one of them can be provided at a time!")
    pass

    # Field 'dataloader_val' or 'dataloader_test' depending on 'validation_on_test_subset'
    subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'
    subbatch_size = subbatch_size if subbatch_size is not None else dataset_dict[subset_for_validation].batch_size

    # Unify the case of single or multiple classifiers
    flag_initial_classifier_nn_was_list = False
    list_classifiers = []
    if isinstance(classifier_nn, (list, tuple)):
        list_classifiers = list(classifier_nn)
        flag_initial_classifier_nn_was_list = True
    else:
        list_classifiers = [classifier_nn]
        flag_initial_classifier_nn_was_list = False
    pass
    # Check at least if the objects are subclass of nn.Module
    for classifier_nn_i in list_classifiers:
        if not isinstance(classifier_nn_i, nn.Module):
            raise ValueError(f"The classifier must be a subclass of torch.nn.Module! However, " +
                             f"it is of type {type(classifier_nn_i)} and is not!")
        pass
    pass

    # We process the classifiers separately and obtain the following data per classifier
    # - loss for the original images
    # - loss for the adversarial images
    # - accuracy for the original images
    # Finally we additionally combine the result of all classifiers
    list_summary_classifiers = []

    for i, classifier_nn_i in enumerate(list_classifiers):
        #
        if verbose in ['high', 'medium', 'low'] and len(list_classifiers) > 1:
            print(f"CLASSIFIER [{(i + 1):3d}/{len(list_classifiers):3d}]")
        pass
        #
        # Check the device of the classifier
        device_classifier_nn_i = next(iter(classifier_nn_i.parameters())).data.device.type
        classifier_nn_i.eval()
        #
        # Auxiliary variables for the assessment of the classifier as accumulated values from each batch
        total_num_images_images = 0
        #
        cum_loss_not_averaged = 0.0
        cum_perturbed_loss_not_averaged = 0.0
        #
        cum_right_classifications_not_averaged = 0.0
        cum_perturbed_right_classifications_not_averaged = 0.0
        #
        fp_conv_err = 0.0
        fp_perturbed_conv_err = 0.0
        #
        if classifier_sm is not None:
            classifier_sm = classifier_sm.to(device_classifier_nn_i)
            classifier_sm.eval()
            cum_perturbed_loss_not_averaged_sm = 0.0
            cum_perturbed_right_classifications_not_averaged_sm = 0.0
            sm_cum_loss_not_averaged = 0.0
            sm_cum_right_classifications_not_averaged = 0.0
            sm_cum_perturbed_loss_not_averaged = 0.0
            sm_cum_perturbed_right_classifications_not_averaged = 0.0
            sm_cum_perturbed_loss_not_averaged_sm = 0.0
            sm_cum_perturbed_right_classifications_not_averaged_sm = 0.0
            fp_perturbed_conv_err_sm = 0.0
            if classifier_L0 is not None:
                sm_cum_perturbed_loss_not_averaged_L0 = 0.0
                sm_cum_perturbed_right_classifications_not_averaged_L0 = 0.0
        pass
        #
        if classifier_L0 is not None:
            classifier_L0 = classifier_L0.to(device_classifier_nn_i)
            classifier_L0.eval()
            cum_perturbed_loss_not_averaged_L0 = 0.0
            cum_perturbed_right_classifications_not_averaged_L0 = 0.0
            fp_perturbed_conv_err_L0 = 0.0
            L0_cum_loss_not_averaged = 0.0
            L0_cum_right_classifications_not_averaged = 0.0
            L0_cum_perturbed_loss_not_averaged = 0.0
            L0_cum_perturbed_right_classifications_not_averaged = 0.0
            L0_cum_loss_not_averaged_L0 = 0.0
            L0_cum_right_classifications_not_averaged_LO = 0.0
            L0_cum_perturbed_loss_not_averaged_L0 = 0.0
            L0_cum_perturbed_right_classifications_not_averaged_LO = 0.0
            if classifier_sm is not None:
                L0_cum_perturbed_loss_not_averaged_sm = 0.0
                L0_cum_perturbed_right_classifications_not_averaged_sm = 0.0
        pass

        for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
            #
            # We perform everything in the device of the network:
            images_batch = images_batch.to(device_classifier_nn_i)
            labels_batch = labels_batch.to(device_classifier_nn_i)
            #
            if attack_type == 'FGSM':
                # Calculate the output and the gradient of each one of the images of the batch and store them
                output_images_batch = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                grad_images_batch = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                #
                if classifier_sm is not None:
                    output_images_batch_sm = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                    grad_images_batch_sm = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                pass
                #
                if classifier_L0 is not None:
                    output_images_batch_L0 = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                    grad_images_batch_L0 = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                pass
                #
                for start_b in range(0, images_batch.size(0), subbatch_size):
                    end_b = min(start_b + subbatch_size, images_batch.size(0))
                    #

                    if verbose == 'high':
                        print((f"            Gradient calculation: " +
                               f"val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]:   " +
                               f"im [{(start_b + 1):5d}/{images_batch.size(0):5d}]"),
                              end='\r')
                    pass
                    #
                    # Zero the gradient of the network (in case they affect)
                    classifier_nn_i.zero_grad()
                    # Calculate the output of the image and the gradient
                    image_b = images_batch[start_b:end_b].clone().detach()
                    image_b.requires_grad = True
                    output_image_b = classifier_nn_i(image_b)
                    loss_image_b = loss_function(output_image_b, labels_batch[start_b:end_b])
                    # print(f"loss_image_b = {loss_image_b}")
                    loss_image_b.backward()
                    # print(f"image_b = {image_b}")
                    # print(f"image_b.grad = {image_b.grad}")
                    grad_images_batch[start_b:end_b] = image_b.grad.data.clone().detach()
                    #
                    if classifier_sm is not None:
                        classifier_sm.zero_grad()
                        output_image_b_sm = classifier_sm(image_b)
                        loss_image_b_sm = loss_function(output_image_b_sm, labels_batch[start_b:end_b])
                        loss_image_b_sm.backward()
                        grad_images_batch_sm[start_b:end_b] = image_b.grad.data.clone().detach()
                    pass
                    #
                    if classifier_L0 is not None:
                        classifier_L0.zero_grad()
                        output_image_b_L0 = classifier_L0(image_b)
                        loss_image_b_L0 = loss_function(output_image_b_L0, labels_batch[start_b:end_b])
                        loss_image_b_L0.backward()
                        grad_images_batch_L0[start_b:end_b] = image_b.grad.data.clone().detach()
                    pass
                pass
            pass
            #
            classifier_nn_i.eval()
            #
            if classifier_sm is not None:
                classifier_sm.eval()
            pass
            #
            if classifier_L0 is not None:
                classifier_L0.eval()
            pass
            #

            # With the gradient we can calculate the adversarial images of the batch
            if attack_type == 'FGSM':
                perturbed_images_batch = _generate_fgsm_attack_images_from_grad(
                    images_batch, attack_params['epsilon'], grad_images_batch,
                    normalization=dataset_dict['normalized_dataset']
                )
                if classifier_sm is not None:
                    perturbed_images_batch_sm = _generate_fgsm_attack_images_from_grad(
                        images_batch, attack_params['epsilon'], grad_images_batch_sm,
                        normalization=dataset_dict['normalized_dataset']
                    )
                if classifier_L0 is not None:
                    perturbed_images_batch_L0 = _generate_fgsm_attack_images_from_grad(
                        images_batch, attack_params['epsilon'], grad_images_batch_L0,
                        normalization=dataset_dict['normalized_dataset']
                    )
            elif attack_type == 'PGD':
                raise Exception("Only a placeholder for now! Not implemented yet!")
            elif attack_type == 'GN':
                perturbed_images_batch = _generate_gn_attack_images(images_batch, attack_params,
                                                                    normalization=dataset_dict['normalized_dataset']
                                                                    )
            else:
                raise ValueError(f"The attack type '{attack_type}' is not recognized!")
            pass
            #
            # And we evaluate the output of the classifier for the adversarial images
            classifier_nn_i.zero_grad()

            with (torch.no_grad()):
                output_images_batch = classifier_nn_i(images_batch)
                fp_conv_err += int(sum(
                    classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[0].detach().cpu().numpy() == True))

                output_perturbed_images_batch = classifier_nn_i(perturbed_images_batch)
                fp_perturbed_conv_err += int(sum(
                    classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[0].detach().cpu().numpy() == True))
                #
                loss_images_batch = loss_function(output_images_batch, labels_batch)
                loss_perturbed_images_batch = loss_function(output_perturbed_images_batch, labels_batch)
                #
                # Accumulate the values for out-of-the-calculation, including un-averaging the loss
                total_num_images_images += output_images_batch.size(0)
                cum_loss_not_averaged += loss_images_batch.item() * output_images_batch.size(0)
                cum_perturbed_loss_not_averaged += loss_perturbed_images_batch.item() * output_images_batch.size(0)

                output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
                output_perturbed_labels_batch = torch.max(output_perturbed_images_batch, dim=-1)[1]
                right_classifications_batch = (output_labels_batch.to(device_classifier_nn_i) == labels_batch.to(
                    device_classifier_nn_i)).sum().item()
                right_classifications_perturbed_batch = (
                            output_perturbed_labels_batch.to(device_classifier_nn_i) == labels_batch.to(
                        device_classifier_nn_i)).sum().item()

                cum_right_classifications_not_averaged += right_classifications_batch
                cum_perturbed_right_classifications_not_averaged += right_classifications_perturbed_batch
                #
                #
                # Log the specific images (correct in perturbed, incorrect in original)
                incorrect_indices = (output_labels_batch != labels_batch).nonzero(as_tuple=True)[0]
                correct_perturbed_indices = (output_perturbed_labels_batch == labels_batch).nonzero(as_tuple=True)[0]
                # Convert tensors to numpy arrays
                incorrect_indices_np = incorrect_indices.cpu().numpy()
                correct_perturbed_indices_np = correct_perturbed_indices.cpu().numpy()

                # Find the intersection using numpy
                specific_indices_np = np.intersect1d(incorrect_indices_np, correct_perturbed_indices_np)

                # Convert the result back to a tensor
                specific_indices = torch.tensor(specific_indices_np, device=incorrect_indices.device)

                if classifier_sm is not None:
                    output_perturbed_images_batch_sm = classifier_nn_i(perturbed_images_batch_sm)
                    fp_perturbed_conv_err_sm += int(
                        sum(classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[
                                0].detach().cpu().numpy() == True))
                    loss_perturbed_images_batch_sm = loss_function(output_perturbed_images_batch_sm, labels_batch)
                    cum_perturbed_loss_not_averaged_sm += loss_perturbed_images_batch_sm.item() * output_images_batch.size(
                        0)
                    output_perturbed_labels_batch_sm = torch.max(output_perturbed_images_batch_sm, dim=-1)[1]
                    right_classifications_perturbed_batch_sm = (
                                output_perturbed_labels_batch_sm.to(device_classifier_nn_i) == labels_batch.to(
                            device_classifier_nn_i)).sum().item()
                    cum_perturbed_right_classifications_not_averaged_sm += right_classifications_perturbed_batch_sm
                    classifier_sm.zero_grad()
                    sm_output_images_batch = classifier_sm(images_batch)
                    sm_loss_images_batch = loss_function(sm_output_images_batch, labels_batch)
                    sm_output_perturbed_images_batch = classifier_sm(perturbed_images_batch)
                    sm_loss_perturbed_images_batch = loss_function(sm_output_perturbed_images_batch, labels_batch)
                    sm_right_classifications_batch = (torch.max(sm_output_images_batch, dim=-1)[1].to(
                        device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    sm_right_classifications_perturbed_batch = (
                                torch.max(sm_output_perturbed_images_batch, dim=-1)[1].to(
                                    device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    sm_cum_loss_not_averaged += sm_loss_images_batch.item() * output_images_batch.size(0)
                    sm_cum_perturbed_loss_not_averaged += sm_loss_perturbed_images_batch.item() * output_images_batch.size(
                        0)
                    sm_cum_right_classifications_not_averaged += sm_right_classifications_batch
                    sm_cum_perturbed_right_classifications_not_averaged += sm_right_classifications_perturbed_batch
                    sm_output_perturbed_images_batch_sm = classifier_sm(perturbed_images_batch_sm)
                    sm_loss_perturbed_images_batch_sm = loss_function(sm_output_perturbed_images_batch_sm, labels_batch)
                    sm_right_classifications_perturbed_batch_sm = (
                                torch.max(sm_output_perturbed_images_batch_sm, dim=-1)[1].to(
                                    device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    sm_cum_perturbed_loss_not_averaged_sm += sm_loss_perturbed_images_batch_sm.item() * output_images_batch.size(
                        0)
                    sm_cum_perturbed_right_classifications_not_averaged_sm += sm_right_classifications_perturbed_batch_sm
                    if classifier_L0 is not None:
                        sm_output_perturbed_L0_images_batch = classifier_sm(perturbed_images_batch_L0)
                        sm_loss_perturbed_L0_images_batch = loss_function(sm_output_perturbed_L0_images_batch,
                                                                          labels_batch)
                        sm_right_classifications_perturbed_L0_batch = (
                                    torch.max(sm_output_perturbed_L0_images_batch, dim=-1)[1].to(
                                        device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                        sm_cum_perturbed_loss_not_averaged_L0 += sm_loss_perturbed_L0_images_batch.item() * output_images_batch.size(
                            0)
                        sm_cum_perturbed_right_classifications_not_averaged_L0 += sm_right_classifications_perturbed_L0_batch
                pass
                #
                if classifier_L0 is not None:
                    #
                    output_perturbed_images_batch_L0 = classifier_nn_i(perturbed_images_batch_L0)
                    fp_perturbed_conv_err_L0 += int(
                        sum(classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[
                                0].detach().cpu().numpy() == True))
                    #
                    loss_perturbed_images_batch_L0 = loss_function(output_perturbed_images_batch_L0, labels_batch)
                    cum_perturbed_loss_not_averaged_L0 += loss_perturbed_images_batch_L0.item() * output_images_batch.size(
                        0)
                    #
                    output_perturbed_labels_batch_L0 = torch.max(output_perturbed_images_batch_L0, dim=-1)[1]
                    right_classifications_perturbed_batch_L0 = (
                                output_perturbed_labels_batch_L0.to(device_classifier_nn_i) == labels_batch.to(
                            device_classifier_nn_i)).sum().item()
                    cum_perturbed_right_classifications_not_averaged_L0 += right_classifications_perturbed_batch_L0
                    #
                    classifier_L0.zero_grad()
                    #
                    L0_output_images_batch = classifier_L0(images_batch)
                    #
                    L0_loss_images_batch = loss_function(L0_output_images_batch, labels_batch)
                    L0_output_perturbed_images_batch = classifier_L0(perturbed_images_batch)
                    L0_loss_perturbed_images_batch = loss_function(L0_output_perturbed_images_batch, labels_batch)
                    L0_right_classifications_batch = (torch.max(L0_output_images_batch, dim=-1)[1].to(
                        device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    L0_right_classifications_perturbed_batch = (
                                torch.max(L0_output_perturbed_images_batch, dim=-1)[1].to(
                                    device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    #
                    L0_cum_loss_not_averaged += L0_loss_images_batch.item() * output_images_batch.size(0)
                    L0_cum_loss_not_averaged += L0_loss_perturbed_images_batch.item() * output_images_batch.size(0)
                    L0_cum_right_classifications_not_averaged += L0_right_classifications_batch
                    L0_cum_perturbed_right_classifications_not_averaged += L0_right_classifications_perturbed_batch
                    #
                    L0_output_perturbed_L0_images_batch = classifier_L0(perturbed_images_batch_L0)
                    #
                    L0_loss_perturbed_L0_images_batch = loss_function(L0_output_perturbed_L0_images_batch, labels_batch)
                    L0_right_classifications_perturbed_L0_batch = (
                                torch.max(L0_output_perturbed_L0_images_batch, dim=-1)[1].to(
                                    device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                    #
                    L0_cum_loss_not_averaged_L0 += L0_loss_perturbed_L0_images_batch.item() * output_images_batch.size(
                        0)
                    L0_cum_perturbed_right_classifications_not_averaged_LO += L0_right_classifications_perturbed_L0_batch

                    if classifier_sm is not None:
                        L0_output_perturbed_sm_images_batch = classifier_L0(perturbed_images_batch_sm)
                        L0_loss_perturbed_sm_images_batch = loss_function(L0_output_perturbed_sm_images_batch,
                                                                          labels_batch)
                        L0_right_classifications_perturbed_sm_batch = (
                                    torch.max(L0_output_perturbed_sm_images_batch, dim=-1)[1].to(
                                        device_classifier_nn_i) == labels_batch.to(device_classifier_nn_i)).sum().item()
                        L0_cum_perturbed_loss_not_averaged_sm += L0_loss_perturbed_sm_images_batch.item() * output_images_batch.size(
                            0)
                        L0_cum_perturbed_right_classifications_not_averaged_sm += L0_right_classifications_perturbed_sm_batch

                pass
                if specific_indices.numel() > 0:
                    log_side_by_side_images(images_batch, perturbed_images_batch, specific_indices, num_batch + 1,
                                            "incorrect_to_correct", perturbed_images_sm=None,
                                            labels=labels_batch,
                                            predicted_labels=output_labels_batch,
                                            predicted_labels_perturbed=output_perturbed_labels_batch
                                            )
                    if classifier_sm is not None:
                        log_side_by_side_images(images_batch, perturbed_images_batch, specific_indices, num_batch + 1,
                                                "incorrect_to_correct",
                                                perturbed_images_sm=perturbed_images_batch_sm,
                                                labels=labels_batch,
                                                predicted_labels=output_labels_batch,
                                                predicted_labels_perturbed=output_perturbed_labels_batch,
                                                predicted_labels_perturbed_sm=output_perturbed_labels_batch_sm
                                                )
                pass

                #

                if verbose in ['medium', 'high']:
                    end_batch = "\n" if verbose == 'high' else "\r"
                    if classifier_sm is not None:
                        print((f"   " +
                               f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
                               f"loss={loss_images_batch:.3f}   " +
                               f"perturbed_loss={loss_perturbed_images_batch:.3f}   " +
                               f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
                               f"perturbed_acc={100 * right_classifications_perturbed_batch / output_images_batch.size(0):.3f} %" +
                               f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}" +
                               f"perturbed_loss_sm={loss_perturbed_images_batch_sm:.3f}   " +
                               f"perturbed_acc_sm={100 * right_classifications_perturbed_batch_sm / output_images_batch.size(0):.3f} %" +
                               f"perturbed_conv_err_sm={fp_perturbed_conv_err_sm}"),
                              end=end_batch)
                    else:
                        print((f"   " +
                               f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
                               f"loss={loss_images_batch:.3f}   " +
                               f"perturbed_loss={loss_perturbed_images_batch:.3f}   " +
                               f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
                               f"perturbed_acc={100 * right_classifications_perturbed_batch / output_images_batch.size(0):.3f} %" +
                               f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}"
                               ),
                              end=end_batch)
                pass
                if mlflow_run_id is not None:
                    # Log the input images
                    log_random_images(perturbed_images_batch, num_batch + 1)
                pass
                #
            pass
            #
        pass  # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[dataloader_subset])

        # Calculate the final values for the classifier
        summary_classifier_nn_i = {
            'total_num_images_images': total_num_images_images,
            'loss': cum_loss_not_averaged / total_num_images_images,
            'perturbed_loss': cum_perturbed_loss_not_averaged / total_num_images_images,
            'acc': cum_right_classifications_not_averaged / total_num_images_images,
            'perturbed_acc': cum_perturbed_right_classifications_not_averaged / total_num_images_images,
            'fp_conv_err': fp_conv_err / total_num_images_images,
            'fp_perturbed_conv_err': fp_perturbed_conv_err / total_num_images_images
        }
        if classifier_sm is not None:
            summary_classifier_nn_i.update({
                'perturbed_loss_sm': cum_perturbed_loss_not_averaged_sm / total_num_images_images,
                'perturbed_acc_sm': cum_perturbed_right_classifications_not_averaged_sm / total_num_images_images,
                'fp_perturbed_conv_err_sm': fp_perturbed_conv_err_sm / total_num_images_images
            })
        if mlflow_run_id is not None:
            mlflow.log_metrics(summary_classifier_nn_i,
                               run_id=mlflow_run_id)
            mlflow.log_params({'attack_type': attack_type, 'attack_params': attack_params}, run_id=mlflow_run_id)
        pass
        # Append the summary of the classifier to the list of summaries
        list_summary_classifiers.append(summary_classifier_nn_i)
        #
        if verbose in ['high', 'medium']:
            print(f"Summary:{' ' * 40}")
            for key in summary_classifier_nn_i:
                end_particle = " % \n" if 'acc' in key else "\n"
                factor = 100.0 if 'acc' in key else 1.0
                print(f"   {key:<15}  {factor * summary_classifier_nn_i[key]:.3f}", end=end_particle)
            pass
        pass
        #

    pass

    # Return the summary or list of summaries of the classifiers
    return list_summary_classifiers if flag_initial_classifier_nn_was_list else list_summary_classifiers[0]


def assess_classifier_for_white_box_aa_sm(
        classifier_nn, dataset_dict, loss_function, attack_type, attack_params, classifier_sm,
        validation_on_test_subset=True, subbatch_size=None, mlflow_run_id=None, run_name=None, verbose='medium'):
    """
    The function assesses the performance of a given classifier `classifier_nn` for a given white-box \
    adversarial attack `attack_type` and its parameters `attack_params` using the data provided by the `data_loader`. \
    The loss used for the training should be also provided.

    Parameters
    ----------
    classifier_nn : torch.nn.Module or list[torch.nn.Module] or tuple[torch.nn.Module]
        The classifier neural network to be assessed, if it is a subclass of `nn.Module`, or the family of \
        classifier neural networks to be assessed together, if it is a list/tuple of subclasses of `nn.Module`.
    dataset_dict : LoadedDatasetDict
        The data loader providing the data to be used for the assessment.
    loss_function : ~collections.abc.Callable
        Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
        user-defined functions of *(1)* the output of the network and (2) the target can be used too
    attack_type : str
        The type of the adversarial attack to be used. The possible values are 'FGSM' and 'PGD'
    attack_params : dict
        The dictionary of parameters for the adversarial attack. The parameters depend on the attack type:
        - 'FGSM': {'epsilon': float}
        - 'PGD': {'epsilon': float, 'alpha': float, 'num_iter': int}
    classifier_sm: torch.nn.Module, optional
        The classifier neural network Standard Model to be assessed.
        Default: ``None``
    validation_on_test_subset : bool, optional
        The flag indicating whether the validation should be performed on the test subset of the data loader. \
        If `True`, the test subset is used; if `False`, the validation ('val')subset is used. Default: ``True``
    subbatch_size : int, optional
        The size of the sub-batches to be used for the gradient calculation. If it is not provided the same \
        batch size used in the dataset `dataset_dict` is used. Default: ``None``
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
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of progress printing. \
        Default: ``'medium'``

    Returns
    -------
    float
        The accuracy of the classifier `classifier_nn` for the adversarial attack `attack_type` and its parameters \
        `attack_params` using the data provided by the `data_loader`.
    """

    if verbose == 'high':
        print(f"Training of the classifier network:\n")
        print(str(classifier_nn))
        attributes_to_display = classifier_nn.fields_to_log
        print(f"\n\tFields to log:\n")
        for key in attributes_to_display:
            print(f"\t\t{key}: {attributes_to_display[key]}")
        pass
    pass

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
        raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
                        "only one of them can be provided at a time!")
    pass

    # Field 'dataloader_val' or 'dataloader_test' depending on 'validation_on_test_subset'
    subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'
    subbatch_size = subbatch_size if subbatch_size is not None else dataset_dict[subset_for_validation].batch_size

    # Unify the case of single or multiple classifiers
    flag_initial_classifier_nn_was_list = False
    list_classifiers = []
    if isinstance(classifier_nn, (list, tuple)):
        list_classifiers = list(classifier_nn)
        flag_initial_classifier_nn_was_list = True
    else:
        list_classifiers = [classifier_nn]
        flag_initial_classifier_nn_was_list = False
    pass
    # Check at least if the objects are subclass of nn.Module
    for classifier_nn_i in list_classifiers:
        if not isinstance(classifier_nn_i, nn.Module):
            raise ValueError(f"The classifier must be a subclass of torch.nn.Module! However, " +
                             f"it is of type {type(classifier_nn_i)} and is not!")
        pass
    pass

    # We process the classifiers separately and obtain the following data per classifier
    # - loss for the original images
    # - loss for the adversarial images
    # - accuracy for the original images
    # Finally we additionally combine the result of all classifiers
    list_summary_classifiers = []

    for i, classifier_nn_i in enumerate(list_classifiers):
        #
        if verbose in ['high', 'medium', 'low'] and len(list_classifiers) > 1:
            print(f"CLASSIFIER [{(i + 1):3d}/{len(list_classifiers):3d}]")
        pass
        #
        # Check the device of the classifier
        device_classifier_nn_i = next(iter(classifier_nn_i.parameters())).data.device.type
        #
        # Auxiliary variables for the assessment of the classifier as accumulated values from each batch
        total_num_images_images = 0
        #
        cum_loss_not_averaged = 0.0
        cum_perturbed_loss_not_averaged = 0.0
        #
        cum_right_classifications_not_averaged = 0.0
        cum_perturbed_right_classifications_not_averaged = 0.0
        #
        fp_conv_err = 0.0
        fp_perturbed_conv_err = 0.0
        #
        if classifier_sm is not None:
            classifier_sm = classifier_sm.to(device_classifier_nn_i)
            classifier_sm.eval()
            cum_perturbed_loss_not_averaged_sm = 0.0
            cum_perturbed_right_classifications_not_averaged_sm = 0.0
            sm_cum_loss_not_averaged = 0.0
            sm_cum_right_classifications_not_averaged = 0.0
            sm_cum_perturbed_loss_not_averaged = 0.0
            sm_cum_perturbed_right_classifications_not_averaged = 0.0
            sm_cum_perturbed_loss_not_averaged_sm = 0.0
            sm_cum_perturbed_right_classifications_not_averaged_sm = 0.0
            fp_perturbed_conv_err_sm = 0.0
        pass

        for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
            #
            # We perform everything in the device of the network:
            images_batch = images_batch.to(device_classifier_nn_i)
            labels_batch = labels_batch.to(device_classifier_nn_i)
            #
            if attack_type == 'FGSM':
                # Calculate the output and the gradient of each one of the images of the batch and store them
                output_images_batch_sm = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                grad_images_batch_sm = torch.empty(images_batch.size(), device=device_classifier_nn_i)
                #
                for start_b in range(0, images_batch.size(0), subbatch_size):
                    end_b = min(start_b + subbatch_size, images_batch.size(0))
                    #

                    if verbose == 'high':
                        print((f"            Gradient calculation: " +
                               f"val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]:   " +
                               f"im [{(start_b + 1):5d}/{images_batch.size(0):5d}]"),
                              end='\r')
                    pass
                    #
                    # Zero the gradient of the network (in case they affect)
                    classifier_sm.zero_grad()  # Calculate the output of the image and the gradient
                    image_b = images_batch[start_b:end_b].clone().detach()
                    image_b.requires_grad = True

                    output_image_b_sm = classifier_sm(image_b)
                    loss_image_b_sm = loss_function(output_image_b_sm, labels_batch[start_b:end_b])
                    loss_image_b_sm.backward()
                    grad_images_batch_sm[start_b:end_b] = image_b.grad.data.clone().detach()
                pass
            pass

            # With the gradient we can calculate the adversarial images of the batch
            if attack_type == 'FGSM':
                perturbed_images_batch_sm = _generate_fgsm_attack_images_from_grad(
                    images_batch, attack_params['epsilon'], grad_images_batch_sm,
                    normalization=dataset_dict['normalized_dataset']
                )
            elif attack_type == 'PGD':
                raise Exception("Only a placeholder for now! Not implemented yet!")
            elif attack_type == 'GN':
                perturbed_images_batch = _generate_gn_attack_images(images_batch, attack_params,
                                                                    normalization=dataset_dict['normalized_dataset']
                                                                    )
            else:
                raise ValueError(f"The attack type '{attack_type}' is not recognized!")
            pass
            #
            # And we evaluate the output of the classifier for the adversarial images
            #
            classifier_nn_i.eval()
            classifier_sm.eval()
            #
            classifier_nn_i.zero_grad()

            with (torch.no_grad()):
                output_images_batch = classifier_nn_i(images_batch)
                fp_conv_err += int(sum(
                    classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[0].detach().cpu().numpy() == True))

                #
                loss_images_batch = loss_function(output_images_batch, labels_batch)
                #
                # Accumulate the values for out-of-the-calculation, including un-averaging the loss
                total_num_images_images += output_images_batch.size(0)
                cum_loss_not_averaged += loss_images_batch.item() * output_images_batch.size(0)

                output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
                right_classifications_batch = (output_labels_batch.to(device_classifier_nn_i) == labels_batch.to(
                    device_classifier_nn_i)).sum().item()

                cum_right_classifications_not_averaged += right_classifications_batch

                output_perturbed_images_batch_sm = classifier_nn_i(perturbed_images_batch_sm)
                fp_perturbed_conv_err_sm += int(sum(classifier_nn_i._nn[1][0][0].get_last_forward_convergence_info()[
                                                        0].detach().cpu().numpy() == True))
                loss_perturbed_images_batch_sm = loss_function(output_perturbed_images_batch_sm, labels_batch)
                cum_perturbed_loss_not_averaged_sm += loss_perturbed_images_batch_sm.item() * output_images_batch.size(
                    0)
                output_perturbed_labels_batch_sm = torch.max(output_perturbed_images_batch_sm, dim=-1)[1]
                right_classifications_perturbed_batch_sm = (
                            output_perturbed_labels_batch_sm.to(device_classifier_nn_i) == labels_batch.to(
                        device_classifier_nn_i)).sum().item()
                cum_perturbed_right_classifications_not_averaged_sm += right_classifications_perturbed_batch_sm
                #
                #
                # Log the specific images (correct in perturbed, incorrect in original)
                incorrect_indices = (output_labels_batch != labels_batch).nonzero(as_tuple=True)[0]
                correct_perturbed_indices = (output_perturbed_labels_batch_sm == labels_batch).nonzero(as_tuple=True)[0]
                # Convert tensors to numpy arrays
                incorrect_indices_np = incorrect_indices.cpu().numpy()
                correct_perturbed_indices_np = correct_perturbed_indices.cpu().numpy()

                # Find the intersection using numpy
                specific_indices_np = np.intersect1d(incorrect_indices_np, correct_perturbed_indices_np)

                # Convert the result back to a tensor
                specific_indices = torch.tensor(specific_indices_np, device=incorrect_indices.device)

                if specific_indices.numel() > 0:
                    log_side_by_side_images(images_batch, perturbed_images_batch_sm, specific_indices, num_batch + 1,
                                            "incorrect_to_correct", perturbed_images_sm=None,
                                            labels=labels_batch,
                                            predicted_labels=output_labels_batch,
                                            predicted_labels_perturbed=output_perturbed_labels_batch_sm
                                            )
                pass

                #

                if verbose in ['medium', 'high']:
                    end_batch = "\n" if verbose == 'high' else "\r"
                    print((f"   " +
                           f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
                           f"loss={loss_images_batch:.3f}   " +
                           f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
                           f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}" +
                           f"perturbed_loss_sm={loss_perturbed_images_batch_sm:.3f}   " +
                           f"perturbed_acc_sm={100 * right_classifications_perturbed_batch_sm / output_images_batch.size(0):.3f} %" +
                           f"perturbed_conv_err_sm={fp_perturbed_conv_err_sm}"),
                          end=end_batch)
                pass
                #
            pass
            #
        pass  # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[dataloader_subset])

        # Calculate the final values for the classifier
        summary_classifier_nn_i = {
            'total_num_images_images': total_num_images_images,
            'loss': cum_loss_not_averaged / total_num_images_images,
            'acc': cum_right_classifications_not_averaged / total_num_images_images,
            'fp_conv_err': fp_conv_err / total_num_images_images,
        }
        summary_classifier_nn_i.update({
            'perturbed_loss_sm': cum_perturbed_loss_not_averaged_sm / total_num_images_images,
            'perturbed_acc_sm': cum_perturbed_right_classifications_not_averaged_sm / total_num_images_images,
            'fp_perturbed_conv_err_sm': fp_perturbed_conv_err_sm / total_num_images_images
        })
        if mlflow_run_id is not None:
            mlflow.log_metrics(summary_classifier_nn_i,
                               run_id=mlflow_run_id)
            mlflow.log_params({'attack_type': attack_type, 'attack_params': attack_params}, run_id=mlflow_run_id)
        pass
        # Append the summary of the classifier to the list of summaries
        list_summary_classifiers.append(summary_classifier_nn_i)
        #
        if verbose in ['high', 'medium']:
            print(f"Summary:{' ' * 40}")
            for key in summary_classifier_nn_i:
                end_particle = " % \n" if 'acc' in key else "\n"
                factor = 100.0 if 'acc' in key else 1.0
                print(f"   {key:<15}  {factor * summary_classifier_nn_i[key]:.3f}", end=end_particle)
            pass
        pass
        #

    pass

    # Return the summary or list of summaries of the classifiers
    return list_summary_classifiers if flag_initial_classifier_nn_was_list else list_summary_classifiers[0]


def assess_classifier_for_white_box_aa_sm_with_torchattacks(
        classifier_nn, dataset_dict, loss_function, attack_type, attack_params, classifier_sm_surrogate,
        validation_on_test_subset=True, subbatch_size=None, mlflow_run_id=None, run_name=None, verbose='medium'):
    """
    The function assesses the performance of a given classifier `classifier_nn` for a given white-box \
    adversarial attack `attack_type` and its parameters `attack_params` using the data provided by the `data_loader`. \
    The loss used for the training should be also provided.

    Parameters
    ----------
    classifier_nn : torch.nn.Module or list[torch.nn.Module] or tuple[torch.nn.Module]
        The classifier neural network to be assessed, if it is a subclass of `nn.Module`, or the family of \
        classifier neural networks to be assessed together, if it is a list/tuple of subclasses of `nn.Module`.
    dataset_dict : LoadedDatasetDict
        The data loader providing the data to be used for the assessment.
    loss_function : ~collections.abc.Callable
        Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
        user-defined functions of *(1)* the output of the network and (2) the target can be used too
    attack_type : str
        The type of the adversarial attack to be used. The possible values are 'FGSM' and 'PGD'
    attack_params : dict
        The dictionary of parameters for the adversarial attack. The parameters depend on the attack type:
        - 'FGSM': {'epsilon': float}
        - 'PGD': {'epsilon': float, 'alpha': float, 'num_iter': int}
        - 'OnePixel': {'pixel_count': int, 'max_iter': int, 'popsize': int}
    classifier_sm_surrogate: torch.nn.Module, optional
        The classifier neural network Standard Model to be assessed.
        Default: ``None``
    validation_on_test_subset : bool, optional
        The flag indicating whether the validation should be performed on the test subset of the data loader. \
        If `True`, the test subset is used; if `False`, the validation ('val')subset is used. Default: ``True``
    subbatch_size : int, optional
        The size of the sub-batches to be used for the gradient calculation. If it is not provided the same \
        batch size used in the dataset `dataset_dict` is used. Default: ``None``
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
    verbose : str, optional
        Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of progress printing. \
        Default: ``'medium'``

    Returns
    -------
    float
        The accuracy of the classifier `classifier_nn` for the adversarial attack `attack_type` and its parameters \
        `attack_params` using the data provided by the `data_loader`.
    """

    if verbose == 'high':
        print(f"Training of the classifier network:\n")
        print(str(classifier_nn))
        attributes_to_display = classifier_nn.fields_to_log
        print(f"\n\tFields to log:\n")
        for key in attributes_to_display:
            print(f"\t\t{key}: {attributes_to_display[key]}")
        pass
    pass

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
        raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
                        "only one of them can be provided at a time!")
    pass

    ###################################################
    # Log the parameters of the attack (metric, when possible, and always, as param)
    ###################################################

    # Log the dataset info
    if mlflow_run_id is not None:
        mlflow_log_dataset(dataset_dict, mlflow_run_id)
    pass

    if mlflow_run_id is not None:
        mlflow.log_params({'attack_type': attack_type}, run_id=mlflow_run_id)
        for key in attack_params:
            if isinstance(attack_params[key], (int, float)):
                mlflow.log_metrics({f'attack_parameter_{key}': attack_params[key]}, run_id=mlflow_run_id)
            pass
            mlflow.log_params({f'attack_parameter_{key}': str(attack_params[key])}, run_id=mlflow_run_id)
        pass
    pass

    # Field 'dataloader_val' or 'dataloader_test' depending on 'validation_on_test_subset'
    subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'
    subbatch_size = subbatch_size if subbatch_size is not None else dataset_dict[subset_for_validation].batch_size

    # Check at least if the objects are subclass of nn.Module
    if not isinstance(classifier_nn, nn.Module):
        raise ValueError(f"The classifier must be a subclass of torch.nn.Module! However, " +
                         f"it is of type {type(classifier_nn)} and is not!")
    pass

    # We process the classifiers separately and obtain the following data per classifier
    # - loss for the original images
    # - loss for the adversarial images
    # - accuracy for the original images
    # Finally we additionally combine the result of all classifiers
    list_summary_classifiers = []

    # Check the device of the classifier
    device_classifier_nn = next(iter(classifier_nn.parameters())).data.device.type
    #
    # Auxiliary variables for the assessment of the classifier as accumulated values from each batch
    total_num_images_images = 0
    #
    cum_loss_not_averaged = 0.0
    cum_perturbed_loss_not_averaged = 0.0
    #
    cum_right_classifications_not_averaged = 0.0
    cum_perturbed_right_classifications_not_averaged = 0.0
    #
    cum_non_convergent_for_unperturbed_images = 0.0
    cum_non_convergent_for_perturbed_images = 0.0
    #
    fp_conv_err = 0.0
    fp_perturbed_conv_err = 0.0
    #
    if classifier_sm_surrogate is None:
        raise Exception("Either no `classifier_sm_surrogate`, or an invalid object (None), has been provided.")
    else:
        classifier_sm_surrogate = classifier_sm_surrogate.to(device_classifier_nn)
        classifier_sm_surrogate.eval()
        cum_perturbed_loss_not_averaged_sm = 0.0
        cum_perturbed_right_classifications_not_averaged_sm = 0.0
        sm_cum_loss_not_averaged = 0.0
        sm_cum_right_classifications_not_averaged = 0.0
        sm_cum_perturbed_loss_not_averaged = 0.0
        sm_cum_perturbed_right_classifications_not_averaged = 0.0
        sm_cum_perturbed_loss_not_averaged_sm = 0.0
        sm_cum_perturbed_right_classifications_not_averaged_sm = 0.0
        fp_perturbed_conv_err_sm = 0.0
    pass

    for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
        #
        # We perform everything in the device of the network:
        images_batch = images_batch.to(device_classifier_nn)
        labels_batch = labels_batch.to(device_classifier_nn)
        #

        # With the gradient we can calculate the adversarial images of the batch
        if attack_type.lower() == 'fgsm':
            attack = torchattacks.FGSM(classifier_sm_surrogate, attack_params['epsilon'])
            perturbed_images_batch_sm = attack(images_batch, labels_batch)
        elif attack_type.lower() == 'onepixel':
            attack = torchattacks.OnePixel(classifier_sm_surrogate, attack_params['pixel_count'],
                                           attack_params['max_iter'], attack_params['popsize'])
            perturbed_images_batch_sm = attack(images_batch, labels_batch)
        elif attack_type.lower() == 'deepfool':
            attack = torchattacks.DeepFool(classifier_sm_surrogate, attack_params['steps'], attack_params['overshoot'])
            perturbed_images_batch_sm = attack(images_batch, labels_batch)
        elif attack_type.lower() == 'pgd':
            attack = torchattacks.PGD(classifier_sm_surrogate, attack_params['epsilon'], attack_params['alpha'],
                                      attack_params['num_iter'])
            perturbed_images_batch_sm = attack(images_batch, labels_batch)
        elif attack_type.lower() == 'pixle':
            attack = torchattacks.Pixle(classifier_sm_surrogate, attack_params['x_dimensions'],
                                        attack_params['y_dimensions'], attack_params['pixel_mapping'],
                                        attack_params['restarts'], attack_params['max_iter'],
                                        attack_params['update_each_iteration'])
            perturbed_images_batch_sm = attack(images_batch, labels_batch)
        elif attack_type.lower() == 'gn':
            perturbed_images_batch_sm = _generate_gn_attack_images(images_batch, attack_params,
                                                                   normalization=dataset_dict['normalized_dataset'])
        else:
            raise ValueError(f"The attack type '{attack_type}' is not recognized!")
        pass
        #
        # And we evaluate the output of the classifier for the adversarial images
        #
        classifier_nn.eval()
        classifier_sm_surrogate.eval()
        #
        classifier_nn.zero_grad()

        with (torch.no_grad()):
            #
            output_images_batch = classifier_nn(images_batch)
            #
            # Statistics regarding convergence
            _, warning_lack_of_convergence, _, _ = classifier_nn.get_last_forward_convergence_info()
            cum_non_convergent_for_unperturbed_images += warning_lack_of_convergence.sum(dim=None).item()
            #
            loss_images_batch = loss_function(output_images_batch, labels_batch)
            #
            # Accumulate the values for out-of-the-calculation, including un-averaging the loss
            total_num_images_images += output_images_batch.size(0)
            cum_loss_not_averaged += loss_images_batch.item() * output_images_batch.size(0)

            output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
            right_classifications_batch = (output_labels_batch.to(device_classifier_nn) == labels_batch.to(
                device_classifier_nn)).sum().item()

            cum_right_classifications_not_averaged += right_classifications_batch
            #
            cum_non_convergent_images = 0.0

            output_perturbed_images_batch_sm = classifier_nn(perturbed_images_batch_sm)
            #
            # Statistics regarding convergence
            _, warning_lack_of_convergence, _, _ = classifier_nn.get_last_forward_convergence_info()
            cum_non_convergent_for_perturbed_images += warning_lack_of_convergence.sum(dim=None).item()

            loss_perturbed_images_batch_sm = loss_function(output_perturbed_images_batch_sm, labels_batch)
            cum_perturbed_loss_not_averaged_sm += loss_perturbed_images_batch_sm.item() * output_images_batch.size(0)
            output_perturbed_labels_batch_sm = torch.max(output_perturbed_images_batch_sm, dim=-1)[1]
            right_classifications_perturbed_batch_sm = (
                    output_perturbed_labels_batch_sm.to(device_classifier_nn) == labels_batch.to(device_classifier_nn)
            ).sum().item()
            cum_perturbed_right_classifications_not_averaged_sm += right_classifications_perturbed_batch_sm
            #
            #
            # Log the specific images (correct in perturbed, incorrect in original)
            incorrect_indices = (output_labels_batch != labels_batch).nonzero(as_tuple=True)[0]
            correct_perturbed_indices = (output_perturbed_labels_batch_sm == labels_batch).nonzero(as_tuple=True)[0]
            # Convert tensors to numpy arrays
            incorrect_indices_np = incorrect_indices.cpu().numpy()
            correct_perturbed_indices_np = correct_perturbed_indices.cpu().numpy()

            # Find the intersection using numpy
            specific_indices_np = np.intersect1d(incorrect_indices_np, correct_perturbed_indices_np)

            # Convert the result back to a tensor
            specific_indices = torch.tensor(specific_indices_np, device=incorrect_indices.device)

            # if specific_indices.numel() > 0:
            #     log_side_by_side_images(images_batch, perturbed_images_batch_sm, specific_indices, num_batch + 1,
            #                             "incorrect_to_correct", perturbed_images_sm=None,
            #                             labels=labels_batch,
            #                             predicted_labels=output_labels_batch,
            #                             predicted_labels_perturbed=output_perturbed_labels_batch_sm
            #                             )
            # pass
            #
            if verbose in ['medium', 'high']:
                end_batch = "\n" if verbose == 'high' else "\r"
                print((f"   " +
                       f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
                       f"loss={loss_images_batch:.3f}   " +
                       f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
                       f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}   " +
                       f"perturbed_loss_sm={loss_perturbed_images_batch_sm:.3f}   " +
                       f"perturbed_acc_sm={100 * right_classifications_perturbed_batch_sm / output_images_batch.size(0):.3f} %   " +
                       f"perturbed_conv_err_sm={fp_perturbed_conv_err_sm}"),
                      end=end_batch)
            pass
            #
        pass
        #
    pass  # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[dataloader_subset])

    # Generate the summary about convergence
    proportion_non_convergent_for_unperturbed_images = \
        cum_non_convergent_for_unperturbed_images / total_num_images_images
    proportion_non_convergent_for_perturbed_images = \
        cum_non_convergent_for_perturbed_images / total_num_images_images

    # Calculate the final values for the classifier
    summary_classifier_nn = {
        'total_num_images_images': total_num_images_images,
        'loss': cum_loss_not_averaged / total_num_images_images,
        'acc': cum_right_classifications_not_averaged / total_num_images_images,
        'proportion_conv_err': proportion_non_convergent_for_unperturbed_images,
        'perturbed_loss_sm': cum_perturbed_loss_not_averaged_sm / total_num_images_images,
        'perturbed_acc_sm': cum_perturbed_right_classifications_not_averaged_sm / total_num_images_images,
        'perturbed_proportion_conv_err_sm': proportion_non_convergent_for_perturbed_images
    }
    if mlflow_run_id is not None:
        mlflow.log_metrics(summary_classifier_nn, run_id=mlflow_run_id)
    pass
    #
    if verbose in ['high', 'medium']:
        print(f"Summary:{' ' * 40}")
        print(f"   {'attack':<35}{attack_type}, {str(attack_params)}")
        for key in summary_classifier_nn:
            end_particle = " % \n" if ('acc' in key or 'proportion' in key) else "\n"
            factor = 100.0 if 'acc' in key else 1.0
            print(f"   {key:<35}{factor * summary_classifier_nn[key]:.3f}", end=end_particle)
        pass
    pass
    #
    ####################################################################
    # Log the info of the classifier (for easier retrieval)
    ####################################################################
    if mlflow_run_id is not None:
        dict_params_to_log = classifier_nn.fields_to_log
        for key in dict_params_to_log:
            if isinstance(dict_params_to_log[key], (int, float)):
                mlflow.log_metric(key, dict_params_to_log[key], run_id=mlflow_run_id)
            pass
            mlflow.log_params({key: str(dict_params_to_log[key])}, run_id=mlflow_run_id)
        pass
    pass

    # Return the summary or list of summaries of the classifiers
    return summary_classifier_nn

# def assess_classifier_for_white_box_aa_hyperparams_with_torchattacks(
#         examined_classifier, dataset_dict, loss_function, attack_type, attack_params,
#         baseline_classifier=None, classifier_for_aa_generation=None,
#         validation_on_test_subset=True, subbatch_size=None, mlflow_run_id=None, run_name=None, verbose='medium'):
#     """
#     The function assesses the performance of a given classifier `examined_classifier` for a given white-box \
#     adversarial attack `attack_type` and its parameters `attack_params` using the data provided by the `data_loader`. \
#     The loss used for the training should be also provided.
#
#     The attacks are indicated using the `attack_type`, which takes one single string value among the options \
#     ``'FGSM'``, ``'OnePixel'``, ``'GN'``, and ``'DeepFool'``, and **with parameters expressed as a dictionary** \
#     whose keys depend on the attack type and **whose values, when lists, are interpreted as multiple options to** \
#     **combine and try (i.e. as a cartesian product).**
#
#     The function evaluates the compulsory classifier `examined_classifier` but with two potential options:
#
#     - If a `baseline_classifier` (defaulting ``None``) is provided then the baseline model is also evaluated. \
#       The typical use of this option is to provide a classifier analogous to the `examined_classifier` of interest \
#       but simpler in some aspect: e.g. same architecture of `examined_classifier` but with standard model at all layers.
#
#     - If a `classifier_for_aa_generation` (defaulting ``None``) is provided then the adversarially-attacked images \
#       are generated using such common model; if not provided, `examined_classifier` and, if provided, `baseline_classifier`, \
#       are evaluated using adversarial images generated using themselves as generator.
#
#     Parameters
#     ----------
#     examined_classifier : torch.nn.Module or list[torch.nn.Module] or tuple[torch.nn.Module]
#         The classifier neural network to be assessed, if it is a subclass of `nn.Module`, or the family of \
#         classifier neural networks to be assessed together, if it is a list/tuple of subclasses of `nn.Module`.
#     dataset_dict : LoadedDatasetDict
#         The data loader providing the data to be used for the assessment.
#     loss_function : ~collections.abc.Callable
#         Examples are provided in `Loss Functions <hhttps://pytorch.org/docs/stable/nn.html#loss-functions>`_ but \
#         user-defined functions of *(1)* the output of the network and (2) the target can be used too
#     attack_type : str
#         The type of the adversarial attack to be used. The possible values are: ``'FGSM'``, ``'OnePixel'``, \
#         ``'GN'``, and ``'DeepFool'``
#     attack_params : dict
#         The dictionary of parameters for the adversarial attack. The parameters depend on the attack type:
#         - 'FGSM': {'epsilon': float}
#         - 'OnePixel': {'pixel_count': int, 'max_iter': int, 'popsize': int}
#         - 'GN': {'mean': float, 'std': float}
#         - 'DeepFool': {'max_iter': int, 'overshoot': float}
#         When the fields are not a scalar but a list of scalars then all the combinations of the parameters \
#         are considered, i.e. the cartesian product of the parameters is used.
#     baseline_classifier: torch.nn.Module, optional
#         The classifier neural network (e.g. Standard Model) evaluated as a baseline in the assessment of the \
#         adversarial attacks.
#         Default: ``None``
#     classifier_for_aa_generation: torch.nn.Module, optional
#         The classifier neural network used to generate the adversarial images for the assessment of the \
#         adversarial attacks. If not provided, the `examined_classifier` is used to generate the adversarial images \
#         for the assessment of `examined_classifier` and, if provided, `baseline_classifier` is used to generate the \
#         adversarial images for the assessment of `baseline_classifier`.
#         Default:
#     validation_on_test_subset : bool, optional
#         The flag indicating whether the validation should be performed on the test subset of the data loader. \
#         If `True`, the test subset is used; if `False`, the validation ('val')subset is used. Default: ``True``
#     subbatch_size : int, optional
#         The size of the sub-batches to be used for the gradient calculation. If it is not provided the same \
#         batch size used in the dataset `dataset_dict` is used. Default: ``None``
#     mlflow_run_id : str, optional
#         If provided, it is the ID of the MLFlow run, already started, to which the current experiment will be logged. \
#         Said run must exist and be active. If ``None`` no experiment where the run is to be stored. \
#         If ``None`` the new run is a top-level run and will not refer to any previous run. \
#         Default: ``None``
#     run_name : str, optional
#         Name of the run, to be used in the local logging functions if necessary, in case `mlflow_run_id` is not \
#         provided and therefore the run name cannot be obtained from it; if a valid MLFlow \
#         run is provided, and `run_name` is also provided, an exception is raised. \
#         In the case local logging is requested and no run name is provided, neither through `mlflow_run_id` nor \
#         in the form of a directly provided `run_name`, a name for the experiment related to the host machine, \
#         start time of the training, and certain attributes of the classifier, will be automatically generated.
#         Default: ``None``
#     verbose : str, optional
#         Value among ``'high'``, ``'medium'``, ``'low'``, ``'none'``, indicating the mode of progress printing. \
#         Default: ``'medium'``
#
#     Returns
#     -------
#     float
#         The accuracy of the classifier `examined_classifier` for the adversarial attack `attack_type` and its parameters \
#         `attack_params` using the data provided by the `data_loader`.
#     """
#
#     if verbose == 'high':
#         print(f"Training of the classifier network:\n")
#         print(str(examined_classifier))
#         attributes_to_display = examined_classifier.fields_to_log
#         print(f"\n\tFields to log:\n")
#         for key in attributes_to_display:
#             print(f"\t\t{key}: {attributes_to_display[key]}")
#         pass
#     pass
#
#     ###########################
#     # Make the attack_params (which is a dictionary), if multiple, a cartesian product of combinations
#     ###########################
#
#     dict_parameter_names_per_attack_type = {
#         'fgsm': ['epsilon'],
#         'onepixel': ['pixel_count', 'max_iter', 'popsize'],
#         'gn': ['mean', 'std'],
#         'deepfool': ['max_iter', 'overshoot']
#     }
#
#     assert isinstance(attack_type, str), \
#         f"The attack_type must be a string! However, it is of type {type(attack_type)} and is not!"
#     assert attack_type.lower() in dict_parameter_names_per_attack_type, \
#         f"The attack_type must be one of {list(dict_parameter_names_per_attack_type.keys())} (case ignored)! " + \
#         f"However, it is '{attack_type}' and is not!"
#     list_parameter_names_for_attack_type = dict_parameter_names_per_attack_type[attack_type.lower()]
#
#     assert isinstance(attack_params, dict), \
#         f"The attack_params must be a dictionary! However, it is of type {type(attack_params)} and is not!"
#     assert all(key in list_parameter_names_for_attack_type for key in attack_params), \
#         f"The attack_params must contain only the keys {list_parameter_names_for_attack_type}! " + \
#         f"However, it contains the keys {list(attack_params.keys())} and is not!"
#     assert all(key in attack_params for key in list_parameter_names_for_attack_type), \
#         f"The attack_params must contain all the keys {list_parameter_names_for_attack_type}! " + \
#         f"However, it contains the keys {list(attack_params.keys())} and is not!"
#
#     # Check that the parameters are scalars and make them, also for the 1-element case, lists (of scalars)
#     for key in attack_params:
#         if isinstance(attack_params[key], (list, tuple)):
#             assert all(isinstance(item, (int, float)) for item in attack_params[key]), \
#                 f"The attack_params['{key}'] must be a list or tuple of numbers! " + \
#                 f"However, it is of type {type(attack_params[key])} and is not!"
#         elif isinstance(attack_params[key], (int, float)):
#             attack_params[key] = [attack_params[key]]
#         else:
#             raise ValueError(f"The attack_params['{key}'] must be a list or tuple of numbers! " +
#                              f"However, it is of type {type(attack_params[key])} and is not!")
#         pass
#     pass
#
#     # Make the cartesian product of the attack_params
#     attack_params_list = list(itertools.product(*attack_params.values()))
#     ###
#     # NONE: now, each "cartesian" combination of parameters, including keys, can be accessed as:
#     # for attack_params_list_i in attack_params_list:
#     #     attack_params_i = dict(zip(attack_params.keys(), attack_params_list_i))
#     #     print(attack_params_i)
#     #     ...
#     ###
#
#     # We also keep track of the indices corresponding to each attack_params_list_i, for later storage
#     indices_attack_params_list = list(itertools.product(*[range(len(attack_params[key])) for key in attack_params]))
#
#     ###########################
#     # Check the models
#     ###########################
#
#     assert isinstance(examined_classifier, nn.Module), \
#         f"The classifier must be a subclass of torch.nn.Module! However, " + \
#         f"it is of type {type(examined_classifier)} and is not!"
#
#     assert baseline_classifier is None or isinstance(baseline_classifier, nn.Module), \
#         f"The baseline_classifier must be a subclass of torch.nn.Module, or None! However, " + \
#         f"it is of type {type(baseline_classifier)}!"
#
#     assert classifier_for_aa_generation is None or isinstance(classifier_for_aa_generation, nn.Module), \
#         f"The baseline_classifier must be a subclass of torch.nn.Module, or None! However, " + \
#         f"it is of type {type(classifier_for_aa_generation)}!"
#
#     ###########################
#     # Move all the models to the device of the examined classifier
#     ###########################
#
#     device_examined_classifier = next(iter(examined_classifier.parameters())).data.device.type
#     if baseline_classifier is not None:
#         baseline_classifier.to(device_examined_classifier)
#     pass
#     if classifier_for_aa_generation is not None:
#         classifier_for_aa_generation.to(device_examined_classifier)
#     pass
#
#     ###########################
#     # Definition, or load, of the name defining the experiment/run
#     ###########################
#
#     if mlflow_run_id is not None:
#         if run_name is not None:
#             raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
#                             "only one of them can be provided at a time!")
#         else:
#             # Check if the run with the provided ID is the active run
#             if mlflow.active_run() is not None and mlflow.active_run().info.run_id == mlflow_run_id:
#                 # Great, the active run is the provided run
#                 pass
#             else:
#                 # Re-start the MLFlow run corresponding to the provided ID
#                 mlflow.start_run(run_id=mlflow_run_id)
#             pass
#             # And obtain the name of the run name
#             run_name = mlflow.active_run().info.run_name
#         pass
#     elif run_name is None:
#         raise Exception("Both 'mlflow_run_id' and 'run_name' are provided: " +
#                         "only one of them can be provided at a time!")
#     pass
#
#     ###########################
#     # Subdataset to use
#     ###########################
#     # Field 'dataloader_val' or 'dataloader_test' depending on 'validation_on_test_subset'
#     subset_for_validation = 'dataloader_test' if validation_on_test_subset else 'dataloader_val'
#     subbatch_size = subbatch_size if subbatch_size is not None else dataset_dict[subset_for_validation].batch_size
#
#     ###########################
#     # The result will be hypercubes wherein each dimension corresponds to a parameter, and one hypercube per:
#     # - "examined_classifier" and (if provided) "baseline_classifier"
#     # - loss and acc
#     ###########################
#
#     # Classifier dictionary
#     classifier_dict = {'examined_classifier': examined_classifier} if baseline_classifier is None else \
#         {'examined_classifier': examined_classifier, 'baseline_classifier': baseline_classifier}
#
#     # Final structures for the results with AA
#     shape_attack_params = tuple([len(attack_params[key]) for key in attack_params])
#     val_acc_aa_dict = {'examined_classifier': torch.empty(shape_attack_params)} if baseline_classifier is None else \
#         {'examined_classifier': torch.empty(shape_attack_params),
#          'baseline_classifier': torch.empty(shape_attack_params)}
#     loss_acc_aa_dict = copy.deepcopy(val_acc_aa_dict)
#     # Final structures for the results without AA
#     val_acc_no_aa_dict = {'examined_classifier': 0.0} if baseline_classifier is None else \
#         {'examined_classifier': 0.0, 'baseline_classifier': 0.0}
#     loss_acc_no_aa_dict = copy.deepcopy(val_acc_no_aa_dict)
#
#     ###########################
#
#     # The "None" case has been included for the case of NO adversarial attack (i.e. original images)
#     for indices_attack_params_list, attack_params_list_i in zip([[None]+indices_attack_params_list], [[None]+attack_params_list]):
#         #
#         #
#         DO SOMETHING WITH indices_attack_params_list !!!
#         #
#         attack_params_i = dict(zip(attack_params.keys(), attack_params_list_i)) \
#             if attack_params_list_i is not None else None
#
#         print(f"Evaluation for: ", end='')
#         if attack_params_list_i is None:
#             print("no adversarial attack")
#         else:
#             print(f"adversarial attack {attack_type} with parameters: {attack_params_list_i}")
#         pass
#         print("-----")
#
#         # Auxiliary variables for accumulated values from each batch
#         cum_loss_acc_not_averaged_dict = {'examined_classifier': 0.0} if baseline_classifier is None else \
#             {'examined_classifier': 0.0, 'baseline_classifier': 0.0}
#         cum_correct_images_not_averaged_dict = copy.deepcopy(cum_loss_acc_not_averaged_dict)
#
#         total_num_images_images = 0
#
#         for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
#             #
#             # Accumulate the total number of images so far
#             total_num_images_images += output_images_batch.size(0)
#             #
#             # We perform everything in the device of the network:
#             images_batch = images_batch.to(device_examined_classifier)
#             labels_batch = labels_batch.to(device_examined_classifier)
#             #
#             #########
#             # With the gradient we can calculate the adversarial images of the batch
#             #########
#             # Remember:
#             # - if "classifier_for_aa_generation" is provided, the attack is generated using it for both \
#             #   "examined_classifier" and "baseline_classifier";
#             # - otherwise, each one has a different attack generator
#             #########
#
#             perturbed_images_dict = {'examined_classifier': None} if baseline_classifier is None else \
#                 {'examined_classifier': None, 'baseline_classifier': None}
#
#             for key in perturbed_images_dict:
#                 ###
#                 # Set the classifier to use for AA generation depending on the conditions (or even copying the
#                 # perturbed images from the other classifier if it is the same)
#                 ###
#                 classifier_for_aa_generation_i = None
#                 if attack_params_i is None:
#                     # If there are no attack parameters, we do not generate adversarial images
#                     perturbed_images_dict[key] = images_batch
#                     continue
#                 elif key == 'baseline_classifier':
#                     if baseline_classifier is None:
#                         continue
#                     elif classifier_for_aa_generation is not None: # Same as generated before for 'examined_classifier'
#                         perturbed_images_dict['baseline_classifier'] = perturbed_images_dict['examined_classifier']
#                         continue
#                     else:
#                         classifier_for_aa_generation_i = baseline_classifier
#                 else:  # key == 'examined_classifier'
#                     if classifier_for_aa_generation is not None:
#                         classifier_for_aa_generation_i = classifier_for_aa_generation
#                     else:
#                         classifier_for_aa_generation_i = examined_classifier
#                     pass
#                     #
#                     if attack_type.lower() == 'fgsm':
#                         # For "examined_classifier"
#                         attack_object = torchattacks.FGSM(classifier_for_aa_generation_i, attack_params_i['epsilon'])
#                         perturbed_images_dict[key] = attack_object(images_batch, labels_batch)
#                     elif attack_type.lower() == 'onepixel':
#                         attack_object = torchattacks.OnePixel(classifier_for_aa_generation_i,
#                                                               attack_params['pixel_count'],
#                                                               attack_params['max_iter'], attack_params['popsize'])
#                         perturbed_images_dict[key] = attack_object(images_batch, labels_batch)
#                     elif attack_type.lower() == 'deepfool':
#                         attack_object = torchattacks.DeepFool(classifier_for_aa_generation_i, attack_params['steps'],
#                                                               attack_params['overshoot'])
#                         perturbed_images_dict[key] = attack_object(images_batch, labels_batch)
#                     elif attack_type.lower() == 'gn':
#                         perturbed_images_dict[key] = _generate_gn_attack_images(
#                             images_batch, attack_params, normalization=dataset_dict['normalized_dataset']
#                         )
#                     elif attack_type.lower() == 'pgd':
#                         raise Exception("Only a placeholder for now! Not implemented yet!")
#                     else:
#                         raise ValueError(f"The attack type '{attack_type}' is not recognized!")
#                     pass
#                     #
#                 pass
#             pass
#
#             #########
#             # And we evaluate the output of the classifier for the adversarial images
#             #########
#
#             loss_images_batch_dict = {}
#             num_correct_images_not_averaged_dict = {}
#
#             for key in classifier_dict:
#                 #
#                 # Predict the output of the classifier
#                 classifier_dict[key].eval()
#                 with (torch.no_grad()):
#                     output_images_batch = classifier_dict[key](images_batch)
#                     # NOTE: no info about convergence is saved
#                 pass
#                 #
#                 # Calculate losses and number of correct classifications
#                 loss_images_batch_dict[key] = loss_function(output_images_batch, labels_batch).item()
#                 output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
#                 num_correct_images_not_averaged_dict[key] = (
#                         output_labels_batch.to(device_examined_classifier) ==
#                         labels_batch.to(device_examined_classifier)
#                 ).sum().item()
#                 #
#                 # Accumulate the values for out-of-the-calculation, including un-averaging the loss
#                 cum_loss_acc_not_averaged_dict[key] += loss_images_batch_dict[key] * output_images_batch.size(0)
#                 cum_correct_images_not_averaged_dict[key] += num_correct_images_not_averaged_dict[key]
#                 #
#             pass    # END OF: for key in classifier_dict ...
#             #
#             # Batch summary
#             line_to_print = f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]"
#             for key in classifier_dict:
#                 line_to_print += (
#                         f" --- {key} " +
#                         f"loss={loss_images_batch_dict[key]:6.3f} " +
#                         f"acc={100.0 * num_correct_images_not_averaged_dict[key] / output_images_batch.size(0):5.2f} %"
#                 )
#             pass
#             if verbose in ['medium', 'high']:
#                 end_batch = "\n" if verbose == 'high' else "\r"
#                 print(line_to_print, end=end_batch)
#             pass
#             #
#         pass    # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation])
#
#         # Now calculate the final values for the classifiers
#         if attack_params_i is None:     # This means: no attack
#             for key in val_acc_no_aa_dict:
#                 val_acc_no_aa_dict[key] = cum_correct_images_not_averaged_dict[key] / total_num_images_images
#                 loss_acc_no_aa_dict[key] = cum_loss_acc_not_averaged_dict[key] / total_num_images_images
#             pass
#         else:   # This means: attack
#             for key in val_acc_aa_dict:
#                 val_acc_aa_dict[key][attack_params_list_i] = \
#                     cum_correct_images_not_averaged_dict[key] / total_num_images_images
#                 loss_acc_aa_dict[key][attack_params_list_i] = \
#                     cum_loss_acc_not_averaged_dict[key] / total_num_images_images
#             pass
#         pass
#
#         indices_attack_params_list
#
#         #
#         # cum_loss_acc_not_averaged_dict[key] += loss_images_batch.item() * output_images_batch.size(0)
#         # cum_correct_images_not_averaged_dict[key] += right_classifications_batch
#         #
#         #
#         #         if verbose in ['medium', 'high']:
#         #             end_batch = "\n" if verbose == 'high' else "\r"
#         #             print((f"   " +
#         #                    f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
#         #                    f"loss={loss_images_batch:.3f}   " +
#         #                    f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
#         #                    f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}   "+
#         #                    f"perturbed_loss_sm={loss_perturbed_images_batch_sm:.3f}   " +
#         #                    f"perturbed_acc_sm={100*right_classifications_perturbed_batch_sm/output_images_batch.size(0):.3f} %" +
#         #                    f"perturbed_conv_err_sm={fp_perturbed_conv_err_sm}"),
#         #                   end=end_batch)
#         #         pass
#         #             #
#         #     pass
#     pass    # END OF: for attack_params_list_i in [None, attack_params_list]: ...
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     for attack_params_list_i in attack_params_list:
#
#         attack_params_i = dict(zip(attack_params.keys(), attack_params_list_i))
#         print(attack_params_i)
#
#         for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[subset_for_validation]):
#             #
#             # We perform everything in the device of the network:
#             images_batch = images_batch.to(device_examined_classifier)
#             labels_batch = labels_batch.to(device_examined_classifier)
#             #
#             # With the gradient we can calculate the adversarial images of the batch
#             if attack_type.lower() == 'fgsm':
#                 attack = torchattacks.FGSM(classifier_for_gradient_calculation, attack_params['epsilon'])
#                 perturbed_images_batch_sm = attack(images_batch, labels_batch)
#             elif attack_type.lower() == 'onepixel':
#                 attack = torchattacks.OnePixel(classifier_for_gradient_calculation, attack_params['pixel_count'], attack_params['max_iter'],
#                                                attack_params['popsize'])
#                 perturbed_images_batch_sm = attack(images_batch, labels_batch)
#             elif attack_type.lower() == 'deepfool':
#                 attack = torchattacks.DeepFool(classifier_for_gradient_calculation, attack_params['steps'], attack_params['overshoot'])
#                 perturbed_images_batch_sm = attack(images_batch, labels_batch)
#             elif attack_type.lower() == 'gn':
#                 perturbed_images_batch_sm = _generate_gn_attack_images(
#                     images_batch, attack_params, normalization=dataset_dict['normalized_dataset']
#                 )
#             elif attack_type.lower() == 'pgd':
#                 raise Exception("Only a placeholder for now! Not implemented yet!")
#             else:
#                 raise ValueError(f"The attack type '{attack_type}' is not recognized!")
#             pass
#             #
#             # And we evaluate the output of the classifier for the adversarial images
#             #
#             examined_classifier.eval()
#             classifier_for_gradient_calculation.eval()
#             #
#             examined_classifier.zero_grad()
#
#             with (torch.no_grad()):
#                 output_images_batch = examined_classifier(images_batch)
#                 # Check if the layer has a forward convergence info
#                 if hasattr(examined_classifier._nn[1][0][0], 'get_last_forward_convergence_info'):
#                     # If it does, we can use it
#                     fp_conv_err += int(sum(examined_classifier._nn[1][0][0].get_last_forward_convergence_info()[0].detach().cpu().numpy()==True))
#                 elif hasattr(examined_classifier._nn[1][0][-1], 'get_last_forward_convergence_info'):
#                     # If it does, we can use it
#                     fp_conv_err += int(sum(examined_classifier._nn[1][0][-1].get_last_forward_convergence_info()[0].detach().cpu().numpy()==True))
#                 pass
#                 #fp_conv_err += int(sum(examined_classifier._nn[0][0].get_last_forward_convergence_info()[0].detach().cpu().numpy()==True))
#
#                 #
#                 loss_images_batch = loss_function(output_images_batch, labels_batch)
#                 #
#                 # Accumulate the values for out-of-the-calculation, including un-averaging the loss
#                 total_num_images_images += output_images_batch.size(0)
#                 cum_loss_not_averaged += loss_images_batch.item() * output_images_batch.size(0)
#
#                 output_labels_batch = torch.max(output_images_batch, dim=-1)[1]
#                 right_classifications_batch = (output_labels_batch.to(device_examined_classifier) == labels_batch.to(device_examined_classifier)).sum().item()
#
#                 cum_right_classifications_not_averaged += right_classifications_batch
#
#                 output_perturbed_images_batch_sm = examined_classifier(perturbed_images_batch_sm)
#                 # Check if the layer has a forward convergence info
#                 if hasattr(examined_classifier._nn[1][0][0], 'get_last_forward_convergence_info'):
#                     # If it does, we can use it
#                     fp_perturbed_conv_err_sm += int(sum(examined_classifier._nn[1][0][0].get_last_forward_convergence_info()[
#                                                         0].detach().cpu().numpy() == True))
#                 elif hasattr(examined_classifier._nn[1][0][-1], 'get_last_forward_convergence_info'):
#                     # If it does, we can use it
#                     fp_perturbed_conv_err_sm += int(sum(examined_classifier._nn[1][0][-1].get_last_forward_convergence_info()[
#                                                         0].detach().cpu().numpy() == True))
#
#                 loss_perturbed_images_batch_sm = loss_function(output_perturbed_images_batch_sm, labels_batch)
#                 cum_perturbed_loss_not_averaged_sm += loss_perturbed_images_batch_sm.item() * output_images_batch.size(0)
#                 output_perturbed_labels_batch_sm = torch.max(output_perturbed_images_batch_sm, dim=-1)[1]
#                 right_classifications_perturbed_batch_sm = (
#                         output_perturbed_labels_batch_sm.to(device_examined_classifier) == labels_batch.to(device_examined_classifier)
#                 ).sum().item()
#                 cum_perturbed_right_classifications_not_averaged_sm += right_classifications_perturbed_batch_sm
#                 #
#                 #
#                 # Log the specific images (correct in perturbed, incorrect in original)
#                 incorrect_indices = (output_labels_batch != labels_batch).nonzero(as_tuple=True)[0]
#                 correct_perturbed_indices = (output_perturbed_labels_batch_sm == labels_batch).nonzero(as_tuple=True)[0]
#                 # Convert tensors to numpy arrays
#                 incorrect_indices_np = incorrect_indices.cpu().numpy()
#                 correct_perturbed_indices_np = correct_perturbed_indices.cpu().numpy()
#
#                 # Find the intersection using numpy
#                 specific_indices_np = np.intersect1d(incorrect_indices_np, correct_perturbed_indices_np)
#
#                 # Convert the result back to a tensor
#                 specific_indices = torch.tensor(specific_indices_np, device=incorrect_indices.device)
#
#
#
#                 # if specific_indices.numel() > 0:
#                 #     log_side_by_side_images(images_batch, perturbed_images_batch_sm, specific_indices, num_batch + 1,
#                 #                             "incorrect_to_correct", perturbed_images_sm=None,
#                 #                             labels=labels_batch,
#                 #                             predicted_labels=output_labels_batch,
#                 #                             predicted_labels_perturbed=output_perturbed_labels_batch_sm
#                 #                             )
#                 # pass
#
#                 #
#
#                 if verbose in ['medium', 'high']:
#                     end_batch = "\n" if verbose == 'high' else "\r"
#                     print((f"   " +
#                            f"Val batch [{(num_batch + 1):5d}/{len(dataset_dict[subset_for_validation]):5d}]   " +
#                            f"loss={loss_images_batch:.3f}   " +
#                            f"acc={100.0 * right_classifications_batch / output_images_batch.size(0):.3f} %   " +
#                            f"   conv_err={fp_conv_err}   perturbed_conv_err={fp_perturbed_conv_err}   "+
#                            f"perturbed_loss_sm={loss_perturbed_images_batch_sm:.3f}   " +
#                            f"perturbed_acc_sm={100*right_classifications_perturbed_batch_sm/output_images_batch.size(0):.3f} %" +
#                            f"perturbed_conv_err_sm={fp_perturbed_conv_err_sm}"),
#                           end=end_batch)
#                 pass
#                     #
#             pass
#             #
#         pass  # END OF: for num_batch, (images_batch, labels_batch) in enumerate(dataset_dict[dataloader_subset])
#
#         # Calculate the final values for the classifier
#         summary_examined_classifier = {
#             'total_num_images_images': total_num_images_images,
#             'loss': cum_loss_not_averaged / total_num_images_images,
#             'acc': cum_right_classifications_not_averaged / total_num_images_images,
#             'fp_conv_err': fp_conv_err/total_num_images_images,
#         }
#         summary_examined_classifier.update({
#             'perturbed_loss_sm': cum_perturbed_loss_not_averaged_sm / total_num_images_images,
#             'perturbed_acc_sm': cum_perturbed_right_classifications_not_averaged_sm / total_num_images_images,
#             'fp_perturbed_conv_err_sm': fp_perturbed_conv_err_sm / total_num_images_images
#         })
#         if mlflow_run_id is not None:
#             mlflow.log_metrics(summary_examined_classifier, run_id=mlflow_run_id)
#             mlflow.log_params({'attack_type': attack_type, 'attack_params': attack_params}, run_id=mlflow_run_id)
#         pass
#         #
#         if verbose in ['high', 'medium']:
#             print(f"Summary:{' '*40}")
#             print(f"   {'attack':<25} {attack_type}, {str(attack_params)}")
#             for key in summary_examined_classifier:
#                 end_particle = " % \n" if 'acc' in key else "\n"
#                 factor = 100.0 if 'acc' in key else 1.0
#                 print(f"   {key:<25}{factor*summary_examined_classifier[key]:.3f}", end=end_particle)
#             pass
#         pass
#         #
#         # Log the info of the classifier (for easier retrieval)
#         if mlflow_run_id is not None:
#             dict_params_to_log = examined_classifier.fields_to_log
#             for key in dict_params_to_log:
#                 mlflow.log_params({key: str(dict_params_to_log[key])}, run_id=mlflow_run_id)
#             pass
#             for key in dict_params_to_log:
#                 if isinstance(dict_params_to_log[key], (int, float)):
#                     mlflow.log_metric(key, dict_params_to_log[key], run_id=mlflow_run_id)
#                 elif isinstance(dict_params_to_log[key], (list, tuple)) and \
#                         all(elem in (int, float) for elem in dict_params_to_log[key]):
#                     mlflow.log_metric(key, tuple(dict_params_to_log[key]), run_id=mlflow_run_id)
#                 pass
#
#             pass
#         pass
#         #
#     pass # END OF: for attack_params_list_i in attack_params_list ...
#
#     # Return the summary or list of summaries of the classifiers
#     return summary_examined_classifier
