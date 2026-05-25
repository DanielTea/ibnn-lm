# Description of the current formats of the configuration files
(Current version: 2026-05-24)

## General description of the organization of the configuration files:

The creation and training of classifiers performed by the script
(``/src/``)``main_train_classifiers.py`` (e.g. through command line) is based on configuration files
written in TOML format. In particular, the file runs using a configuration file as an argument,
but **requires two configuration files to run**, referred to in the following (and in the code) as
**BASE** and **GROUP** files. Both files are organized using the same hierarchical structure,
but have different purposes and have some differences:

  - The script takes a **GROUP** file as an argument; the location of the **BASE** file is only
    referred to, internally, by the **GROUP** file.  
  - The **GROUP** file uses the parameters defined in the **BASE** file it refers to
    but substitute the parameters contained by itself: the content in **GROUP** is meant to override
    those "default" parameters set by the **BASE**. And those parameters overriding
    those in the **BASE** are set in the form of vectors, indicating that all the hyperparameters indicated by
    the vector must originate corresponding experiments.
  - If multiple parameters listing multiple alternative appear overridden in **GROUP**, then each element in the
    the Cartesian product of the requested hyperparameters will correspond to one experiment.
  - The **GROUP** file contains certain fields which are not hyperparameters and which serve to point at its
    intended **BASE** file and to name the group of experiments to perform, and said **fields** are exclusive to the
    **GROUP** file. Said fields are:
    - `base_experiment_specification_file`, containing the absolute or relative path to the **BASE** file;
    - `experiment_name` and `purpose`, containing respectively the name to give to the group of experiments
      and a short description of the purpose of the group; and
    - `mlflow_logging` and `local_log_folder`, regarding the MLFlow logging of the experiments.

These configuration files contain the following blocks of parameters:
  - Only in the group file: experiment fields `experiment_name`, `purpose`, and `base_experiment_specification_file` and
    the logging fields `mlflow_logging` and `local_log_folder` in the group file ([link](#outmost-fields))
  - `[dataset]` (and children) ([link](#descriptor-dataset))
  - `[classifier]` (and children) ([link](#descriptor-classifier))
  - `[training]` (and children) ([link](#descriptor-training))
   
Regarding how the fields in the common block `[classifier]` is treated by the Python scripts processing them:
  - The functions processing the **training** configuration **treat the fields in the block `[classifier]` of
    the configuration files as compulsory**,
    that is, raise exceptions if any (important) field is missing. (They behave in such manner because there is no
    source to pick the missing information from.)
  - The functions processing the **retraining** and the **AA** configuration files
    **treat all the fields in the block `[classifier]` of
    the configuration files but `[classifier].net` as optional** (**the omission of the latter currently raises
    an exception)**; since each AA experiment is built on an already trained classifier,
    the missing parameters regarding `[classifier]` are simply left as they are. (**In future versions also the
    `[classifier].net` field will be optional**).

## Specific parameters and blocks: description

### Outmost fields in the group configuration files <a name="outmost-fields"></a>

```
experiment_name = "svhn_alexnet_no_normalization"
purpose = "Training of different alexnet classifiers on SVHN dataset."
base_experiment_specification_file = "./2025-06-19_svhn_alexnet_base.toml"

mlflow_logging = true
local_log_folder = "../runs/base_SINGLE_LAYER" # string or false (false means no saved logging)
```


### `[dataset]` (and children) with examples <a name="descriptor-dataset"></a>

```
[dataset]
  name = "svhn" # "mnist", "fashion-mnist", "svhn", "cifar10", "cifar100", "food101", "places365_small"
  colorspace = "rgb" # "gray", "rgb", "cieluv"
  force_im_size = false # list e.g. [128, 128] or false (which means leave the original size)
  # Proportion of the original train set of the dataset devoted to train ("train_proportion") vs to validation
  # They do not need to sum 1, data can be left out
  train_proportion = 1.0
  mislabeled_proportion = 0.0
  val_proportion = 0.0
  batch_size = 100
  num_workers = 4
  generator_seed = false # int or false
  root_folder = "../data/"
  ```


### `[classifier]` (and children) with examples <a name="descriptor-classifier"></a>

```
[classifier]

  net = 'multi_layer' # 'multi_layer', single_layer', 'double_layer', 'vggx', 'alexnet', 'efficientnetv2s'
  
  conv_like_type = "sm" # "sm", "inrfv1", "inrfv2", "inrfv3", "ibnn_lite", "ibnn_internal", "ibnn"
  conv_like_type_position = "everywhere" # 'everywhere', 'first', 'last'
  
  prenormalization = false # bool

  [classifier.architecture_specific]

    # CLASSIFIER-SPECIFIC PARAMETERS!!!
    ###############################
    # FOR 'multi_layer':
    ###############################
    [classifier.architecture_sptrecific.multi_layer]
      num_conv_like_layers = 2 # list of 1s or false (which means same number of channels as the input images)
      channels_per_conv_layer = [3, 3] # list[int/float]
      m_kernel_size_per_conv_layer = false # false of list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
      phi_activation_per_conv_layer = false # false or list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
      batch_normalization_per_conv_layer = false # false or list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
      maxpool_reduction_per_conv_block = false # false or list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
    ###############################
    # FOR 'vggx':
    ###############################
    [classifier.architecture_specific.vggx]
      conv_block_specification = [3, 1] # int or false (which means same number of channels as the input images)
      base_channels = 4 # int
      m_kernel_size_per_conv_layer = false # list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
    ###############################
    # FOR 'alexnet':
    ###############################
    [classifier.architecture_specific.alexnet]
      conv_block_specification = [1, 1, 3] # int or false (which means same number of channels as the input images)
      channels_per_conv_layer = [64, 192, 384, 256, 256] # list[int/float]
      m_kernel_size_per_conv_layer = [11, 5, 3, 3, 3] # m_kernel_size_per_conv_layer = false # list of ints/floats fitting 'conv_block_specification' or False (which means common kernel size for all layers)
    ###############################
    # FOR 'efficientnetv2s':
    ###############################
    [classifier.architecture_specific.efficientnetv2s]
      m_kernel_size_per_conv_layer = [3, 1] # list of 2 ints/floats or False (which means common kernel size for all layers)
      num_hidden_channels_2 = 1280 # int
  
  [classifier.fully_connected]
  
    fc_num_layers = 1 # int
    fc_num_units_intermediate_layers = -1 # int
    fc_batch_normalization = true # bool
    fc_dropout = 0.0 # float between 0 and 1
    penciled_decision = false # bool
    softmax_output = false # bool
  
  [classifier.conv_like_layer]

    [classifier.conv_like_layer.traditional]

      overall_batch_normalization_conv_layers = true # bool
      overall_maxpool_reduction_conv_blocks = 1 # int; 1 means no maxpooling

      overall_phi_activation_conv_layers = false # "identity", "relu", "leaky_relu", "elu", "selu", "celu", "prelu", "rrelu", "tanh", "hardtanh", "softsign", "sigmoid", "silu"
      overall_m_kernel_size_conv_layers = 0.15 # int, common kernel size for all layers (float=percentage or int=pixels) unless architecture-specifically defined (m_kernel_size_per_conv_layer)
       
      m_independent_channels = false # int
      m_padding = "same" # "same", "valid"
      m_padding_mode = "zeros" # "zeros", "reflect", "replicate", "circular"
      m_initialization = "zeros"  # "delta", "zeros"
      m_trainable = true

      b_type = "scalar_per_channel" # "scalar_per_channel", "scalar"
      initial_b = 0.0 # float/int or list of float/int
      b_trainable = true

    [classifier.conv_like_layer.nonlinear_bias]

      sigma_activation = "tanh" # "identity", "relu", "leaky_relu", "elu", "selu", "celu", "prelu", "rrelu", "tanh", "hardtanh", "softsign", "sigmoid", "silu"

      sigma_x_compress = 3.0 # float  (neutral is 1.0)
      sigma_y_stretch = 1.0 # float (neutral is 1.0)
      sigma_x_offset = 0.0 # float (neutral is 0.0)
      sigma_y_offset = 0.0 # float (neutral is 0.0)
      sigma_x_compress_trainable = false
      sigma_y_stretch_trainable = false
      sigma_x_offset_trainable = false
      sigma_y_offset_trainable = false

      lambda_type = "scalar" # "scalar_per_channel", "scalar"
      initial_lambda = -0.5 # float/int or list of float/int
      lambda_trainable = false

      w_kernel_size = 0 # Percentage (float) of the im size or absolute int size (or tuple); 0 (or empty tuple () or list []) means full-im uniform kernel.
      w_independent_channels = true # bool: "true" that all w channels are considered independent of "false" together
      w_padding_mode = "zeros" # "zeros", "reflect", "replicate", "circular"
      # FOR "w_initialization":
      # "initialization_type" options: "zeros", "ones", "delta', 'gaussian', 'difference_of_gaussians', and 'gabor'
      # "normalization" options: None, "individual", "group", "full"
      # Dictionary possible. Examples:
      #       {"initialization_type"="ones", "normalization"="group"}
      #        {"initialization_type"="gaussian", "normalization"="group", "rel_sigma"=0.1},
      #        {"initialization_type"="gaussian", "normalization"="group", "sigma"=2.5},
      #        {"initialization_type"="ones", "normalization"="full"},
      #        {"initialization_type"="ones", "normalization"="group"},
      w_initialization = {"initialization_type"="ones", "normalization"="full"}
      w_trainable = false

      [classifier.conv_like_layer.nonlinear_bias.cross_conv_computation]

        calculation_mode = "interpolated" # "interpolated", "n4". Default: "interpolated"
        num_sampling_points = 11 # int
        range_std_sigma = [-4.0, 4.0] # list of 2 floats: start and end
        memory_saving_version = true

    [classifier.conv_like_layer.fixed_point]

      batched_fixed_point = true
      
      f_tau = 0.1                       # float or false (which means leave the default value)
      f_solver = "pgd"                  # "fixed_point_iter", "anderson", "broyden", "fpgd" and "pgd"
      f_max_iter = false                # int of false, e.g. 50
      f_tol = false                     # float or false, e.g. 1e-5
      
      b_solver = "broyden"              # "fixed_point_iter", "anderson", "broyden"
      b_max_iter = false                # int of false, e.g. 40
      b_tol = false                     # float or false, e.g. 1e-6
      
      abs_error_threshold = false       # float or false, e.g. 1e-5,
      
```


### `[training]` (and children) with examples <a name="descriptor-training"></a>

```
[training]

  maximum_epochs = 25 # int
  loss_function = 'cross_entropy' # 'cross_entropy', 'mse_loss', 'nll_loss', 'l1_loss', 'smooth_l1_loss', 'kl_div'
  validations_per_epoch = 3 # int
  validation_on_test_subset = true  # false means on the 'dataset_val'
  early_stop_epochs = false # int or false
  epochs_sm_based_warmup = false # int or false
  min_acc_threshold = false # 0<float<1.0 or false
  max_number_of_retries = false # 0<int or false

  [training.optimizer]
    type = "sgd" # String. For now only "adam", "sgd", and "asgd" is considered
    initial_lr = 0.001 # float or false (false if left default)
    arguments = {'momentum' = 0.9} # dict or false (false if left default)

  [training.scheduler]
  # String or false (false means no scheduler): options "ReduceLROnPlateau" and "ExponentialLR"
  # Below we provide examples for both (non-false) schedulers
    type = "ExponentialLR" # "warmup"/"ConstantLR", "ReduceLROnPlateau", "ExponentialLR" or false
    arguments = false # See examples below
    # type = "ReduceLROnPlateau"
    # arguments = {mode = "min", factor = 0.1, patience = 2, threshold = 0.05, threshold_mode = "rel"}
    # type = "ExponentialLR"
    # arguments = {gamma = 0.9}
    # type = "linear_warmup" # Or "LinearLR"
    # arguments = {factor = 0.1, total_iters = 5} # Total iters: num. of epochs until end of the warm up
    # type = "warmup" # Or "ConstantLR"
    # arguments = {factor = 0.1, total_iters = 5} # Total iters: num. of epochs where the factor is applied to the "basic" LR
    # type = "CyclicLR"
    # arguments = {low_lr = 0.01, num_iters_cycle=10} or {low_lr = 0.01, num_cycles=10}
 
  [training.adversarial]
    type = "pgd" # "fgsm", "pgd" (or false)
    arguments = {eps = 0.010, alpha = 0.0010, steps = 10} # dict with parameters for the adversarial training
    proportion = 1.0 # float between 0 and 1
```
