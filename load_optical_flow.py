from __future__ import absolute_import, division, print_function
import os
import sys
from copy import deepcopy


ROOT_DIR = os.path.abspath("./OpticalFlow/tpflow/tfoptflow-master/tfoptflow/")

# Import PWC-Net optical flow
sys.path.append(ROOT_DIR)  # To find local version of the library
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS


def load_optical_flow_model():
    # Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
    gpu_devices = ['/device:GPU:0']
    controller = '/device:GPU:0'

    # Set the path to the trained model
    ckpt_path = ROOT_DIR + '/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller

    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    nn_opts['adapt_info'] = (1, 436, 1024, 2)

    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)

    #Return model
    return nn