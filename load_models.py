from __future__ import absolute_import, division, print_function
import os
import sys
from copy import deepcopy
import numpy as np
import tensorflow as tf
from io import BytesIO
import tarfile
import tempfile
from PIL import Image

ROOT_DIR = os.path.abspath("./OpticalFlow/tpflow/tfoptflow-master/tfoptflow/")

# OPTICAL FLOW MODEL #

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



# SEMANTIC SEGMENTATION MODEL #

model_path = "./Segmentation/deeplabv3_cityscapes_train_2018_02_06.tar.gz"

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 1024
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    def __init__(self):
    #"""Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(model_path)
        for tar_info in tar_file.getmembers():
          if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
            file_handle = tar_file.extractfile(tar_info)
            graph_def = tf.GraphDef.FromString(file_handle.read())
            break

        tar_file.close()

        if graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

        #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            seg_map: Segmentation map of the image.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]
            })
        seg_map = batch_seg_map[0]
        return seg_map

ROOT_DIR_MASK_RCNN = os.path.abspath("./HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn/Mask_RCNN")



# MASK-RCNN MODEL #

sys.path.append(ROOT_DIR_MASK_RCNN)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR_MASK_RCNN, "samples/coco/"))  # To find local version
import coco


# Local path to trained weights file of mask-rcnn
COCO_MODEL_PATH = os.path.join(ROOT_DIR_MASK_RCNN, "mask_rcnn_coco.h5")

# Directory to save logs
MODEL_DIR = os.path.join(ROOT_DIR_MASK_RCNN, "logs")

# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
    #utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


def get_mask_rcnn():
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model



# HUMAN POSE MODEL #

ROOT_DIR_HUMAN_POSE = os.path.abspath("./PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/")

# Import OpenPose
sys.path.append(ROOT_DIR_HUMAN_POSE)  # To find local version of the library

from model.cmu_model import get_testing_model
from config_reader import config_reader

keras_weights_file = "./PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/model/keras/model.h5"


def get_open_pose():

    #load model
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    return model, params, model_params
