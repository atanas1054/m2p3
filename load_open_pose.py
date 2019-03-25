import os
import sys


ROOT_DIR = os.path.abspath("./PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/")

# Import OpenPose
sys.path.append(ROOT_DIR)  # To find local version of the library

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