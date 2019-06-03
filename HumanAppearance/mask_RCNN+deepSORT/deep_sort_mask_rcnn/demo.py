#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import math
from PIL import Image
from copy import deepcopy
from skimage.io import imread
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from sklearn.cluster import KMeans

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda, RepeatVector, Dropout, Activation, Flatten
from keras.layers.merge import dot, add, multiply, concatenate
from keras.engine import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K
from tensorflow import convert_to_tensor

from tensorflow import Graph, Session

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

OPTICAL_FLOW_DIR = os.path.abspath("../../../OpticalFlow/tpflow/tfoptflow-master/tfoptflow/")
sys.path.append(OPTICAL_FLOW_DIR)

from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows
from optflow import flow_to_img

pose_dir = os.path.abspath("../../../PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/")

# Import OpenPose
sys.path.append(pose_dir )  # To find local version of the library

from config_reader import config_reader
from model.cmu_model import get_testing_model
import util
from utils_ import *

#POSE ESTIMATION

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

observed_frame_num = 8
predicting_frame_num = 12
batch_size = 256
train_samples = 10
test_samples = 100
latent_dim = 64
person_appearance_size = 14*14
person_pose_size = 17*2
scene_features = 64
optic_flow_features = 50
# 19 semantic classes of size 36x64
semantic_maps = (19, 36, 64)
# 4 * 30 people per frame
person_interaction = 120

#best of many reconstrunction loss
def bms_loss( y_true, y_pred ):

    y_true = K.reshape( y_true, (batch_size,train_samples,predicting_frame_num,4) );
    y_pred = K.reshape( y_pred, (batch_size,train_samples,predicting_frame_num,4) );
    rdiff = K.mean(K.square(y_pred - y_true),axis=(2,3));
    rdiff_min = K.min( rdiff, axis = 1);
    return K.mean(rdiff_min)

#kl_loss
def kl_activity_reg( args ):
    z_mean = args[:,: latent_dim]
    z_log_var = args[:,latent_dim:]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)

def sample_z( args ):
    z_mean = args[:,:latent_dim]
    z_log_var = args[:,latent_dim:]
    epsilon = K.random_normal(shape=(K.shape(args)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def get_person_pose(oriImg, model, params, model_params):

    # if a person is too small dont calculate pose and continue
    if oriImg.shape[0] < 15 or oriImg.shape[1] < 15:
        joint_coords = np.zeros((17, 2))

    # else calculate pose
    else:
        multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        for m in range(len(multiplier)):
            scale = multiplier[m]

            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                              model_params['padValue'])

            input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                     (3, 0, 1, 2))  # required shape (1, width, height, channels)

            output_blobs = model.predict(input_img)

            # extract outputs, resize, and remove padding
            heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                                 interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # coordinates of 17 joints of the human body
        # 0: "head", 1: "neck" ,2: "r.shoulder", 3: "r.elbow", 4: "r.wrist", 5: "l.shoulder", 6: "l.elbow",
        # 7: "l.wrist", 8: "r.hip", 9: "r.knee", 10: "r.ankle", 11: "l.hip", 12: "l.knee", 13: "l.ankle",
        # 14: "l.eye", 15: "r.eye", 16: "l.ear", 17: "r.ear"
        joint_coords = np.zeros((17, 2))

        for part in range(17):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            if len(peaks) != 0:
                joint_coords[part] = peaks[0]
                # normalize coordinates
                joint_coords[part][0] /= oriImg.shape[1]
                joint_coords[part][1] /= oriImg.shape[0]

        return joint_coords.flatten()


def kmeans_cluster(samples):

    kmeans = KMeans(n_clusters=2).fit(samples)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #print(centroids)
    #print(labels)
    return centroids, labels



def get_bms_model(input_shape, person_appearance_shape, person_pose_shape, person_interaction_shape, seg_mask_shape, optical_flow_shape, input_shape_latent, predicting_frame_num):

    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(64, activation='tanh'))(input_latent)
    h = GRU(128, implementation=1, dropout = 0.2)(input_latent_);
    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(h)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    input1 = Input(shape=input_shape)
    #input2 = Input(shape=person_appearance_shape)
    input3 = Input(shape=person_pose_shape)
    #input4 = Input(shape=seg_mask_shape)
    #input5 = Input(shape=optical_flow_shape)
    #input6 = Input(shape=person_interaction_shape)

    #conv_layer = Conv2D(64, (3, 3), activation='relu', padding='same', strides = 2, data_format="channels_first") (input4)
    #scene_features = Lambda(pool_seg_features, output_shape=(observed_frame_num, 64,))([conv_layer, input1])
    #scene_encoder = GRU(256, implementation=1, dropout=0.2)(scene_features)

    #person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    #person_appearance_encoder = GRU(256, implementation=1, dropout=0.2)(person_appearance_encoder)

    person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input3)
    person_pose_encoder = GRU(256, implementation=1, dropout=0.2)(person_pose_encoder);

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1, dropout=0.2)(location_scale_encoder);

    #optical_flow_encoder = TimeDistributed(Dense(128, activation='tanh'))(input5)
    #optical_flow_encoder = GRU(256, implementation=1, dropout=0.2)(optical_flow_encoder);

    #person_interaction_encoder = TimeDistributed(Dense(128, activation='tanh'))(input6)
    #person_interaction_encoder = GRU(256, implementation=1, dropout=0.2)(person_interaction_encoder);

    decoder = concatenate([location_scale_encoder, person_pose_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True, dropout=0.2)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input3, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss= bms_loss)

    return full_model

def test(model, x_batch_test, pose_test_batch):

    dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)
    preds = model.predict([x_batch_test, pose_test_batch,
                            dummy_y], batch_size=batch_size * test_samples, verbose=1)
    #preds = np.reshape(preds, (int(x_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))

    return preds

def process (input_image, params, model_params, pose_model):

    oriImg = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    scale_search = [1, .5, 1.5, 2] # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:4]

    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in scale_search]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = pose_model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = input_image

    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

keras_weights_file = '../../../PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/model/keras/model.h5'


# load model
# authors of original model don't use
# vgg normalization (subtracting mean) on input images



#OPTICAL FLOW SETTINGS

# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:0']
controller = '/device:GPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = OPTICAL_FLOW_DIR + '/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

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

#MASK_RCNN SETTINGS
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main():

   # Definition of the parameters of DEEP_SORT
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    config = InferenceConfig()
    config.display()

    # Create mask-RCNN model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load mask-RCNN weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    #OpenPose model
    pose_model = get_testing_model()
    pose_model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()


    model_bms = get_bms_model((observed_frame_num, 4), (observed_frame_num, person_appearance_size),
                          (observed_frame_num, person_pose_size),
                          (observed_frame_num, person_interaction), semantic_maps,
                          (observed_frame_num, optic_flow_features), (predicting_frame_num, 4),
                          predicting_frame_num)

    model_bms.load_weights('M3P_LS + PP.h5')

    writeVideo_flag = True

    previousFrame = None
    
    video_capture = cv2.VideoCapture(sys.argv[1])


    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output_detections.avi', fourcc, 15, (w, h))
        out_flow = cv2.VideoWriter('output_flow.avi', fourcc, 15, (1500, 500))
        list_file = open('detection.txt', 'w')
        frame_index = -1


    observed_frames = 0
    loc_scale_input = np.zeros((observed_frame_num,4))

    #17 2D pose locations
    person_pose_input = np.zeros((observed_frame_num, 17*2))

    fps = 0.0
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break;
        t1 = time.time()

        if previousFrame is not None:
            img_pairs = []
            img1 = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pairs.append((img1, img2))

            #Run optical flow on the image pair and display
            #flow = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
            #plot = display_img_pairs_w_flows(img_pairs, flow)
            #fig = plot.gcf()
            #fig.canvas.draw()


            #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            #                    sep='')
            #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            #img_flow = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #out_flow.write(img_flow)

            # display image with opencv or any operation you like
            #cv2.imshow("plot", img_flow)


        frame_ = deepcopy(frame)
        previousFrame = frame_

        image = Image.fromarray(frame)
        image_ = np.asarray(image)
        results = model.detect([image_], verbose=1)

        r = results[0]
        idx = r['class_ids'] == 1
        boxs = r['rois'][idx]
        N = boxs.shape[0]
        for i in range(N):
            y1, x1, y2, x2 = boxs[i]
            boxs[i] = [x1, y1, x2 - x1, y2 - y1]

        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            # collect bounding boxes as an input vector to the prediction model
            loc_scale_input[observed_frames][0] = bbox[0] / w
            loc_scale_input[observed_frames][1] = bbox[1] / h
            loc_scale_input[observed_frames][2] = (bbox[2] - bbox[0]) / w
            loc_scale_input[observed_frames][3] = (bbox[3] - bbox[1]) / h

            # collect pose information as an input vector to the prediction model
            cropped_person = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            person_pose_input[observed_frames] = get_person_pose(cropped_person, pose_model, params,
                                                                 model_params)

            observed_frames += 1

            frame = process(frame, params, model_params, pose_model)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        if observed_frames == observed_frame_num:

            loc_scale_input_ = np.expand_dims(loc_scale_input, axis=0)
            loc_scale_input_ = np.repeat(loc_scale_input_, test_samples, axis=0)

            person_pose_input_ = np.expand_dims(person_pose_input, axis=0)
            person_pose_input_ = np.repeat(person_pose_input_, test_samples, axis=0)

            dummy_y = np.zeros((test_samples * test_samples, predicting_frame_num, 4)).astype(np.float32)
            preds = model_bms.predict([loc_scale_input_, person_pose_input_, dummy_y], batch_size=batch_size * test_samples, verbose=1)
            #preds = np.mean(preds, axis = 0)

            samples = np.zeros((test_samples, 4))

            #record possible directions
            for f in range(test_samples):
                sample = [int(preds[f][0][0] * w)+ int(preds[f][0][2] * w), int(preds[f][0][1] * h) + int(preds[f][0][3] * h),
                          int(preds[f][predicting_frame_num-1][0] * w) + int(preds[f][predicting_frame_num-1][2] * w),
                           int(preds[f][predicting_frame_num-1][1] * h) + int(preds[f][predicting_frame_num-1][3] * h)]
                samples[f] = sample
                #cv2.arrowedLine(frame, (int(preds[f][0][0] * w)+ int(preds[f][0][2] * w), int(preds[f][0][1] * h) + int(preds[f][0][3] * h)),
                         #(int(preds[f][predicting_frame_num-1][0] * w)+ int(preds[f][predicting_frame_num-1][2] * w), int(preds[f][predicting_frame_num-1][1] * h) + int(preds[f][predicting_frame_num-1][3] * h)), (0, 255, 0), 1)


            colors = [[255,0,0],[0,255,0], [0,0,255]]
            #cluster possible directions
            centroids, labels = kmeans_cluster(samples)
            labels = list(labels)

            #plot clusters + probability for each
            for s in range(centroids.shape[0]):
                cv2.arrowedLine(frame, (int(bbox[2]), int(bbox[3])), (int(centroids[s][2]),int(centroids[s][3])), tuple(colors[s]), 1)
                p = labels.count(s) / test_samples
                p = p * 100
                cv2.putText(frame, str(int(p)) + '%', (int(centroids[s][2]), int(centroids[s][3])), 0, 5e-3 * 200, (0, 0, 0), 2)

            #start observing frames again in a sliding window approach
            observed_frames -= 1
            loc_scale_input = np.roll(loc_scale_input, observed_frame_num-1, axis=0)
        #for det in detections:
            #bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




    video_capture.release()
    if writeVideo_flag:
        out.release()
        out_flow.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
