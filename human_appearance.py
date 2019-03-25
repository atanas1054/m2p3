import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import cv2
import copy

from scipy.ndimage.filters import gaussian_filter

ROOT_DIR = os.path.abspath("./PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/")

# Import OpenPose
sys.path.append(ROOT_DIR)  # To find local version of the library
import util


def get_person_appearance(model, obs, paths, path_to_images):

    #14x14 = 196 sized flattened feature vector
    feature_size = 14*14
    activation_ = np.zeros((obs[0].shape[1], feature_size))
    activations_ = []
    count = 0
    total = len(obs)*obs[0].shape[0]

    for i in range(len(obs)):
        for person in range(obs[i].shape[0]):
            count += 1

            for frame in range(obs[i].shape[1]):

                image = skimage.io.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")
                height_, width_, _ = image.shape

                x1 = int(obs[i][person][frame][2] * width_)
                y1 = int(obs[i][person][frame][3] * height_)
                x2 = int(obs[i][person][frame][2] * width_) + int(obs[i][person][frame][4] * width_)
                y2 = int(obs[i][person][frame][3] * height_) + int(obs[i][person][frame][5] * height_)

                cropped_person = image[y1:y2, x1:x2]


                #activations_74 is the final layer of the network
                activations = model.run_graph([cropped_person], [("activation_74", model.keras_model.get_layer("activation_74").output)])


                #extract feature vector of size 14x14x256 and average along the channel dimension
                activation = np.transpose(activations["activation_74"][0,0,:,:,:], [2, 0, 1])
                activation = np.mean(activation, axis=0)

                activation_[frame] = activation.flatten()

            activations_.append(activation_)
            print(str(count)+"/"+str(total))

    activations_ = np.reshape(activations_, [len(activations_), obs[0].shape[1], feature_size])

    return activations_


def get_person_pose(model, params, model_params, obs, paths, path_to_images):

    #17 2D joint locations
    feature_size = 17*2
    coords = np.zeros((obs[0].shape[1], feature_size))
    all_coords = []


    for i in range(len(obs)):
        #i = 42
        for person in range(obs[i].shape[0]):
            #person = 74
            for frame in range(obs[i].shape[1]):
                #frame = 0
                image = cv2.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                    int(obs[i][person][frame][1])) + ".png")

                height_, width_, _ = image.shape
                print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                    int(obs[i][person][frame][1])) + ".png")
                print('i: ',i," person: ",person, " frame: ", frame)
                x1 = int(obs[i][person][frame][2] * width_)
                y1 = int(obs[i][person][frame][3] * height_)
                x2 = int(obs[i][person][frame][2] * width_) + int(obs[i][person][frame][4] * width_)
                y2 = int(obs[i][person][frame][3] * height_) + int(obs[i][person][frame][5] * height_)

                cropped_person = image[y1:y2, x1:x2]

                oriImg = cropped_person

                #if a person is too small dont calculate pose and continue
                if oriImg.shape[0] < 15 or oriImg.shape[1] < 15:
                    joint_coords = np.zeros((17, 2))
                    coords[frame] = joint_coords.flatten()

                #else calculate pose
                else:
                    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

                    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
                    print(oriImg.shape[0], " ", oriImg.shape[1])
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

                    #coordinates of 17 joints of the human body
                    # 0: "head", 1: "neck" ,2: "r.shoulder", 3: "r.elbow", 4: "r.wrist", 5: "l.shoulder", 6: "l.elbow",
                    # 7: "l.wrist", 8: "r.hip", 9: "r.knee", 10: "r.ankle", 11: "l.hip", 12: "l.knee", 13: "l.ankle",
                    # 14: "l.eye", 15: "r.eye", 16: "l.ear", 17: "r.ear"
                    joint_coords = np.zeros((17,2))

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
                            #normalize coordinates
                            joint_coords[part][0] /= cropped_person.shape[1]
                            joint_coords[part][1] /= cropped_person.shape[0]

                    coords[frame] = joint_coords.flatten()

            all_coords.append(coords)

    all_coords_ = np.reshape(all_coords, [len(all_coords), obs[0].shape[1], feature_size])

    return all_coords_