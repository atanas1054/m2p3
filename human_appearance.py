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
                del activations
                gc.collect()

            activations_.append(activation_)
            print(str(count)+"/"+str(total))

    activations_ = np.reshape(activations_, [len(activations_), obs[0].shape[1], feature_size])

    return activations_


def get_person_pose(obs, paths):

    return 0