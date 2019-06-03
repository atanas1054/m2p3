from skimage.io import imread
import os
from keras.layers import Input
from utils_ import *
from keras.models import Model
import tensorflow as tf
from scipy.misc import imresize
import cv2
from scipy.signal import medfilt2d

#Extracts (local) optical flow from the pedestrian bounding boxes using ROI pooling
def get_optical_flow(model, obs, paths, path_to_images):

    #Define ROI Pooling
    batch_size = 1
    img_height = 1080
    img_width = 1920
    n_channels = 1
    n_rois = 1
    pooled_height = 5
    pooled_width = 5

    feature_maps_shape = (batch_size, img_height, img_width, n_channels)
    feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
    roiss_tf = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
    pooled_features = roi_layer([feature_maps_tf, roiss_tf])

    observed_frames_num = obs[0].shape[1]

    #ROI size (5x5x2)
    feature_size = 50
    roi_person = np.zeros((observed_frames_num, feature_size))
    roi_final = []

    for i in range(len(obs)):
        for person in range(obs[i].shape[0]):

            for frame in range(observed_frames_num-1):

                img_pairs = []
                image1 = imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")
                image2 = imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame+1][1])) + ".png")
                height_, width_, _ = image2.shape
                print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")

                img_pairs.append((image1, image2))

                #calculate optical flow
                pred_labels = model.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)

                #Location of the person
                x1 = obs[i][person][frame][2]
                y1 = obs[i][person][frame][3]
                x2 = obs[i][person][frame][2] + obs[i][person][frame][4]
                y2 = obs[i][person][frame][3] + obs[i][person][frame][5]

                #Optical flow in x and y directions
                opt_flow_x = pred_labels[0][:, :, 0]
                if opt_flow_x.shape[0] != 1080 or opt_flow_x.shape[1] != 1920:
                    opt_flow_x = cv2.resize(opt_flow_x, (1920, 1080))

                opt_flow_x = np.reshape(opt_flow_x, (1, img_height, img_width, 1))

                opt_flow_y = pred_labels[0][:, :, 1]
                if opt_flow_y.shape[0] != 1080 or opt_flow_y.shape[1] != 1920:
                    opt_flow_y = cv2.resize(opt_flow_y, (1920, 1080))

                opt_flow_y = np.reshape(opt_flow_y, (1, img_height, img_width, 1))

                rois = [x1, y1, x2, y2]
                rois = np.reshape(rois, (1, num_rois, 4))

                #Get roi vectors in x and y
                with tf.Session() as session:
                    roi_vector_x = session.run(pooled_features,
                                         feed_dict={feature_maps_tf: opt_flow_x,
                                                    roiss_tf: rois})
                    roi_vector_y = session.run(pooled_features,
                                               feed_dict={feature_maps_tf: opt_flow_y,
                                                          roiss_tf: rois})

                roi_vector = np.stack((roi_vector_x, roi_vector_y), axis=1)

                #get a ROI vector of size 50 (5x5x2 for x and y)
                roi_person[frame] = roi_vector.flatten()
                print(i, " person: ", person, " frame: ", frame)

            #extrapolate optical flow for last frame
            last = np.array([[roi_person[observed_frames_num-3]], [roi_person[observed_frames_num-2]]])
            diff = np.diff(last, axis=0)
            roi_person[observed_frames_num-1] = roi_person[observed_frames_num-2] + diff

            roi_final.append(roi_person)


    roi_final = np.reshape(roi_final, [len(roi_final), obs[0].shape[1], feature_size])

    return roi_final


#Exctracts (global) optical flow for each pixel to represent ego-motion
def get_optical_flow_scene(model, obs, paths, path_to_images):

    observed_frames_num = obs[0].shape[1]

    #3x4 grids for x and y directions = 24 dimensions
    optic_flow_feature_size = 24

    flow = np.zeros((observed_frames_num, optic_flow_feature_size))

    final_flow = []

    for i in range(len(obs)):
        for person in range(obs[i].shape[0]):

            for frame in range(observed_frames_num-1):

                img_pairs = []
                image1 = imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")
                image2 = imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame+1][1])) + ".png")
                height_, width_, _ = image2.shape
                print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")

                img_pairs.append((image1, image2))

                #calculate optical flow
                pred_labels = model.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)

                #Optical flow in x direction
                opt_flow_x = pred_labels[0][:, :, 0]

                opt_flow_x = cv2.resize(opt_flow_x, (1600, 900))
                opt_flow_x = medfilt2d(opt_flow_x, 5)


                #reshape optical flow into 4x3 grids
                nrows = 300
                ncols = 400
                h, w = opt_flow_x.shape
                grids = opt_flow_x.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)

                #calculate x-direction mean flow  in every grid
                m_x = np.mean(grids, axis=(1, 2))

                # Optical flow in y direction
                opt_flow_y = pred_labels[0][:, :, 1]

                opt_flow_y = cv2.resize(opt_flow_y, (1600, 900))
                opt_flow_y = medfilt2d(opt_flow_y, 5)

                grids = opt_flow_y.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)

                #calculate y-direction mean flow in every grid
                m_y = np.mean(grids, axis=(1, 2))

                #concatenate mean x and mean y flows into one 24D vector
                current_flow = np.hstack((m_x, m_y))
                print(current_flow)

                flow[frame] = current_flow

            # extrapolate optical flow for last frame
            last_flow = np.array([[flow[observed_frames_num - 3]], [flow[observed_frames_num - 2]]])
            diff = np.diff(last_flow, axis=0)
            flow[observed_frames_num - 1] = flow[observed_frames_num - 2] + diff

            final_flow.append(flow)

    final_flow = np.reshape(final_flow, [len(final_flow), obs[0].shape[1], optic_flow_feature_size])

    return final_flow



