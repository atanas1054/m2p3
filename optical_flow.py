from skimage.io import imread
import os
from keras.layers import Input
from utils_ import *
from keras.models import Model
import tensorflow as tf
from scipy.misc import imresize
import cv2
from scipy.signal import medfilt2d
import scipy.misc
import tensorflow as tf
import sys

#SfMLearner
#SfMdir = os.path.abspath("./OpticalFlow/SfMLearner-master/")

#sys.path.append(SfMdir)  # To find local version of the library

#from SfMLearner import SfMLearner
#from kitti_eval.pose_evaluation_utils import *

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
                rois = np.reshape(rois, (1, n_rois, 4))

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

                flow[frame] = current_flow

            # extrapolate optical flow for last frame
            last_flow = np.array([[flow[observed_frames_num - 3]], [flow[observed_frames_num - 2]]])
            diff = np.diff(last_flow, axis=0)
            flow[observed_frames_num - 1] = flow[observed_frames_num - 2] + diff

            final_flow.append(flow)

    final_flow = np.reshape(final_flow, [len(final_flow), obs[0].shape[1], optic_flow_feature_size])

    return final_flow



#https://github.com/tinghuiz/SfMLearner extract ego pose

def get_ego_pose(obs, paths, path_to_images):

    #model takes 5 frames as input
    seq_length = 5

    img_height = 128
    img_width = 416
    #camera tx, ty, tz, rx, ry, rz
    pose_features = 6
    max_src_offset = (seq_length - 1) // 2

    observed_frames_num = obs[0].shape[1]
    ckpt_file = "./OpticalFlow/SfMLearner-master/models/kitti_pose_model/model-100280"
    total_pose = []
    #initiate SfMlearner
    sfm = SfMLearner()
    sfm.setup_inference(img_height,
                        img_width,
                        'pose',
                        seq_length)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])

    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)

        for i in range(len(obs)):
                for person in range(obs[i].shape[0]):
                    pose_first_half = []
                    pose_second_half = []
                    for frame in range(observed_frames_num):

                        img = scipy.misc.imread(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                            int(obs[i][person][frame][1])) + ".png")
                        print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                            int(obs[i][person][frame][1])) + ".png")
                        img = scipy.misc.imresize(img, (img_height, img_width))

                        if frame == seq_length or frame == 0:

                            if frame == seq_length:
                                img = scipy.misc.imread(
                                    path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                                        int(obs[i][person][frame-1][1])) + ".png")
                                img = scipy.misc.imresize(img, (img_height, img_width))
                            image_seq = img
                            #print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(
                                #int(obs[i][person][frame][1])) + ".png")

                        else:
                            image_seq = np.hstack((image_seq, img))

                        if image_seq.shape[1] == img_width*seq_length:
                            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
                            pred_poses = pred['pose'][0]
                            # Insert the target pose [0, 0, 0, 0, 0, 0]
                            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=0)

                            # First frame as the origin
                            first_pose = pose_vec_to_mat(pred_poses[0])
                            for p in range(seq_length):
                                this_pose = pose_vec_to_mat(pred_poses[p])
                                this_pose = np.dot(first_pose, np.linalg.inv(this_pose))
                                tx = this_pose[0, 3]
                                ty = this_pose[1, 3]
                                tz = this_pose[2, 3]
                                rot = this_pose[:3, :3]
                                rz, ry, rx = mat2euler(rot)
                                pose= [tx, ty, tz, rx, ry, rz]
                                if frame <= seq_length:
                                    pose_first_half.append(pose)
                                else:
                                    pose = [tx + pose_first_half[seq_length-1][0], ty + pose_first_half[seq_length-1][1],
                                            tz + pose_first_half[seq_length-1][2], rx + pose_first_half[seq_length-1][3],
                                            ry + pose_first_half[seq_length-1][4], rz + pose_first_half[seq_length-1][5]]
                                    pose_second_half.append(pose)
                    total_pose_ = np.concatenate((pose_first_half, pose_second_half), axis=0)
                    total_pose.append(total_pose_)

    final_pose = np.reshape(total_pose, [len(total_pose), obs[0].shape[1], pose_features])

    return final_pose