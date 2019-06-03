import glob
import os
import sys
import cv2
import numpy as np
import random
import skimage.io
from PIL import Image
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
import tensorflow as tf

from load_models import *

from utils_ import *

tracker = os.path.abspath("./HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn")
sys.path.append(tracker)  # To find local version of the library

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

person_detector_path = os.path.abspath("./HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn/Mask_RCNN/")
sys.path.append(person_detector_path)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
# Import COCO config
sys.path.append(os.path.join(person_detector_path, "samples/coco/"))  # To find local version
import coco

COCO_MODEL_PATH = os.path.join(person_detector_path, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(person_detector_path, "logs")

#Person Detector SETTINGS
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#Pose Estimation

pose_estimation_dir = os.path.abspath("./PoseEstimation/keras_Realtime_Multi-Person_Pose_Estimation-master/")

# Import OpenPose
sys.path.append(pose_estimation_dir)  # To find local version of the library
import util


observed_frame_num = 10
predicting_frame_num = 15
batch_size = 256
train_samples = 1
test_samples = 20
latent_dim = 24
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

    kmeans = KMeans(n_clusters=3).fit(samples)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #print(centroids)
    #print(labels)
    return centroids, labels



def get_bms_model(input_shape, person_appearance_shape, person_pose_shape, person_interaction_shape, seg_mask_shape, optical_flow_shape, input_shape_latent, predicting_frame_num):
    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(128, activation='relu'))(input_latent)
    h = GRU(128, implementation=1)(input_latent_);

    input1 = Input(shape=input_shape)
    # input2 = Input(shape=person_appearance_shape)
    input3 = Input(shape=person_pose_shape)
    # input4 = Input(shape=seg_mask_shape)
    # input5 = Input(shape=optical_flow_shape)
    # input6 = Input(shape=person_interaction_shape)
    # input7 = Input(shape=ego_motion_shape)

    # conv_layer = Conv2D(64, (3, 3), activation='relu', padding='same', strides = 2, data_format="channels_first") (input4)
    # scene_features = Lambda(pool_seg_features, output_shape=(observed_frame_num, 64,))([conv_layer, input1])
    # scene_encoder = GRU(256, implementation=1, dropout=0.2)(scene_features)

    # person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    # person_appearance_encoder = GRU(256, implementation=1, dropout=0.2)(person_appearance_encoder)

    person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input3)
    person_pose_encoder = GRU(256, implementation=1, dropout=0.2)(person_pose_encoder);

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1, dropout=0.2)(location_scale_encoder);

    concat = concatenate([location_scale_encoder, person_pose_encoder, h])

    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(concat)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    # ego_motion_encoder = TimeDistributed(Dense(128, activation='tanh'))(input7)
    # ego_motion_encoder = GRU(256, implementation=1, dropout=0.2)(ego_motion_encoder);

    # optical_flow_encoder = TimeDistributed(Dense(128, activation='tanh'))(input5)
    # optical_flow_encoder = GRU(256, implementation=1, dropout=0.2)(optical_flow_encoder);

    # person_interaction_encoder = TimeDistributed(Dense(128, activation='tanh'))(input6)
    # person_interaction_encoder = GRU(256, implementation=1, dropout=0.2)(person_interaction_encoder);

    decoder = concatenate([location_scale_encoder, person_pose_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True, dropout=0.2)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input3, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss=bms_loss)

    return full_model


def test(model, x_batch_test, pose_test_batch):

    dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)
    preds = model.predict([x_batch_test, pose_test_batch,
                            dummy_y], batch_size=batch_size * test_samples, verbose=1)
    #preds = np.reshape(preds, (int(x_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))

    return preds



def main():
    # Definition of the parameters of DEEP_SORT tracker
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = './HumanAppearance/mask_RCNN+deepSORT/deep_sort_mask_rcnn/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    config = InferenceConfig()
    config.display()

    # person detector in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # load person detector weights
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    #load person pose model
    open_pose_model, open_pose_params, open_pose_model_params = get_open_pose()

    #load prediction model
    model_bms = get_bms_model((observed_frame_num, 4), (observed_frame_num,person_appearance_size), (observed_frame_num, person_pose_size),
                          (observed_frame_num, person_interaction), semantic_maps, (observed_frame_num, optic_flow_features), (predicting_frame_num, 4),
                          predicting_frame_num)

    model_bms.load_weights('CVAE_RLS_PP_1000epochs_1_train_sample_24_latent_dim.h5')
    #model_bms.load_weights('M3P_LS.h5')

    video_capture = cv2.VideoCapture('video6.mkv')
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*"XVID"), 30, (1920, 1080))
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    observed_frames = 0
    frame_counter = 0
    loc_scale_input = np.zeros((observed_frame_num,4))

    #17 2D pose locations
    person_pose_input = np.zeros((observed_frame_num, 17*2))

    while True:
        frame_counter +=1
        ret, frame = video_capture.read()
        if ret != True:
            break;

        #observe past data every 5th frame observed_frame_num times
        if frame_counter % 1 == 0:

            image = Image.fromarray(frame)
            image_ = np.asarray(image)
            results = model.detect([image_], verbose=0)

            r = results[0]
            idx = r['class_ids'] == 1
            boxs = r['rois'][idx]
            N = boxs.shape[0]
            for i in range(N):
                y1, x1, y2, x2 = boxs[i]
                boxs[i] = [x1, y1, x2 - x1, y2 - y1]

            features = encoder(frame, boxs)

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
                #collect bounding boxes as an input vector to the prediction model
                loc_scale_input[observed_frames][0] = bbox[0] / w
                loc_scale_input[observed_frames][1] = bbox[1] / h
                loc_scale_input[observed_frames][2] = (bbox[2] - bbox[0]) / w
                loc_scale_input[observed_frames][3] = (bbox[3] - bbox[1]) / h

                #collect pose information as an input vector to the prediction model
                cropped_person = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                person_pose_input[observed_frames] = get_person_pose(cropped_person, open_pose_model, open_pose_params, open_pose_model_params)

                observed_frames += 1
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
                #cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        #if we have observed enough frames, do a prediction
        if observed_frames == observed_frame_num:

            print(loc_scale_input)

            loc_scale_input_ = np.expand_dims(loc_scale_input, axis=0)
            loc_scale_input_ = np.repeat(loc_scale_input_, test_samples, axis=0)

            person_pose_input_ = np.expand_dims(person_pose_input, axis=0)
            person_pose_input_ = np.repeat(person_pose_input_, test_samples, axis=0)

            dummy_y = np.zeros((test_samples * test_samples, predicting_frame_num, 4)).astype(np.float32)
            preds = model_bms.predict([loc_scale_input_, person_pose_input_, dummy_y], batch_size=batch_size * test_samples, verbose=1)

            # add last observed frame to the relative output to get absolute output
            x_test = np.reshape(loc_scale_input_,
                               (int(loc_scale_input_.shape[0] / test_samples), test_samples, observed_frame_num, 4))
            x_t = x_test[:, :, observed_frame_num - 1, :]
            x_t = np.expand_dims(x_t, axis=2)
            x_t = np.repeat(x_t, predicting_frame_num, axis=2)
            preds = preds + x_t
            preds = np.reshape(preds,(preds.shape[1],preds.shape[2],preds.shape[3]))
            #preds = np.mean(preds, axis = 0)
            print(preds.shape)

           # for f in range(predicting_frame_num):
                #cv2.rectangle(frame, (int(preds[f][0] * w), int(preds[f][1] * h)), (int(preds[f][0] * w)+ int(preds[f][2] * w),
                                                                                     #int(preds[f][1] * h) + int(preds[f][3] * h)), (255, 255, 255), 2)

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
                direction = [int(centroids[s][2]) - int(centroids[s][0]),
                             int(centroids[s][3]) - int(centroids[s][1])]
                cv2.arrowedLine(frame, (int(bbox[2]), int(bbox[3])), (int(bbox[2]) + 7*direction[0], int(bbox[3]) + 7*direction[1]), tuple(colors[s]), 1)
                #cv2.arrowedLine(frame, (int(bbox[2]), int(bbox[3])), (int(preds[predicting_frame_num-1][0] * w)+ int(preds[predicting_frame_num-1][2] * w),
                                                                      #int(preds[predicting_frame_num-1][1] * h) + int(preds[predicting_frame_num-1][3] * h)), tuple(colors[s]), 1)
                p = labels.count(s) / test_samples
                p = p * 100
                cv2.putText(frame, str(int(p)) + '%', (int(bbox[2]) + 7*direction[0], int(bbox[3]) + 7*direction[1]), 0, 5e-3 * 200, (0, 0, 0), 2)

            #start observing frames again in a sliding window approach
            observed_frames -= 1
            loc_scale_input = np.roll(loc_scale_input, observed_frame_num-1, axis=0)
            #observed_frames = 0

        cv2.imwrite("results6_location+pose/frame%d.jpg" % frame_counter, frame)
        video.write(frame)
        #cv2.imshow('', frame)
        cv2.waitKey(1)


        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
