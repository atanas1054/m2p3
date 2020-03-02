import os
import cv2
import numpy as np
import random
import skimage.io
from scipy import stats
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import sys
from utils_ import *
from visualize_result import *
#from human_appearance import *
#from segmentation_map import *
#from person_interaction import *
#from optical_flow import *
#from load_models import *

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda, RepeatVector, Dropout, Activation, Flatten
from keras.layers.merge import dot, add, multiply, concatenate
from keras.engine import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, Adadelta
from keras import backend as K
from tensorflow import convert_to_tensor
import tensorflow as tf
import argparse
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tqdm import tqdm

#TRAIN/TEST images and annotations JAAD dataset
train_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_frames/'
train_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_annotations/'
test_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_frames/'
test_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_annotations/'

#TRAIN/TEST annotations ActeV dataset
train_annotations_actev = '/media/atanas/New Volume/M3P/Datasets/ActeV/actev-data-repo/actev-v1-drop4-yaml/annotations/train_annotations/'
test_annotations_actev = '/media/atanas/New Volume/M3P/Datasets/ActeV/actev-data-repo/actev-v1-drop4-yaml/annotations/test_annotations/'

observed_frame_num = 10
predicting_frame_num = 15
batch_size = 1024
train_samples = 1
test_samples = 1
epochs = 6000
latent_dim = 24

#get mask_rcnn model
#mask_rcnn_model = get_mask_rcnn()

#get OpenPose model
#open_pose_model, open_pose_params, open_pose_model_params = get_open_pose()

#get DeepLab model
#deeplab_model = DeepLabModel()

#get OpticalFlow model
#optical_flow_model = load_optical_flow_model()

def get_test_batches(x_data_test, y_data_test, input_seq, output_seq, test_samples):

    x_batch = x_data_test
    y_batch = y_data_test


    x_batch = np.expand_dims(x_batch, axis=1)
    x_batch = np.repeat(x_batch, test_samples, axis=1)

    y_batch = np.expand_dims(y_batch, axis=1)
    y_batch = np.repeat(y_batch, test_samples, axis=1)

    x_batch = np.reshape(x_batch, (x_data_test.shape[0]*test_samples, input_seq, 4))

    y_batch = np.reshape(y_batch, (y_data_test.shape[0]*test_samples, output_seq, 4))

    return (x_batch, y_batch)

def get_batch_gen(x_data, y_data, batch_size, input_seq, output_seq, train_samples):

    while 1:
        for i in range((int(x_data.shape[0]/batch_size))):
            idx = random.randint(0,(int(x_data.shape[0]/batch_size))-1)

            x_batch = x_data[idx*batch_size:idx*batch_size+batch_size,:]

            y_batch = y_data[idx*batch_size:idx*batch_size+batch_size,:]

            x_batch = np.expand_dims(x_batch,axis=1);
            x_batch = np.repeat(x_batch, train_samples, axis=1);


            y_batch = np.expand_dims(y_batch, axis=1);
            y_batch = np.repeat(y_batch, train_samples, axis=1);


            x_batch = np.reshape(x_batch,(batch_size*train_samples,input_seq,4));
            y_batch = np.reshape(y_batch,(batch_size*train_samples,output_seq,4));

            yield [x_batch, y_batch], y_batch


#reconstrunction loss
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


#CVAE GRU encoder-decoder
def get_cvae_model(input_shape, input_shape_latent, predicting_frame_num):

    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(128, activation='tanh'))(input_latent)
    h = GRU(256, implementation=1)(input_latent_);

    input1 = Input(shape=input_shape)

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1)(location_scale_encoder);

    concat = concatenate([location_scale_encoder, h])

    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(concat)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    decoder = concatenate([location_scale_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=0.0001), loss= bms_loss)

    return full_model

#Train CVAE GRU encoder-decoder
def train(model, epochs, batch_gen, input, batch_size):

    for epoch in range(1, epochs):

        model.fit_generator(batch_gen, steps_per_epoch=(int(input.shape[0]/batch_size)), epochs=1, verbose=0, workers=1)
        print("Epoch :", epoch)

    print("Training done. Saving model...")
    model.save_weights('models/CVAE_model.h5')


def test(model, num_samples):

    final_preds = np.zeros((x_batch_test.shape[0], num_samples * test_samples, predicting_frame_num, 4))
    final_gt = np.zeros((x_batch_test.shape[0], num_samples * test_samples, predicting_frame_num, 4))

    #make output absolute
    x_t = x_batch_test[:, observed_frame_num - 1, :]
    x_t = np.expand_dims(x_t, axis=1)
    x_t = np.repeat(x_t, predicting_frame_num, axis=1)

    gt = y_batch_test
    gt = gt + x_t


    #generate num_samples ammount of samples (predictions)

    for sample in range(num_samples):

        dummy_y = np.zeros((y_batch_test.shape[0], predicting_frame_num, 4)).astype(np.float32)

        preds = model.predict([x_batch_test, dummy_y], batch_size=y_batch_test.shape[0], verbose=0)



        # add last observed frame to the relative output to get absolute output
        preds = preds + x_t
        final_preds[:,sample,:,:] = preds
        final_gt[:,sample,:,:] = gt



    if num_samples == 1:
        final_preds = np.reshape(final_preds, [preds.shape[0], predicting_frame_num, 4])
        final_gt = np.reshape(final_gt, [preds.shape[0], predicting_frame_num, 4])

        average_iou = bbox_iou(final_preds, final_gt)

        mse = calc_mse(final_preds, final_gt)

        ade = calc_ade(final_preds, final_gt)

        fde = calc_fde(final_preds, final_gt)

        print("ADE: " + str(ade))
        print("FDE: " + str(fde))
        print("Average IoU: " + str(average_iou))
        print("MSE: " + str(mse))

        return final_preds, []

    else:
        # K-means clustering
        clusters = 3
        probs = np.zeros((y_batch_test.shape[0], clusters))
        clustered_preds = np.zeros((y_batch_test.shape[0], clusters, predicting_frame_num, 4))
        for s in tqdm(range(final_preds.shape[0])):
            start = time.time()
            X = to_time_series_dataset(final_preds[s])
            km = TimeSeriesKMeans(n_clusters=clusters, metric="euclidean", verbose=False).fit(X)
            counts = np.bincount(km.labels_)
            counts = counts / num_samples
            probs[s, :] = counts
            clustered_preds[s, :, :, :] = km.cluster_centers_
            end = time.time()
            print((end - start) / y_batch_test.shape[0])
        final_preds = clustered_preds
        final_gt = final_gt[:, :clusters, :, :]

        average_iou = bbox_iou(final_preds, final_gt)

        mse = calc_mse(final_preds, final_gt)

        ade = calc_ade(final_preds, final_gt)

        fde = calc_fde(final_preds, final_gt)

        print("ADE: " + str(ade))
        print("FDE: " + str(fde))
        print("Average IoU: " + str(average_iou))
        print("MSE: " + str(mse))

        return final_preds, probs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=False, action='store_true', help='Train a model')
    parser.add_argument('--test', default=False, action='store_true', help='Test a model')
    parser.add_argument('--vis', default=False, action='store_true', help='Visualize results')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--num_samples', type=int, default=1,  help='Number of output predictions')

    args = parser.parse_args()

    #Get training data (past and future pedestrian bounding boxes)
    obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)
    input_train = get_location_scale(obs_train, observed_frame_num)
    output_train = get_output(pred_train, predicting_frame_num)

    print("Location scale train shape=", input_train.shape)

    #make output relative to the last observed frame
    i_t  = input_train[:,observed_frame_num - 1, :]
    i_t =  np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t , predicting_frame_num, axis=1)
    output_train = output_train - i_t

    print("Output train shape=", output_train.shape)

    #Get testing data (past and future pedestrian bounding boxes)
    obs_test, pred_test, test_paths = get_raw_data(test_annotations, observed_frame_num, predicting_frame_num)

    input_test = get_location_scale(obs_test, observed_frame_num)

    print("Location scale test shape=", input_train.shape)

    output_test = get_output(pred_test, predicting_frame_num)

    # make output relative to the last observed frame
    i_t = input_test[:, observed_frame_num - 1, :]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, predicting_frame_num, axis=1)
    output_test = output_test - i_t

    print("Output test shape=", output_train.shape)

    (x_batch_test, y_batch_test) = get_test_batches(input_test, output_test, observed_frame_num, predicting_frame_num, test_samples)

    batch_gen = get_batch_gen(input_train, output_train, batch_size, observed_frame_num, predicting_frame_num, train_samples)


    if args.train == True:
        ###################################CVAE TRAINING#################################
        model = get_cvae_model((observed_frame_num, 4), (predicting_frame_num, 4), predicting_frame_num)

        train(model, epochs, batch_gen, input_train, batch_size)

        #################################################################################

    if args.test == True:
        ###################################CVAE TESTING##################################
        num_samples = args.num_samples
        model_path = args.model
        model = get_cvae_model((observed_frame_num, 4), (predicting_frame_num, 4), predicting_frame_num)
        model.load_weights(model_path)
        predictions, probs = test(model, num_samples)

        #################################################################################

    if args.vis == True:
        ###################################VISUALIZATION#################################
        visualize_result(predictions, probs, obs_test, pred_test, test_paths, test_images)
        ###########################################################################

