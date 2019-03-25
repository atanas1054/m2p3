import glob
import os
import cv2
import numpy as np
import random
import skimage.io


import sys
from utils import *
from visualize_result import *
from load_mask_rcnn import *
from human_appearance import *
from load_open_pose import *

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda, RepeatVector, Dropout, Activation, Flatten
from keras.layers.merge import dot, add, multiply, concatenate
from keras.engine import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K


#TRAIN/TEST images and annotations JAAD dataset
train_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_frames/'
train_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_annotations/'
test_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_frames/'
test_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_annotations/'


observed_frame_num = 8
predicting_frame_num = 12
batch_size = 256
train_samples = 10
test_samples = 20
epochs = 1000
latent_dim = 64
person_appearance_size = 14*14
person_pose_size = 17*2

#get mask_rcnn model
mask_rcnn_model = get_mask_rcnn()

#get OpenPose model
open_pose_model, open_pose_params, open_pose_model_params = get_open_pose()


def get_test_batches(x_data_test, appearance_test, pose_test, y_data_test, input_seq, output_seq, test_samples):
    x_batch = x_data_test
    y_batch = y_data_test
    appearance_test_batch = appearance_test
    pose_test_batch = pose_test
    tbatch_size = x_data_test.shape[0]
    appearance_test_batch = np.expand_dims(appearance_test_batch, axis=1)
    appearance_test_batch = np.repeat(appearance_test_batch, test_samples, axis=1)
    pose_test_batch = np.expand_dims(pose_test_batch, axis=1)
    pose_test_batch = np.repeat(pose_test_batch, test_samples, axis=1)
    x_batch = np.expand_dims(x_batch, axis=1)
    x_batch = np.repeat(x_batch, test_samples, axis=1)
    y_batch = np.expand_dims(y_batch, axis=1)
    y_batch = np.repeat(y_batch, test_samples, axis=1)
    x_batch = np.reshape(x_batch, (tbatch_size*test_samples, input_seq, 4))
    appearance_test_batch = np.reshape(appearance_test_batch, (tbatch_size*test_samples, input_seq, person_appearance_size))
    pose_test_batch = np.reshape(pose_test_batch, (tbatch_size*test_samples, input_seq, person_pose_size))
    y_batch = np.reshape(y_batch, (tbatch_size*test_samples, output_seq, 4))

    return (x_batch, appearance_test_batch, pose_test_batch, y_batch)

def get_batch_gen(x_data, pose_data, appearance_data, y_data,batch_size,input_seq,output_seq,train_samples):
    while 1:
        for i in range((int(x_data.shape[0]/batch_size))):
            idx = random.randint(0,(int(x_data.shape[0]/batch_size))-1)
            appearance_batch = appearance_data[idx*batch_size:idx*batch_size+batch_size,:]
            pose_batch = pose_data[idx*batch_size:idx*batch_size+batch_size,:]
            x_batch = x_data[idx*batch_size:idx*batch_size+batch_size,:]
            y_batch = y_data[idx*batch_size:idx*batch_size+batch_size,:]

            x_batch = np.expand_dims(x_batch,axis=1);
            x_batch = np.repeat(x_batch, train_samples, axis=1);
            appearance_batch = np.expand_dims(appearance_batch, axis=1)
            appearance_batch = np.repeat(appearance_batch, train_samples, axis=1)
            pose_batch = np.expand_dims(pose_batch, axis=1)
            pose_batch = np.repeat(pose_batch, train_samples, axis=1)
            y_batch = np.expand_dims(y_batch, axis=1);
            y_batch = np.repeat(y_batch, train_samples, axis=1);

            appearance_batch = np.reshape(appearance_batch,(batch_size*train_samples,input_seq, person_appearance_size))
            pose_batch = np.reshape(pose_batch, (batch_size*train_samples,input_seq, person_pose_size))
            x_batch = np.reshape(x_batch,(batch_size*train_samples,input_seq,4));
            y_batch = np.reshape(y_batch,(batch_size*train_samples,output_seq,4));

            yield [x_batch, appearance_batch, pose_batch, y_batch], y_batch



def bms_loss( y_true, y_pred ):

    y_true = K.reshape( y_true, (batch_size,train_samples,predicting_frame_num,4) );
    y_pred = K.reshape( y_pred, (batch_size,train_samples,predicting_frame_num,4) );
    rdiff = K.mean(K.square(y_pred - y_true),axis=(2,3));
    rdiff_min = K.min( rdiff, axis = 1);
    return K.mean(rdiff_min)

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

def get_simple_model(location_scale_shape, person_appearance_shape, predicting_frame_num):

    input1 = Input(shape=location_scale_shape)
    input2 = Input(shape=person_appearance_shape)

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(128, implementation=1, dropout = 0.2)(location_scale_encoder);
    person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    person_appearance_encoder = GRU(128, implementation=1, dropout = 0.2)(person_appearance_encoder)

    decoder = concatenate([location_scale_encoder, person_appearance_encoder])
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(128, implementation=1, return_sequences=True, dropout = 0.2)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input2], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss='mse')

    return full_model

def get_bms_model(input_shape, person_appearance_shape, person_pose_shape, input_shape_latent, predicting_frame_num):

    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(64, activation='tanh'))(input_latent)
    h = GRU(256, implementation=1, dropout = 0.2)(input_latent_);
    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(h)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=person_appearance_shape)
    input3 = Input(shape=person_pose_shape)

    person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    person_appearance_encoder = GRU(256, implementation=1)(person_appearance_encoder)

    person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input3)
    person_pose_encoder = GRU(256, implementation=1)(person_pose_encoder);

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1)(location_scale_encoder);

    decoder = concatenate([location_scale_encoder, person_appearance_encoder, person_pose_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input2, input3, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss= bms_loss)

    return full_model

def train(model, epochs, batch_gen, input, batch_size, x_batch_test, appearance_test_batch, pose_test_batch, y_batch_test):

    for epoch in range(1, epochs):

        model.fit_generator(batch_gen,steps_per_epoch=(int(input.shape[0]/batch_size)),epochs=1,verbose=1,workers=1)

    dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)

    preds = model.predict([x_batch_test, appearance_test_batch, pose_test_batch,  dummy_y], batch_size=batch_size * test_samples, verbose=1)
    preds = np.reshape(preds, (int(x_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))

    gt = np.reshape(y_batch_test, (int(y_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))

    average_iou = bbox_iou(preds, gt)

    print("Average IoU: " + str(average_iou))
    #preds = np.reshape(preds, (int(x_batch_test.shape[0]/test_samples), predicting_frame_num, 4))

    return preds

def train_simple(model, train_location, train_appearance, train_y, test_location, test_appearance, test_y):

    model.fit([train_location, train_appearance], train_y,
              batch_size=1024,
              epochs=100,
              verbose=1,
              shuffle=False)

    pred = model.predict([test_location, test_appearance], batch_size=512)
    average_iou = bbox_iou(pred, test_y)

    print("Average IoU: " + str(average_iou))

    return pred



def get_location_scale(obs,observed_frame_num):

    loc_scale_input = []
    for i in range(len(obs)):
        loc_scale_input_ = location_scale_input(obs[i], observed_frame_num)
        loc_scale_input.append(loc_scale_input_)

    loc_scale_input = np.vstack(loc_scale_input)

    return loc_scale_input

def get_output(pred,predicting_frame_num):

    output = []
    for i in range(len(pred)):
        output_ = location_scale_output(pred[i], predicting_frame_num)
        output.append(output_)

    output = np.vstack(output)

    return output

def get_raw_data(path,observed_frame_num,predicting_frame_num):
    total_obs = []
    total_pred = []
    paths = []

    for file in glob.glob(path + "*.csv"):
        raw_data, numPeds = preprocess(file)
        data = get_traj_like(raw_data, numPeds)
        obs, pred = get_obs_pred_like(data, observed_frame_num, predicting_frame_num)

        paths.append(file)
        total_obs.append(obs)
        total_pred.append(pred)

    return total_obs, total_pred, paths


if __name__ == '__main__':

    #Get training data
    obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)
    #person_appearance = get_person_appearance(mask_rcnn_model, obs_train, train_paths, train_images)
    #person_pose_train = get_person_pose(open_pose_model, open_pose_params, open_pose_model_params, obs_train, train_paths, train_images)
    #np.save('person_pose_train.npy', person_pose_train)

    #np.save('person_appearance_vector.npy', person_appearance)
    person_appearance_train = np.load('person_appearance_train.npy')
    person_pose_train = np.load('person_pose_train.npy')
    input_train = get_location_scale(obs_train, observed_frame_num)
    output_train = get_output(pred_train, predicting_frame_num)

    #Get testing data
    obs_test, pred_test, test_paths = get_raw_data(test_annotations, observed_frame_num, predicting_frame_num)
    #person_appearance_test = get_person_appearance(mask_rcnn_model, obs_test, test_paths, test_images)
    #np.save('person_appearance_test.npy', person_appearance_test)
    person_appearance_test = np.load('person_appearance_test.npy')
    person_pose_test = np.load('person_pose_test.npy')
    #person_pose_test = get_person_pose(open_pose_model, open_pose_params, open_pose_model_params, obs_test, test_paths, test_images)
    #print(person_pose_test.shape)
    #np.save('person_pose_test.npy', person_pose_test)

    input_test = get_location_scale(obs_test, observed_frame_num)
    output_test = get_output(pred_test, predicting_frame_num)
    (x_batch_test, appearance_test, pose_test, y_batch_test) = get_test_batches(input_test, person_appearance_test, person_pose_test, output_test, observed_frame_num, predicting_frame_num, test_samples)


    #Get and train model
    batch_gen = get_batch_gen(input_train, person_pose_train, person_appearance_train, output_train, batch_size, observed_frame_num, predicting_frame_num, train_samples)
    #simple_model = get_simple_model((observed_frame_num, 4),(observed_frame_num,person_appearance_size), predicting_frame_num)
    model = get_bms_model((observed_frame_num, 4), (observed_frame_num,person_appearance_size), (observed_frame_num, person_pose_size), (predicting_frame_num, 4), predicting_frame_num)
    #model.summary()
    preds = train(model, epochs, batch_gen, input_train, batch_size, x_batch_test, appearance_test, pose_test, y_batch_test)


    #preds = train_simple(simple_model, input_train, person_appearance_train, output_train, input_test, person_appearance_test, output_test)
    print(preds.shape)
    visualize_result(preds, pred_test, test_paths, test_images)




