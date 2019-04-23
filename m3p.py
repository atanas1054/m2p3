import glob
import os
import cv2
import numpy as np
import random
import skimage.io


import sys
from utils_ import *
from visualize_result import *
from human_appearance import *
from segmentation_map import *
from optical_flow import *
from load_models import *

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
scene_features = 64
optic_flow_features = 50
#19 semantic classes of size 36x64
semantic_maps = (19, 36, 64)

#get mask_rcnn model
#mask_rcnn_model = get_mask_rcnn()

#get OpenPose model
#open_pose_model, open_pose_params, open_pose_model_params = get_open_pose()

#get DeepLab model
deeplab_model = DeepLabModel()

#get OpticalFlow model
#optical_flow_model = load_optical_flow_model()

def get_test_batches(x_data_test, appearance_test, pose_test, segmentation_test, optical_flow_test, y_data_test, input_seq, output_seq, test_samples):
    x_batch = x_data_test
    y_batch = y_data_test

    optical_flow_test_batch = optical_flow_test

    appearance_test_batch = appearance_test

    pose_test_batch = pose_test

    segmentation_test_batch = segmentation_test

    tbatch_size = x_data_test.shape[0]

    appearance_test_batch = np.expand_dims(appearance_test_batch, axis=1)
    appearance_test_batch = np.repeat(appearance_test_batch, test_samples, axis=1)

    pose_test_batch = np.expand_dims(pose_test_batch, axis=1)
    pose_test_batch = np.repeat(pose_test_batch, test_samples, axis=1)

    segmentation_test_batch = np.expand_dims(segmentation_test_batch, axis=1)
    segmentation_test_batch = np.repeat(segmentation_test_batch, test_samples, axis=1)

    optical_flow_test_batch = np.expand_dims(optical_flow_test_batch, axis=1)
    optical_flow_test_batch = np.repeat(optical_flow_test_batch, test_samples, axis=1)

    x_batch = np.expand_dims(x_batch, axis=1)
    x_batch = np.repeat(x_batch, test_samples, axis=1)

    y_batch = np.expand_dims(y_batch, axis=1)
    y_batch = np.repeat(y_batch, test_samples, axis=1)

    x_batch = np.reshape(x_batch, (tbatch_size*test_samples, input_seq, 4))

    appearance_test_batch = np.reshape(appearance_test_batch, (tbatch_size*test_samples, input_seq, person_appearance_size))

    pose_test_batch = np.reshape(pose_test_batch, (tbatch_size*test_samples, input_seq, person_pose_size))

    segmentation_test_batch = np.reshape (segmentation_test_batch, (tbatch_size*test_samples, semantic_maps[0], semantic_maps[1], semantic_maps[2]))

    optical_flow_test_batch = np.reshape(optical_flow_test_batch, (tbatch_size * test_samples, input_seq, optic_flow_features))

    y_batch = np.reshape(y_batch, (tbatch_size*test_samples, output_seq, 4))

    return (x_batch, appearance_test_batch, pose_test_batch, segmentation_test_batch, optical_flow_test_batch, y_batch)

def get_batch_gen(x_data, pose_data, appearance_data, segmentation_data, optic_flow_data, y_data,batch_size,input_seq,output_seq,train_samples):
    while 1:
        for i in range((int(x_data.shape[0]/batch_size))):
            idx = random.randint(0,(int(x_data.shape[0]/batch_size))-1)

            appearance_batch = appearance_data[idx*batch_size:idx*batch_size+batch_size, :]

            pose_batch = pose_data[idx*batch_size:idx*batch_size+batch_size, :]

            segmentation_batch = segmentation_data[idx*batch_size:idx*batch_size+batch_size, :]

            optic_flow_batch = optic_flow_data[idx*batch_size:idx*batch_size+batch_size, :]

            x_batch = x_data[idx*batch_size:idx*batch_size+batch_size,:]

            y_batch = y_data[idx*batch_size:idx*batch_size+batch_size,:]

            x_batch = np.expand_dims(x_batch,axis=1);
            x_batch = np.repeat(x_batch, train_samples, axis=1);

            appearance_batch = np.expand_dims(appearance_batch, axis=1)
            appearance_batch = np.repeat(appearance_batch, train_samples, axis=1)

            pose_batch = np.expand_dims(pose_batch, axis=1)
            pose_batch = np.repeat(pose_batch, train_samples, axis=1)

            segmentation_batch = np.expand_dims(segmentation_batch, axis=1)
            segmentation_batch  = np.repeat(segmentation_batch , train_samples, axis=1)

            optic_flow_batch = np.expand_dims(optic_flow_batch, axis=1)
            optic_flow_batch = np.repeat(optic_flow_batch, train_samples, axis=1)

            y_batch = np.expand_dims(y_batch, axis=1);
            y_batch = np.repeat(y_batch, train_samples, axis=1);

            appearance_batch = np.reshape(appearance_batch,(batch_size*train_samples,input_seq, person_appearance_size))
            pose_batch = np.reshape(pose_batch, (batch_size*train_samples,input_seq, person_pose_size))
            segmentation_batch = np.reshape(segmentation_batch, (batch_size*train_samples, semantic_maps[0], semantic_maps[1], semantic_maps[2]))
            optic_flow_batch = np.reshape(optic_flow_batch, (batch_size*train_samples, input_seq, optic_flow_features))
            x_batch = np.reshape(x_batch,(batch_size*train_samples,input_seq,4));
            y_batch = np.reshape(y_batch,(batch_size*train_samples,output_seq,4));

            yield [x_batch, appearance_batch, pose_batch, segmentation_batch, y_batch], y_batch


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

def pool_seg_features( args):

    seg_maps = args[0] #shape None, 64, 18, 32
    ped_locations = args[1] #shape None, 8, 4
    width = 32
    height = 18

    #shape None, 18, 32 , 64
    seg_maps = tf.transpose(seg_maps, perm=[0, 2, 3, 1])
    #shape None,8
    pos_x = ped_locations[:, :, 0]*width + ped_locations[:, :, 2]*width
    pos_x = K.print_tensor(pos_x, message="pos_x: ")
    print(K.int_shape((pos_x)))
    pos_y = ped_locations[:, :, 1]*height + ped_locations[:, :, 3]*height
    pos_x = K.cast(K.round(pos_x), 'int32')
    pos_y = K.cast(K.round(pos_y), 'int32')

    #pos_x = K.print_tensor(pos_x, message = "pos_x_norm: ")

    feature_maps = []
    b_size = tf.shape(seg_maps)[0]
    batch_idx = tf.range(0, b_size)

    for i in range(observed_frame_num):
        h_i = pos_y[:,i]
        w_i = pos_x[:,i]
        indices = K.stack([batch_idx, h_i, w_i], axis=1)
        f_i = tf.gather_nd(seg_maps, indices)
        feature_maps.append(f_i)

    f = K.stack(feature_maps, axis=1)
    f = K.print_tensor(f, message="seg_maps: ")

    # output shape None,8,64
    return f

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

def get_bms_model(input_shape, person_appearance_shape, person_pose_shape, seg_mask_shape, optical_flow_shape, input_shape_latent, predicting_frame_num):

    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(64, activation='tanh'))(input_latent)
    h = GRU(128, implementation=1, dropout = 0.2)(input_latent_);
    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(h)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=person_appearance_shape)
    input3 = Input(shape=person_pose_shape)
    input4 = Input(shape=seg_mask_shape)
    input5 = Input(shape=optical_flow_shape)

    conv_layer = Conv2D(64, (3, 3), activation='relu', padding='same', strides = 2, data_format="channels_first") (input4)
    scene_features = Lambda(pool_seg_features, output_shape=(observed_frame_num, 64,))([conv_layer, input1])
    scene_encoder = GRU(256, implementation=1)(scene_features)

    person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    person_appearance_encoder = GRU(256, implementation=1)(person_appearance_encoder)

    person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input3)
    person_pose_encoder = GRU(256, implementation=1)(person_pose_encoder);

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1)(location_scale_encoder);

    optical_flow_encoder = TimeDistributed(Dense(128, activation='tanh'))(input5)
    optical_flow_encoder = GRU(256, implementation=1)(optical_flow_encoder);

    decoder = concatenate([location_scale_encoder, person_appearance_encoder, person_pose_encoder, scene_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input2, input3, input4, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss= bms_loss)

    return full_model

def train(model, epochs, batch_gen, input, batch_size, x_batch_test, appearance_test_batch, pose_test_batch, segmentation_test_batch, optic_flow_test_batch, y_batch_test):

    for epoch in range(1, epochs):

        model.fit_generator(batch_gen,steps_per_epoch=(int(input.shape[0]/batch_size)),epochs=1,verbose=1,workers=1)

    dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)

    model.save_weights('M3P_without_PI_OF' + '.h5')

    preds = model.predict([x_batch_test, appearance_test_batch, pose_test_batch,
                           segmentation_test_batch, dummy_y], batch_size=batch_size * test_samples, verbose=1)
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
    #optical_flow_train = get_optical_flow(optical_flow_model, obs_train, train_paths, train_images)
    #np.save('optical_flow_train.npy', optical_flow_train)
    segmentation_masks_train = get_seg_map(deeplab_model, obs_train, train_paths, train_images)
    #np.save('segmentation_masks_train.npy', segmentation_masks_train)
    optical_flow_train = np.load('optical_flow_train.npy')
    print("Optical flow shape= ", optical_flow_train.shape)
    segmentation_masks_train = np.load('segmentation_masks_train.npy')
    print("Seg maps shape= ", segmentation_masks_train)
    person_appearance_train = np.load('person_appearance_train.npy')
    print("Appearance features shape=", person_appearance_train.shape)
    person_pose_train = np.load('person_pose_train.npy')
    print("Person pose shape=", person_pose_train.shape)
    input_train = get_location_scale(obs_train, observed_frame_num)
    print("Location scale shape=", input_train.shape)
    output_train = get_output(pred_train, predicting_frame_num)

    #Get testing data
    obs_test, pred_test, test_paths = get_raw_data(test_annotations, observed_frame_num, predicting_frame_num)
    #optical_flow_test = get_optical_flow(optical_flow_model, obs_test, test_paths, test_images)
   # np.save('optical_flow_test.npy', optical_flow_test)
    #segmentation_masks_test = get_seg_map(deeplab_model, obs_test, test_paths, test_images)
    #np.save('segmentation_masks_test.npy', segmentation_masks_test)
    #person_appearance_test = get_person_appearance(mask_rcnn_model, obs_test, test_paths, test_images)
    #np.save('person_appearance_test.npy', person_appearance_test)
    optical_flow_test = np.load('optical_flow_test.npy')
    #print(optical_flow_test.shape)
    segmentation_masks_test = np.load('segmentation_masks_test.npy')
    person_appearance_test = np.load('person_appearance_test.npy')
    person_pose_test = np.load('person_pose_test.npy')
    #person_pose_test = get_person_pose(open_pose_model, open_pose_params, open_pose_model_params, obs_test, test_paths, test_images)
    #print(person_pose_test.shape)
    #np.save('person_pose_test.npy', person_pose_test)

    input_test = get_location_scale(obs_test, observed_frame_num)
    output_test = get_output(pred_test, predicting_frame_num)
    (x_batch_test, appearance_test, pose_test, segmentation_test, optic_flow_test, y_batch_test) = get_test_batches(input_test, person_appearance_test, person_pose_test, segmentation_masks_test, optical_flow_test, output_test, observed_frame_num, predicting_frame_num, test_samples)

    #Get and train model
    batch_gen = get_batch_gen(input_train, person_pose_train, person_appearance_train, segmentation_masks_train, optical_flow_train, output_train, batch_size, observed_frame_num, predicting_frame_num, train_samples)
    #simple_model = get_simple_model((observed_frame_num, 4),(observed_frame_num,person_appearance_size), predicting_frame_num)
    model = get_bms_model((observed_frame_num, 4), (observed_frame_num,person_appearance_size), (observed_frame_num, person_pose_size), semantic_maps, (observed_frame_num, optic_flow_features), (predicting_frame_num, 4), predicting_frame_num)
    model.summary()
    preds = train(model, epochs, batch_gen, input_train, batch_size, x_batch_test, appearance_test, pose_test, segmentation_test, optic_flow_test, y_batch_test)


    #preds = train_simple(simple_model, input_train, person_appearance_train, output_train, input_test, person_appearance_test, output_test)
    #print(preds.shape)
    visualize_result(preds, pred_test, test_paths, test_images)




