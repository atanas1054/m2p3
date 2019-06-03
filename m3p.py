import os
import cv2
import numpy as np
import random
import skimage.io
from scipy import stats

import sys
from utils_ import *
from visualize_result import *
from human_appearance import *
from segmentation_map import *
from person_interaction import *
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
from sklearn import linear_model


#TRAIN/TEST images and annotations JAAD dataset
train_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_frames/'
train_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/train_annotations/'
test_images = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_frames/'
test_annotations = '/media/atanas/New Volume/M3P/Datasets/JAAD/JAAD_clips/test_annotations/'


observed_frame_num = 10
predicting_frame_num = 15
batch_size = 1024
train_samples = 1
test_samples = 1
epochs = 1000
latent_dim = 24
person_appearance_size = 14*14
person_pose_size = 17*2
scene_features = 64
optic_flow_features = 50
# 19 semantic classes of size 36x64
semantic_maps = (19, 36, 64)
# 4 * 30 people per frame
person_interaction = 120
#4x3 grids in x and y directions
ego_motion = 24

#get mask_rcnn model
#mask_rcnn_model = get_mask_rcnn()

#get OpenPose model
#open_pose_model, open_pose_params, open_pose_model_params = get_open_pose()

#get DeepLab model
#deeplab_model = DeepLabModel()

#get OpticalFlow model
#optical_flow_model = load_optical_flow_model()

def get_test_batches(x_data_test, appearance_test, pose_test, person_interaction_test, segmentation_test, optical_flow_test, ego_motion_test, y_data_test, input_seq, output_seq, test_samples):
    x_batch = x_data_test
    y_batch = y_data_test

    optical_flow_test_batch = optical_flow_test

    appearance_test_batch = appearance_test

    pose_test_batch = pose_test

    person_interaction_batch = person_interaction_test

    segmentation_test_batch = segmentation_test

    ego_motion_test_batch = ego_motion_test

    tbatch_size = pose_test_batch.shape[0]

    appearance_test_batch = np.expand_dims(appearance_test_batch, axis=1)
    appearance_test_batch = np.repeat(appearance_test_batch, test_samples, axis=1)

    pose_test_batch = np.expand_dims(pose_test_batch, axis=1)
    pose_test_batch = np.repeat(pose_test_batch, test_samples, axis=1)

    person_interaction_batch = np.expand_dims(person_interaction_batch, axis=1)
    person_interaction_batch = np.repeat(person_interaction_batch, test_samples, axis=1)

    #segmentation_test_batch = np.expand_dims(segmentation_test_batch, axis=1)
    #segmentation_test_batch = np.repeat(segmentation_test_batch, test_samples, axis=1)

    optical_flow_test_batch = np.expand_dims(optical_flow_test_batch, axis=1)
    optical_flow_test_batch = np.repeat(optical_flow_test_batch, test_samples, axis=1)

    ego_motion_test_batch = np.expand_dims(ego_motion_test_batch, axis=1)
    ego_motion_test_batch = np.repeat(ego_motion_test_batch, test_samples, axis=1)

    x_batch = np.expand_dims(x_batch, axis=1)
    x_batch = np.repeat(x_batch, test_samples, axis=1)

    y_batch = np.expand_dims(y_batch, axis=1)
    y_batch = np.repeat(y_batch, test_samples, axis=1)

    x_batch = np.reshape(x_batch, (x_data_test.shape[0]*test_samples, input_seq, 4))

    #appearance_test_batch = np.reshape(appearance_test_batch, (appearance_test.shape[0]*test_samples, input_seq, person_appearance_size))

    pose_test_batch = np.reshape(pose_test_batch, (tbatch_size*test_samples, input_seq, person_pose_size))

    #segmentation_test_batch = np.reshape (segmentation_test_batch, (tbatch_size*test_samples, semantic_maps[0], semantic_maps[1], semantic_maps[2]))

    #optical_flow_test_batch = np.reshape(optical_flow_test_batch, (optical_flow_test.shape[0] * test_samples, input_seq, optic_flow_features))

    #person_interaction_batch = np.reshape(person_interaction_batch, (person_interaction_test.shape[0] * test_samples, input_seq, person_interaction))
    #ego_motion_test_batch = np.reshape(ego_motion_test_batch, (ego_motion_test.shape[0] * test_samples, input_seq, ego_motion))

    y_batch = np.reshape(y_batch, (y_data_test.shape[0]*test_samples, output_seq, 4))

    return (x_batch, pose_test_batch, y_batch)

def get_batch_gen(x_data, pose_data, appearance_data, person_interaction_data, segmentation_data, optic_flow_data, ego_motion_data, y_data,batch_size,input_seq,output_seq,train_samples):
    while 1:
        for i in range((int(x_data.shape[0]/batch_size))):
            idx = random.randint(0,(int(x_data.shape[0]/batch_size))-1)

            # appearance_batch = appearance_data[idx*batch_size:idx*batch_size+batch_size, :]
            #
            pose_batch = pose_data[idx*batch_size:idx*batch_size+batch_size, :]
            #
            # segmentation_batch = segmentation_data[idx*batch_size:idx*batch_size+batch_size, :]
            #
            # optic_flow_batch = optic_flow_data[idx*batch_size:idx*batch_size+batch_size, :]
            #
            # person_interaction_batch = person_interaction_data[idx*batch_size:idx*batch_size+batch_size, :]
            #
            # ego_motion_batch = ego_motion_data[idx*batch_size:idx*batch_size+batch_size, :]

            x_batch = x_data[idx*batch_size:idx*batch_size+batch_size,:]

            y_batch = y_data[idx*batch_size:idx*batch_size+batch_size,:]

            # xy_batch = np.concatenate([x_batch, y_batch], axis = 1)
            #
            # xy_batch = np.expand_dims(xy_batch, axis=1);
            # xy_batch = np.repeat(xy_batch, train_samples, axis=1);
            #
            x_batch = np.expand_dims(x_batch,axis=1);
            x_batch = np.repeat(x_batch, train_samples, axis=1);
            #
            # appearance_batch = np.expand_dims(appearance_batch, axis=1)
            # appearance_batch = np.repeat(appearance_batch, train_samples, axis=1)
            #
            pose_batch = np.expand_dims(pose_batch, axis=1)
            pose_batch = np.repeat(pose_batch, train_samples, axis=1)

            # segmentation_batch = np.expand_dims(segmentation_batch, axis=1)
            # segmentation_batch  = np.repeat(segmentation_batch , train_samples, axis=1)
            #
            # optic_flow_batch = np.expand_dims(optic_flow_batch, axis=1)
            # optic_flow_batch = np.repeat(optic_flow_batch, train_samples, axis=1)
            #
            # ego_motion_batch = np.expand_dims(ego_motion_batch, axis=1)
            # ego_motion_batch = np.repeat(ego_motion_batch, train_samples, axis=1)
            #
            # person_interaction_batch =  np.expand_dims(person_interaction_batch, axis=1)
            # person_interaction_batch =  np.repeat(person_interaction_batch, train_samples, axis=1)

            y_batch = np.expand_dims(y_batch, axis=1);
            y_batch = np.repeat(y_batch, train_samples, axis=1);

            # appearance_batch = np.reshape(appearance_batch,(batch_size*train_samples,input_seq, person_appearance_size))
            pose_batch = np.reshape(pose_batch, (batch_size*train_samples,input_seq, person_pose_size))
            # segmentation_batch = np.reshape(segmentation_batch, (batch_size*train_samples, semantic_maps[0], semantic_maps[1], semantic_maps[2]))
            # optic_flow_batch = np.reshape(optic_flow_batch, (batch_size*train_samples, input_seq, optic_flow_features))
            # person_interaction_batch = np.reshape(person_interaction_batch, (batch_size*train_samples, input_seq, person_interaction))
            # ego_motion_batch = np.reshape(ego_motion_batch, (batch_size*train_samples, input_seq, ego_motion))
            x_batch = np.reshape(x_batch,(batch_size*train_samples,input_seq,4));
            y_batch = np.reshape(y_batch,(batch_size*train_samples,output_seq,4));

           # xy_batch = np.reshape(xy_batch, (batch_size * train_samples, input_seq + output_seq, 4));

            yield [x_batch, y_batch], y_batch


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
    #pos_x = K.print_tensor(pos_x, message="pos_x: ")

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
    #f = K.print_tensor(f, message="seg_maps: ")

    # output shape None,8,64
    return f


#Vanilla GRU encoder-decoder
def get_simple_model(location_scale_shape, person_pose_shape, predicting_frame_num):

    input1 = Input(shape=location_scale_shape)
    #input2 = Input(shape=person_pose_shape)

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1, dropout = 0.2)(location_scale_encoder);

    #person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    #person_pose_encoder = GRU(256, implementation=1, dropout = 0.2)(person_pose_encoder)

    #decoder = concatenate([location_scale_encoder, person_pose_encoder])
    decoder = Dense(128, activation='tanh')(location_scale_encoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True, dropout = 0.2)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss='mse')

    return full_model

#best of many CVAE GRU encoder-decoder
def get_bms_model(input_shape, person_appearance_shape, person_pose_shape, person_interaction_shape, seg_mask_shape, optical_flow_shape, ego_motion_shape, input_shape_latent, predicting_frame_num):

    input_latent = Input(shape=input_shape_latent)
    input_latent_ = TimeDistributed(Dense(128, activation='relu'))(input_latent)
    h = GRU(128, implementation=1)(input_latent_);

    input1 = Input(shape=input_shape)
    #input2 = Input(shape=person_appearance_shape)
    input3 = Input(shape=person_pose_shape)
    #input4 = Input(shape=seg_mask_shape)
    #input5 = Input(shape=optical_flow_shape)
    #input6 = Input(shape=person_interaction_shape)
    #input7 = Input(shape=ego_motion_shape)



    #conv_layer = Conv2D(64, (3, 3), activation='relu', padding='same', strides = 2, data_format="channels_first") (input4)
    #scene_features = Lambda(pool_seg_features, output_shape=(observed_frame_num, 64,))([conv_layer, input1])
    #scene_encoder = GRU(256, implementation=1, dropout=0.2)(scene_features)

    #person_appearance_encoder = TimeDistributed(Dense(128, activation='tanh'))(input2)
    #person_appearance_encoder = GRU(256, implementation=1, dropout=0.2)(person_appearance_encoder)

    #person_pose_encoder = TimeDistributed(Dense(128, activation='tanh'))(input3)
    #person_pose_encoder = GRU(256, implementation=1, dropout=0.2)(person_pose_encoder);

    location_scale_encoder = TimeDistributed(Dense(128, activation='tanh'))(input1)
    location_scale_encoder = GRU(256, implementation=1, dropout=0.2)(location_scale_encoder);

    concat = concatenate([location_scale_encoder, h])

    z_mean_var = Dense(latent_dim * 2, activity_regularizer=kl_activity_reg)(concat)
    z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
    z = Choose()(z)

    #ego_motion_encoder = TimeDistributed(Dense(128, activation='tanh'))(input7)
    #ego_motion_encoder = GRU(256, implementation=1, dropout=0.2)(ego_motion_encoder);

    #optical_flow_encoder = TimeDistributed(Dense(128, activation='tanh'))(input5)
    #optical_flow_encoder = GRU(256, implementation=1, dropout=0.2)(optical_flow_encoder);

    #person_interaction_encoder = TimeDistributed(Dense(128, activation='tanh'))(input6)
    #person_interaction_encoder = GRU(256, implementation=1, dropout=0.2)(person_interaction_encoder);

    decoder = concatenate([location_scale_encoder, z]);
    decoder = Dense(128, activation='tanh')(decoder);
    decoder = RepeatVector(predicting_frame_num)(decoder);
    decoder = GRU(256, implementation=1, return_sequences=True, dropout=0.2)(decoder);
    decoder = TimeDistributed(Dense(4))(decoder)

    full_model = Model(inputs=[input1, input_latent], outputs=decoder)
    full_model.compile(optimizer=Adam(lr=1e-3), loss= bms_loss)

    return full_model

#Train best of many CVAE GRU encoder-decoder
def train(model, epochs, batch_gen, input, batch_size):

    for epoch in range(1, epochs):

        model.fit_generator(batch_gen,steps_per_epoch=(int(input.shape[0]/batch_size)),epochs=1,verbose=1,workers=1)

    #mu, sigma = 0, 1.0

    model.save_weights('CVAE_RLS_1000epochs_1_train_sample_24_latent_dim' + '.h5')
    # final_preds = []
    # final_gt = []
    # num_saples = 20
    # for sample in range(num_saples):
    #
    #     #dummy_y = np.random.normal(mu, sigma, y_batch_test.shape[0] * test_samples*predicting_frame_num*4)
    #     #dummy_y = np.reshape(dummy_y, [y_batch_test.shape[0] * test_samples, predicting_frame_num, 4])
    #     dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)
    #
    #     preds = model.predict([x_batch_test, pose_test_batch, dummy_y], batch_size=batch_size * test_samples, verbose=1)
    #
    #     preds = np.reshape(preds, (int(x_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))
    #
    #     #add last observed frame to the relative output to get absolute output
    #     x_test = np.reshape(x_batch_test, (int(x_batch_test.shape[0] / test_samples), test_samples, observed_frame_num, 4))
    #     x_t = x_test[:,:,observed_frame_num-1,:]
    #     x_t = np.expand_dims(x_t, axis=2)
    #     x_t = np.repeat(x_t, predicting_frame_num, axis=2)
    #     preds = preds + x_t
    #     final_preds.append(preds)
    #
    #     gt = np.reshape(y_batch_test, (int(y_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))
    #     gt = gt + x_t
    #     final_gt.append(gt)
    #
    #
    # final_preds = np.reshape(final_preds, [preds.shape[0], num_saples*test_samples, predicting_frame_num, 4])
    # final_gt = np.reshape(final_gt, [preds.shape[0], num_saples*test_samples, predicting_frame_num, 4])
    # average_iou = bbox_iou(final_preds, final_gt)
    #
    # mse = calc_mse(final_preds, final_gt)
    #
    # print("Average IoU: " + str(average_iou))
    # print("MSE: " + str(mse))


    return 0


def test(model):

    num_samples = 1
    final_preds = np.zeros((x_batch_test.shape[0], num_samples * test_samples, predicting_frame_num, 4))
    final_gt = np.zeros((x_batch_test.shape[0], num_samples * test_samples, predicting_frame_num, 4))


    #x_test = np.reshape(x_batch_test,
                        #(int(x_batch_test.shape[0] / test_samples), test_samples, observed_frame_num, 4))
    x_t = x_batch_test[:, observed_frame_num - 1, :]
    x_t = np.expand_dims(x_t, axis=1)
    x_t = np.repeat(x_t, predicting_frame_num, axis=1)

    gt = y_batch_test
    #gt = np.reshape(y_batch_test,
                    #(int(y_batch_test.shape[0] / test_samples), test_samples, predicting_frame_num, 4))
    gt = gt + x_t


    #generate num_samples ammount of samples (predictions)
    for sample in range(num_samples):

        dummy_y = np.zeros((y_batch_test.shape[0] * test_samples, predicting_frame_num, 4)).astype(np.float32)

        preds = model.predict([x_batch_test, dummy_y], batch_size=batch_size * test_samples, verbose=1)

        # add last observed frame to the relative output to get absolute output
        preds = preds + x_t
        final_preds[:,sample,:,:] = preds
        final_gt[:,sample,:,:] = gt

    #cluster predictions and return cluster with highest probability
    # final_preds_ = []
    # for p in range(preds.shape[0]):
    #     for f in range(predicting_frame_num):
    #         samples = np.zeros((num_samples * test_samples, 4))
    #         for sample in range(num_samples * test_samples):
    #             samples[sample] = final_preds[p][sample][f][:]
    #         #centroids, labels = km_cluster(samples)
    #         #labels = list(labels)
    #         #weights = [labels.count(x) / (num_samples*test_samples) for x in labels]
    #
    #         #mode = max(set(labels), key=labels.count)
    #
    #         #pred = centroids[mode]
    #         #pred = samples[1]
    #         #pred = np.average(samples, axis=0, weights = weights)
    #         pred = np.mean(samples,axis=0)
    #         #print("SAMPLES: ", samples)
    #         #print("PREDICTION: ",pred)
    #         #print("GT: ", gt[p][f][:])

            #final_preds_.append(pred)

    if num_samples == 1:
        final_preds = np.reshape(final_preds, [preds.shape[0], predicting_frame_num, 4])
        final_gt = np.reshape(final_gt, [preds.shape[0], predicting_frame_num, 4])

    average_iou = bbox_iou(final_preds, final_gt)

    mse = calc_mse(final_preds, final_gt)

    print("Average IoU: " + str(average_iou))
    print("MSE: " + str(mse))

    return final_preds


#train Vanilla GRU encoder-decoder
def train_simple(model, train_location, train_pose, train_y, test_location, test_pose, test_y):

    model.fit([train_location], train_y,
              batch_size=1024,
              epochs=100,
              verbose=0,
              shuffle=False)

    pred = model.predict([test_location], batch_size=512)

    # add last observed frame to the relative output to get absolute output
    x_t = test_location[:, observed_frame_num-1, :]
    x_t = np.expand_dims(x_t, axis=1)
    x_t = np.repeat(x_t, predicting_frame_num, axis=1)
    pred = pred + x_t

    test_y = test_y + x_t
    #average_iou = bbox_iou(pred, test_y)

    #print("Average IoU: " + str(average_iou))

    return pred, test_y


def train_linear_model(train_location, train_pose, train_y, test_location, test_pose, test_y):

    #train_x = np.concatenate((train_location, train_pose), axis=2)
    train_x = train_location
    train_x  = train_x.reshape((train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1] * train_y.shape[2]))

    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)

    #test_x = np.concatenate((test_location, test_pose), axis=2)
    test_x = test_location
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))

    preds = regr.predict(test_x)
    preds = preds.reshape((preds .shape[0], predicting_frame_num, 4))

    # add last observed frame to the relative output to get absolute output
    x_t = test_location[:, observed_frame_num - 1, :]
    x_t = np.expand_dims(x_t, axis=1)
    x_t = np.repeat(x_t, predicting_frame_num, axis=1)
    preds = preds + x_t

    test_y = test_y + x_t

    average_iou = bbox_iou(preds, test_y)
    mse =  calc_mse(preds, test_y)

    print("Average IoU: " + str(average_iou))

    print("MSE: "+ str(mse))

    return preds



if __name__ == '__main__':


    #Get training data
    obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)

    #pose_train = get_person_pose_(obs_train, train_paths, train_images)
    #np.save('pose_train.npy', pose_train)
    #print(pose_train.shape)
    person_pose_train = np.load('pose_train.npy')
    #ego_motion_train = get_optical_flow_scene(optical_flow_model, obs_train, train_paths, train_images)
    #np.save('ego_motion_train.npy', ego_motion_train)
    ego_motion_train = np.load('ego_motion_train.npy')
    print("Ego motion shape= ", ego_motion_train.shape)
    optical_flow_train = np.load('optical_flow_train.npy')
    print("Optical flow shape= ", optical_flow_train.shape)
    segmentation_masks_train = np.load('segmentation_masks_train.npy')
    print("Seg maps shape= ", segmentation_masks_train.shape)
    #person_interaction_train = np.load('person_interaction_train.npy')
    #print("Person interactions shape= ", person_interaction_train.shape)
    person_appearance_train = np.load('person_appearance_train.npy')
    print("Appearance features shape=", person_appearance_train.shape)
    #person_pose_train = np.load('person_pose_train.npy')
    print("Person pose shape=", person_pose_train.shape)
    input_train = get_location_scale(obs_train, observed_frame_num)
    print("Location scale shape=", input_train.shape)
    output_train = get_output(pred_train, predicting_frame_num)
    print("Output shape=", output_train.shape)

    person_interaction_train = np.load('person_interaction_train.npy')
    print("Person interaction shape =", person_interaction_train.shape)
    #segmentation_masks_train = np.load('segmentation_masks_train.npy')

    #make output relative to the last observed frame
    i_t  = input_train[:,observed_frame_num-1,:]
    i_t =  np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t , predicting_frame_num, axis=1)
    output_train = output_train - i_t

    #Get testing data
    obs_test, pred_test, test_paths = get_raw_data(test_annotations, observed_frame_num, predicting_frame_num)



    #pose_test = get_person_pose_(obs_test, test_paths, test_images)
    #np.save('pose_test.npy', pose_test)
    person_pose_test = np.load('pose_test.npy')
    #print(pose_train.shape)
    #ego_motion_test = get_optical_flow_scene(optical_flow_model, obs_test, test_paths, test_images)
    #np.save('ego_motion_test.npy', ego_motion_test)
    ego_motion_test = np.load('ego_motion_test.npy')
    #person_interaction_test = get_geometric_person_interaction(obs_test, test_paths, test_images, tes
    #optical_flow_test = get_optical_flow(optical_flow_model, obs_test, test_paths, test_images)
    #segmentation_masks_test = get_seg_map(deeplab_model, obs_test, test_paths, test_images)
    #np.save('segmentation_masks_test.npy', segmentation_masks_test)
    #person_appearance_test = get_person_appearance(mask_rcnn_model, obs_test, test_paths, test_images)
    #np.save('person_appearance_test.npy', person_appearance_test)
    optical_flow_test = np.load('optical_flow_test.npy')
    #print(optical_flow_test.shape)
    segmentation_masks_test = np.load('segmentation_masks_test.npy')
    person_appearance_test = np.load('person_appearance_test.npy')
    #person_pose_test = np.load('person_pose_test.npy')
    person_interaction_test = np.load('person_interaction_test.npy')
    #person_pose_test = get_person_pose(open_pose_model, open_pose_params, open_pose_model_params, obs_test, test_paths, test_images)
    #print(person_pose_test.shape)
    #np.save('person_pose_test.npy', person_pose_test)

    input_test = get_location_scale(obs_test, observed_frame_num)
    output_test = get_output(pred_test, predicting_frame_num)

    # make output relative to the last observed frame
    i_t = input_test[:, observed_frame_num - 1, :]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, predicting_frame_num, axis=1)
    output_test = output_test - i_t

    (x_batch_test, pose_test, y_batch_test) =\
        get_test_batches(input_test, person_appearance_test, person_pose_test, person_interaction_test, segmentation_masks_test,
                         optical_flow_test, ego_motion_test, output_test, observed_frame_num, predicting_frame_num, test_samples)


    batch_gen = get_batch_gen(input_train, person_pose_train, person_appearance_train, person_interaction_train, segmentation_masks_train,
                              optical_flow_train, ego_motion_train, output_train, batch_size, observed_frame_num, predicting_frame_num, train_samples)





    ##########################################CVAE#################################
    model = get_bms_model((observed_frame_num, 4), (observed_frame_num,person_appearance_size), (observed_frame_num, person_pose_size),
                          (observed_frame_num, person_interaction), semantic_maps, (observed_frame_num, optic_flow_features), (observed_frame_num, ego_motion),
                          (predicting_frame_num, 4), predicting_frame_num)


    #preds = train(model, epochs, batch_gen, input_train, batch_size)
    model.load_weights('CVAE_RLS_1000epochs_1_train_sample_24_latent_dim.h5')
    preds = test(model)
    ################################################################################




   ##########################Vanilla GRU encoder-decoder ##########################
    # num_outputs = 20
    # final_preds = np.zeros((x_batch_test.shape[0], num_outputs, predicting_frame_num, 4))
    # final_gt = np.zeros((x_batch_test.shape[0], num_outputs, predicting_frame_num, 4))
    #
    # for s in range(num_outputs):
    #     print("Training model {}".format(s))
    #     simple_model = get_simple_model((observed_frame_num, 4),(observed_frame_num,person_pose_size), predicting_frame_num)
    #     preds, gt = train_simple(simple_model, input_train, person_pose_train, output_train, input_test, person_pose_test, output_test)
    #     final_preds[:,s,:,:] = preds
    #     final_gt[:,s,:,:] = gt
    #
    # #final_preds = np.reshape(final_preds, [preds.shape[0], s+1, predicting_frame_num, 4])
    # #final_gt = np.reshape(final_gt, [preds.shape[0], s+1, predicting_frame_num, 4])
    # average_iou = bbox_iou(final_preds, final_gt)
    # print(average_iou)
    # mse = calc_mse(final_preds, final_gt)
    # print("MSE: " + str(mse))
    ######################################################################################




    ###########################Linear Model#########################################
    #preds = train_linear_model(input_train, person_pose_train, output_train, input_test, person_pose_test, output_test)
    ######################################################################################




    ##########################Constant Velocity Model#####################################
    # diffs = np.diff(input_test, axis  = 1)
    # av_velocity = np.mean(diffs, axis =1)
    #
    # total_preds = []
    # for i in range(input_test.shape[0]):
    #     preds = np.zeros((predicting_frame_num, 4))
    #     preds[0] = input_test[i, observed_frame_num-1] + av_velocity[i]
    #     for j in range(predicting_frame_num-1):
    #         preds[j+1] = preds[j] + av_velocity[i]
    #     total_preds.append(preds)
    #
    # total_preds = np.reshape(total_preds, [len(total_preds), predicting_frame_num, 4])
    # final_gt = output_test + i_t
    # average_iou = bbox_iou(total_preds, final_gt)
    # print("Average IoU: " + str(average_iou))
    # mse = calc_mse(total_preds, final_gt)
    # print("MSE: " + str(mse))
    #######################################################################################



    ##################VISUALIZATION#############################
    visualize_result(preds, obs_test, pred_test, test_paths, test_images)
    ###########################################################################




