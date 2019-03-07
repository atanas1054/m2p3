import numpy as np
import os
import scipy.cluster
from keras import backend as K
from keras.engine import Layer

predicting_frame_num = 12


class Choose(Layer):
    def __init__(self, **kwargs):
        super(Choose, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        nx = K.random_normal(K.shape(inputs));
        return K.in_train_phase(inputs, nx)

    def get_config(self):
        config = {}
        base_config = super(Choose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):


        return input_shape



def preprocess(path):
    data = np.genfromtxt(path, delimiter=',')
    #numPeds = np.size(np.unique(data[1, :]))
    numPeds = np.unique(data[1, :])

    return data, numPeds


def get_traj_like(data, numPeds):
    '''
    reshape data format from [frame_ID, ped_ID, x,y,w,h]
    to pedestrian_num * [ped_ID, frame_ID, x,y,w,h]
    '''
    traj_data = []

    #sample every n frames
    sample = 5
    for pedIndex in numPeds:
        traj = []
        for i in range(len(data[1])):
            if data[1][i] == pedIndex and i % sample == 0:
                traj.append([data[1][i], data[0][i], data[2][i], data[3][i], data[4][i], data[5][i]])
        traj = np.reshape(traj, [-1, 6])

        traj_data.append(traj)

    return traj_data


def get_obs_pred_like(data, observed_frame_num, predicting_frame_num):
    """
    get input observed data and output predicted data
    """

    obs = []
    pred = []
    count = 0

    for pedIndex in range(len(data)):

        if len(data[pedIndex]) >= observed_frame_num + predicting_frame_num:
            seq = int((len(data[pedIndex]) - (observed_frame_num + predicting_frame_num)) / observed_frame_num) + 1

            for k in range(seq):
                obs_pedIndex = []
                pred_pedIndex = []
                count += 1
                for i in range(observed_frame_num):
                    obs_pedIndex.append(data[pedIndex][i+k*observed_frame_num])
                for j in range(predicting_frame_num):
                    pred_pedIndex.append(data[pedIndex][k*observed_frame_num+j+observed_frame_num])
                obs_pedIndex = np.reshape(obs_pedIndex, [observed_frame_num, 6])
                pred_pedIndex = np.reshape(pred_pedIndex, [predicting_frame_num, 6])

                obs.append(obs_pedIndex)
                pred.append(pred_pedIndex)

    obs = np.reshape(obs, [count, observed_frame_num, 6])
    pred = np.reshape(pred, [count, predicting_frame_num, 6])

    return obs, pred

def location_scale_input(obs, observed_frame_num):

    location_scale_input = []
    for pedIndex in range(len(obs)):
        person_pedIndex = []
        for i in range(observed_frame_num):
            person_pedIndex.append([obs[pedIndex][i][-4],obs[pedIndex][i][-3],obs[pedIndex][i][-2],obs[pedIndex][i][-1]])
        person_pedIndex = np.reshape(person_pedIndex, [observed_frame_num, 4])

        location_scale_input.append(person_pedIndex)

    location_scale_input = np.reshape(location_scale_input, [len(obs), observed_frame_num, 4])

    return location_scale_input

def location_scale_output(pred, predicting_frame_num):

    location_scale_ouput = []
    for pedIndex in range(len(pred)):
        person_pedIndex = []
        for i in range(predicting_frame_num):
            person_pedIndex.append([pred[pedIndex][i][-4],pred[pedIndex][i][-3],pred[pedIndex][i][-2],pred[pedIndex][i][-1]])
        person_pedIndex = np.reshape(person_pedIndex, [predicting_frame_num, 4])

        location_scale_ouput.append(person_pedIndex)

    location_scale_output = np.reshape(location_scale_ouput, [len(pred), predicting_frame_num, 4])

    return location_scale_output

def km_cluster( preds ):

    n_clusters = 4;
    _data_X = np.reshape(preds,(preds.shape[0],-1));
    centroids,_ = scipy.cluster.vq.kmeans(_data_X,n_clusters)
    idx,_ = scipy.cluster.vq.vq(_data_X,centroids)

    clusters = [[] for _ in range(n_clusters)];

    for data_idx in range(preds.shape[0]):
        clusters[idx[data_idx]].append(preds[data_idx,:]);

    cluster_means = [];
    for c_idx in range(n_clusters):
        cluster_means.append( np.mean( np.array(clusters[c_idx]), axis = 0) );

    return clusters