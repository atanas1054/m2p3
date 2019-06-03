import numpy as np
import os
import scipy.cluster
from keras import backend as K
from keras.engine import Layer
import tensorflow as tf
from tensorflow.keras.layers import Layer

from keras.engine.topology import Layer
import keras.backend as K


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling
        for channel-first images and relative bounding box coordinates

        # Constructor parameters
            pooled_height, pooled_width (int) --
              specify height and width of layer outputs

        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """

    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """

        def curried_pool_rois(x):
            return ROIPoolingLayer._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """

        def curried_pool_roi(roi):
            return ROIPoolingLayer._pool_roi(feature_map, roi,
                                             pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])

        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width * roi[1], 'int32')
        h_end = tf.cast(feature_map_height * roi[2], 'int32')
        w_end = tf.cast(feature_map_width * roi[3], 'int32')

        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')

        areas = [[(
            i * h_step,
            j * w_step,
            (i + 1) * h_step if i + 1 < pooled_height else region_height,
            (j + 1) * w_step if j + 1 < pooled_width else region_width
        )
            for j in range(pooled_width)]
            for i in range(pooled_height)]

        # take the maximum of each area and stack the result
        def pool_area(x):
            return tf.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])


        return pooled_features


class RoiPooling(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        assert (len(x) == 2)
        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]


            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            if self.dim_ordering == 'th':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]
                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], input_shape[1],
                                         y2 - y1, x2 - x1]
                            x_crop = img[:, :, y1:y2, x1:x2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3))
                            outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], y2 - y1,
                                         x2 - x1, input_shape[3]]
                            x_crop = img[:, y1:y2, x1:x2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)

        final_output = K.reshape(final_output, (1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output


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

def bbox_iou(bbox_pred, bbox_gt):

    """ Args
    :param bbox_pred NxTx4 : predicted bounding boxes [N, T, x, y, w, h] where N-> batch size, T-> sequence length
    :param bbox_gt NxTx4 : ground truth bounding boxes [N, T, x, y, w, h] where N-> batch size, T-> sequence length
    :return: average iou and final iou
    """
    epsilon = 1e-5

    iou_ = []


    #if bbox have sample dimension in addition to batch size and sequence length
    if len(bbox_pred.shape) > 3:

        for i in range(bbox_pred.shape[0]):
            best_iou = 0
            for j in range(bbox_pred.shape[1]):
                # Coordinates of intersection boxes
                x1 = np.array([bbox_pred[i, j, :, 0], bbox_gt[i, j, :, 0]]).max(axis=0)
                y1 = np.array([bbox_pred[i, j, :, 1], bbox_gt[i, j, :, 1]]).max(axis=0)
                x2 = np.array([bbox_pred[i, j, :, 0] + bbox_pred[i, j, :, 2], bbox_gt[i, j, :, 0] + bbox_gt[i, j, :, 2]]).min(axis=0)
                y2 = np.array([bbox_pred[i, j, :, 1] + bbox_pred[i, j, :, 3], bbox_gt[i, j, :, 1] + bbox_gt[i, j, :, 3]]).min(axis=0)

                # AREAS OF OVERLAP - Area where the boxes intersect
                width = (x2 - x1)
                height = (y2 - y1)

                # handle case where there is NO overlap
                width[width < 0] = 0
                height[height < 0] = 0

                area_overlap = width * height

                # combined areas
                area_a = (bbox_pred[i, j, :, 0] + bbox_pred[i, j, :, 2] - bbox_pred[i, j, :, 0]) * (
                            bbox_pred[i, j, :, 1] + bbox_pred[i, j, :, 3] - bbox_pred[i, j, :, 1])
                area_b = (bbox_gt[i, j, :, 0] + bbox_gt[i, j, :, 2] - bbox_gt[i, j, :, 0]) * (
                            bbox_gt[i, j, :, 1] + bbox_gt[i, j, :, 3] - bbox_gt[i, j, :, 1])
                area_combined = area_a + area_b - area_overlap

                # intersection over union
                iou = area_overlap / (area_combined + epsilon)
                miou = np.mean(iou, axis=0)
                if miou > best_iou:
                    best_iou = miou

            iou_.append(best_iou)
        av_iou = np.mean(iou_)

        return av_iou

    else:
        for i in range(bbox_pred.shape[0]):
            #Coordinates of intersection boxes
            x1 = np.array([bbox_pred[i, :, 0], bbox_gt[i, :, 0]]).max(axis=0)
            y1 = np.array([bbox_pred[i, :, 1], bbox_gt[i, :, 1]]).max(axis=0)
            x2 = np.array([bbox_pred[i, :, 0] + bbox_pred[i, :, 2], bbox_gt[i, :, 0] + bbox_gt[i, :, 2]]).min(axis=0)
            y2 = np.array([bbox_pred[i, :, 1] + bbox_pred[i, :, 3], bbox_gt[i, :, 1] + bbox_gt[i, :, 3]]).min(axis=0)

            # AREAS OF OVERLAP - Area where the boxes intersect
            width = (x2 - x1)
            height = (y2 - y1)

            # handle case where there is NO overlap
            width[width < 0] = 0
            height[height < 0] = 0

            area_overlap = width * height

            # combined areas
            area_a = (bbox_pred[i, :, 0] + bbox_pred[i, :, 2] - bbox_pred[i, :, 0]) * (bbox_pred[i, :, 1] + bbox_pred[i, :, 3] - bbox_pred[i, :, 1])
            area_b = (bbox_gt[i, :, 0] + bbox_gt[i, :, 2] - bbox_gt[i, :, 0]) * (bbox_gt[i, :, 1] + bbox_gt[i, :, 3] - bbox_gt[i, :, 1])
            area_combined = area_a + area_b - area_overlap

            #intersection over union
            iou = area_overlap / (area_combined + epsilon)
            miou = np.mean(iou, axis = 0)
            iou_.append(miou)

        av_iou = np.mean(iou_)

    return av_iou