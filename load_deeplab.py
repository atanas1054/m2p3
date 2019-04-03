import sys
import numpy as np
import tensorflow as tf
import os
from io import BytesIO
import tarfile
import tempfile
from PIL import Image

model_path = "./Segmentation/deeplabv3_cityscapes_train_2018_02_06.tar.gz"


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 1024
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    def __init__(self):
    #"""Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(model_path)
        for tar_info in tar_file.getmembers():
          if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
            file_handle = tar_file.extractfile(tar_info)
            graph_def = tf.GraphDef.FromString(file_handle.read())
            break

        tar_file.close()

        if graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

        #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            seg_map: Segmentation map of the image.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]
            })
        seg_map = batch_seg_map[0]
        return seg_map



