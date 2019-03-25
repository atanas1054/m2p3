import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN-master/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
# Import COCO config
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import cv2
from PIL import Image

#matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
print(type(image))

video_capture = cv2.VideoCapture('video_0001.mp4')
# Run detection

while True:
    ret, frame = video_capture.read()
    image = Image.fromarray(frame)
    image_ = np.asarray(image)
    results = model.detect([image_], verbose=1)

#activations = model.run_graph([image], [
    #("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
    #("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    #("activation_74",           model.keras_model.get_layer("activation_74").output),
    #("roi",                model.keras_model.get_layer("ROI").output),
#])
#extract feature vector of size 14x14x256 (first element of "activation_74")
#display_images(np.transpose(activations["activation_74"][0,0,:,:,:1], [2, 0, 1]))
#model.keras_model.summary()
# Visualize results

    r = results[0]
    idx = r['class_ids'] == 1
    bboxs = r['rois'][idx]
    N = bboxs.shape[0]
    for i in range(N):
        y1, x1, y2, x2 = bboxs[i]
        bboxs[i] = [x1,y1,x2 - x1, y2 - y1]
    #print(r['class_ids'])
    print(bboxs)
    visualize.display_instances(image_, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])