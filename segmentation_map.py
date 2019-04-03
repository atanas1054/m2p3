from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage

semantic_classes = 19
seg_map_size = [36, 64]


def get_seg_map(model, obs, paths, path_to_images):

    segmentation_masks = []

    for i in range(len(obs)):
        for person in range(obs[i].shape[0]):

            binary_mask = np.zeros((obs[i].shape[1], semantic_classes, seg_map_size[0], seg_map_size[1]))

            for frame in range(obs[i].shape[1]):
                image = Image.open(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")
                print(path_to_images + os.path.splitext(os.path.basename(paths[i]))[0] + "/" + str(int(obs[i][person][frame][1])) + ".png")

                #resize_ratio = 1.0 * 1024 / max(image.width, image.height)
                #19 semantic classes array with shape (image width * resize_ratio, image_heigh * resize_ratio)
                seg_map = model.run(image)

                #downsample segmentation map by a factor of 16
                seg_map_scaled = ndimage.interpolation.zoom(seg_map, .0625)

                for c in range(semantic_classes):
                    binary_mask[frame, c, :, :] = (seg_map_scaled == c)*1.0

            binary_mask = np.mean(binary_mask, axis=0)

            segmentation_masks.append(binary_mask)

    segmentation_masks = np.reshape(segmentation_masks, [len(segmentation_masks), semantic_classes, seg_map_size[0], seg_map_size[1]])

    return segmentation_masks