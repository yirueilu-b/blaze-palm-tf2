import os

import cv2
import numpy as np
import tensorflow as tf
from utils.loss_function import SSDLoss

MODEL_PATH = os.path.join('model', 'palm.h5')


class PalmDetector:
    def __init__(self, input_shape=(256, 256, 3)):
        self.padding = None
        self.scale = None
        self.input_shape = input_shape
        self.model = tf.keras.models.load_model(MODEL_PATH,
                                                custom_objects={
                                                    'compute_loss': SSDLoss(alpha=1. / 256.).compute_loss
                                                })

    def preprocess(self, image):
        # convert to rgb
        rgb_image = image[:, :, ::-1]
        # pad to square and resize
        shape = np.r_[rgb_image.shape]
        self.padding = (shape.max() - shape[:2]).astype('uint32') // 2
        self.scale = shape.max() / max(self.input_shape)
        rgb_image = np.pad(rgb_image,
                           ((self.padding[0], self.padding[0]),
                            (self.padding[1], self.padding[1]),
                            (0, 0)),
                           mode='constant')
        rgb_image = cv2.resize(rgb_image, (self.input_shape[0], self.input_shape[1]))
        rgb_image = np.ascontiguousarray(rgb_image)
        # normalize
        rgb_image = np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))
        # reshape as input shape
        rgb_image = rgb_image[tf.newaxis, ...]
        return rgb_image

    def detect(self, original_frame):
        input_image = self.preprocess(original_frame)
        prediction = self.model.predict(input_image)
        return prediction

    def rescale_result(self, bounding_boxes, key_points_list):
        bounding_boxes = bounding_boxes * self.scale
        bounding_boxes[:, 0] = bounding_boxes[:, 0] - self.padding[1]
        bounding_boxes[:, 1] = bounding_boxes[:, 1] - self.padding[0]

        key_points_list = key_points_list * self.scale
        key_points_list = key_points_list - self.padding[::-1]

        return bounding_boxes, key_points_list
