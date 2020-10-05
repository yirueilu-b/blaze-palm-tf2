import glob
import os
import json

import tensorflow as tf
import numpy as np
import cv2

from utils.anchors_generator import AnchorsGenerator
from utils.anchor_config import AnchorsConfig
from utils.encoder import match

anchor_config = AnchorsConfig(fixed_anchor_size=False)
anchor_generator = AnchorsGenerator(anchor_config)
anchors = anchor_generator.generate()
anchors_normalized = np.array([[anchor.x_center, anchor.y_center, anchor.w, anchor.h] for anchor in anchors])


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, annotation_dir, batch_size=32, image_shape=(256, 256, 3),
                 num_coordinates=18, num_anchors=2944,
                 shuffle=True):
        """
        :param image_dir: the directory contains all images
        :param annotation_dir: the directory contains all json annotation files
        :param batch_size: batch_size
        :param image_shape: image shape (width, height, channel)
        :param num_coordinates: 4 coordinates for 1 bounding box and 14 coordinates for 7 key points
        :param num_anchors: 8x8, 16x16 and 32x32 feature maps extract 8*8*6 + 16*16*2 + 32*32*2 = 2944 anchors
        :param shuffle: if shuffle data on each epoch ends
        """

        self.file_name_list = [os.path.split(file_name)[-1].replace('.jpg', '') for file_name in
                               glob.glob(os.path.join(image_dir, '*.jpg'))]
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_coordinates = num_coordinates
        self.num_anchors = num_anchors
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_name_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_name_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        file_name_list_temp = [self.file_name_list[k] for k in indexes]
        # Generate data
        image_batch, annotation_batch = self.data_generation(file_name_list_temp)

        return image_batch, annotation_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_name_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, file_name_list_temp):
        image_batch = np.empty((self.batch_size, *self.image_shape))
        annotation_batch = np.empty((self.batch_size, self.num_anchors, self.num_coordinates + 1))
        # Generate data
        for i, file_name in enumerate(file_name_list_temp):
            input_image = cv2.imread(os.path.join(self.image_dir, file_name + '.jpg'))
            input_image = self.preprocess(input_image)
            image_batch[i] = input_image

            with open(os.path.join(self.annotation_dir, file_name + '.json')) as json_file:
                annotation = json.load(json_file)
            labels = np.array([[int(anno['label'])] for anno in annotation])
            boxes = np.array([anno['bounding_box'] for anno in annotation])
            key_points = np.array([anno['key_points'] for anno in annotation])
            annotation = [labels, np.concatenate([boxes, key_points], axis=-1)]
            conf, loc = match(annotation, anchors_normalized, match_threshold=0.5)
            annotation_batch[i] = np.concatenate([conf, loc*256], axis=-1)

        return image_batch, annotation_batch

    def preprocess(self, bgr_image):
        # convert to rgb
        rgb_image = bgr_image[:, :, ::-1]
        # pad to square and resize
        shape = np.r_[rgb_image.shape]
        padding = (shape.max() - shape[:2]).astype('uint32') // 2
        rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
        rgb_image = cv2.resize(rgb_image, (self.image_shape[0], self.image_shape[1]))
        rgb_image = np.ascontiguousarray(rgb_image)
        # normalize
        rgb_image = np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))
        # reshape as input shape
        rgb_image = rgb_image[tf.newaxis, ...]
        return rgb_image


if __name__ == '__main__':
    train_data_generator = DataGenerator(image_dir=os.path.join('..', 'dataset', 'image'),
                                         annotation_dir=os.path.join('..', 'dataset', 'annotation'))
    batch = train_data_generator.__getitem__(0)
    print(batch[0].shape, batch[1].shape)
