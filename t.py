import tensorflow as tf
import numpy as np
from nets import blaze_palm
from utils.loss_function import SSDLoss

from utils.anchors_generator import AnchorsGenerator
from utils.anchor_config import AnchorsConfig
from utils.encoder import match
import cv2


def preprocess(bgr_image):
    # convert to rgb
    rgb_image = bgr_image[:, :, ::-1]
    # pad to square and resize
    shape = np.r_[rgb_image.shape]
    padding = (shape.max() - shape[:2]).astype('uint32') // 2
    rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
    rgb_image = cv2.resize(rgb_image, (256, 256))
    rgb_image = np.ascontiguousarray(rgb_image)
    # normalize
    rgb_image = np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))
    # reshape as input shape
    rgb_image = rgb_image[tf.newaxis, ...]
    return rgb_image, padding


anchor_config = AnchorsConfig(fixed_anchor_size=False)
anchor_generator = AnchorsGenerator(anchor_config)
anchors = anchor_generator.generate()
anchors_normalized = np.array([[anchor.x_center, anchor.y_center, anchor.w, anchor.h] for anchor in anchors])

model = blaze_palm.build_blaze_palm_model()

image = np.random.random((480, 640, 3))
input_image, _ = preprocess(image)
x_data = np.array([input_image[0], ] * 1000)

ground_truth_labels = np.array([[1]])
ground_truth_box = np.array(
    [129.64256 - 60.13495 / 2, 153.4913 - 60.13495 / 2, 129.64256 + 60.13495 / 2, 153.4913 + 60.13495 / 2, 136.83605957,
     177.14094543, 118.33981323, 127.28881073, 130.82832336, 126.52574158, 142.27386475, 131.28793335, 152.84526062,
     137.67770386, 119.28163147, 173.47927856, 105.67642212, 152.75184631]).reshape(1, -1)
annotation = [ground_truth_labels, ground_truth_box / 256]
num_anchors = 2944
num_coordinates = 18
num_key_points = 7
num_classes = 1
positive_threshold = 0.5
negative_threshold = 0.5
y_data = np.zeros((1000, num_anchors, num_classes + num_coordinates))
for i in range(1000):
    conf, loc = match(annotation, anchors_normalized, match_threshold=0.5)
    y_data[i] = np.concatenate([conf, loc * 256], axis=-1)

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss()
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

history = model.fit(x=x_data, y=y_data, steps_per_epoch=10, epochs=10, batch_size=4)
