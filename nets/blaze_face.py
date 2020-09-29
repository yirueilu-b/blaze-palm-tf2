import tensorflow as tf

# Blaze Face

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 128, 128, 3

n_classes = 2
n_boxes = [2, 6]


# ![](https://i.imgur.com/5MjmVOim.png) ![](https://i.imgur.com/srgPydMm.png)

# Blaze Block

def single_blaze_block(x, optional_path, filters, strides):
    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding='same')(x)
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(dw_conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    if optional_path:
        input_channel = x.shape[-1]
        output_channel = conv.shape[-1]
        max_pooling = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=strides, padding='same')(x)
        if input_channel != output_channel:
            channel_pad = tf.keras.backend.concatenate([max_pooling, tf.zeros_like(max_pooling)], axis=-1)
            out = tf.keras.layers.Add()([conv, channel_pad])
        else:
            out = tf.keras.layers.Add()([conv, max_pooling])
        out = tf.keras.layers.Activation("relu")(out)
    else:
        out = tf.keras.layers.Activation("relu")(conv)
    return out


def double_blaze_block(x, optional_path, strides):
    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=strides, padding='same')(x)
    proj_conv = tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=1, padding='same')(dw_conv)
    proj_conv = tf.keras.layers.BatchNormalization()(proj_conv)
    proj_conv = tf.keras.layers.Activation("relu")(proj_conv)
    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=1, padding='same')(proj_conv)
    expand_conv = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=1, padding='same')(dw_conv)
    expand_conv = tf.keras.layers.BatchNormalization()(expand_conv)
    if optional_path:
        input_channel = x.shape[-1]
        output_channel = expand_conv.shape[-1]
        max_pooling = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=strides, padding='same')(x)
        if input_channel != output_channel:
            channel_pad = tf.keras.backend.concatenate([max_pooling, tf.zeros_like(max_pooling)], axis=-1)
            out = tf.keras.layers.Add()([expand_conv, channel_pad])
        else:
            out = tf.keras.layers.Add()([expand_conv, max_pooling])
        out = tf.keras.layers.Activation("relu")(out)
    else:
        out = tf.keras.layers.Activation("relu")(expand_conv)
    return out


# Feature Extractor

x_in = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

convolution1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=2, padding='same')(x_in)
convolution1 = tf.keras.layers.BatchNormalization()(convolution1)
convolution1 = tf.keras.layers.Activation("relu")(convolution1)

single_blaze_block1 = single_blaze_block(x=convolution1, optional_path=True, filters=24, strides=1)
single_blaze_block2 = single_blaze_block(x=single_blaze_block1, optional_path=True, filters=24, strides=1)
single_blaze_block3 = single_blaze_block(x=single_blaze_block2, optional_path=True, filters=48, strides=2)
single_blaze_block4 = single_blaze_block(x=single_blaze_block3, optional_path=True, filters=48, strides=1)
single_blaze_block5 = single_blaze_block(x=single_blaze_block4, optional_path=True, filters=48, strides=1)

double_blaze_block1 = double_blaze_block(x=single_blaze_block3, optional_path=True, strides=2)
double_blaze_block2 = double_blaze_block(x=double_blaze_block1, optional_path=True, strides=1)
double_blaze_block3 = double_blaze_block(x=double_blaze_block2, optional_path=True, strides=1)
double_blaze_block4 = double_blaze_block(x=double_blaze_block3, optional_path=True, strides=2)
double_blaze_block5 = double_blaze_block(x=double_blaze_block4, optional_path=True, strides=1)
double_blaze_block6 = double_blaze_block(x=double_blaze_block5, optional_path=True, strides=1)

x_out = [double_blaze_block3, double_blaze_block6]
model = tf.keras.Model(inputs=x_in, outputs=x_out)

# Feature to Prediction

# 16x16 bounding box - Confidence, [batch_size, 16, 16, 2]
bb_16_conf = tf.keras.layers.Conv2D(filters=n_boxes[0] * 1, kernel_size=3, padding='same', activation='sigmoid')(
    model.output[0])
# reshape [batch_size, 16**2 * #bbox(2), 1]
bb_16_conf_reshaped = tf.keras.layers.Reshape((16 ** 2 * n_boxes[0], 1))(bb_16_conf)

# 8 x 8 bounding box - Confindece, [batch_size, 8, 8, 6]
bb_8_conf = tf.keras.layers.Conv2D(filters=n_boxes[1] * 1, kernel_size=3, padding='same', activation='sigmoid')(
    model.output[1])
# reshape [batch_size, 8**2 * #bbox(6), 1]
bb_8_conf_reshaped = tf.keras.layers.Reshape((8 ** 2 * n_boxes[1], 1))(bb_8_conf)
# Concatenate confidence prediction
# shape : [batch_size, 896, 1]
conf_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])

# 16x16 bounding box - loc [x, y, w, h]
bb_16_loc = tf.keras.layers.Conv2D(filters=n_boxes[0] * 4, kernel_size=3, padding='same')(model.output[0])
# [batch_size, 16**2 * #bbox(2), 4]
bb_16_loc_reshaped = tf.keras.layers.Reshape((16 ** 2 * n_boxes[0], 4))(bb_16_loc)

# 8x8 bounding box - loc [x, y, w, h]
bb_8_loc = tf.keras.layers.Conv2D(filters=n_boxes[1] * 4, kernel_size=3, padding='same')(model.output[1])
bb_8_loc_reshaped = tf.keras.layers.Reshape((8 ** 2 * n_boxes[1], 4))(bb_8_loc)

# Concatenate  location prediction
loc_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])
output_combined = tf.keras.layers.Concatenate(axis=-1)([conf_of_bb, loc_of_bb])
