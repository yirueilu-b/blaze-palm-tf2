import tensorflow as tf
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 256, 256, 3


def blaze_palm_block(x, filters, strides):
    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(dw_conv)
    if strides == 2:
        input_channel = x.shape[-1]
        output_channel = conv.shape[-1]
        max_pooling = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=strides, padding='same')(x)
        if input_channel != output_channel:
            channel_pad = tf.keras.backend.concatenate([max_pooling, tf.zeros_like(max_pooling)], axis=-1)
            x = channel_pad
        else:
            x = max_pooling
    out = tf.keras.layers.Add()([conv, x])
    out = tf.keras.layers.Activation("relu")(out)
    return out


def build_blaze_palm_model(to_64=False):
    x_in = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

    convolution1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(x_in)
    blaze_palm_block_128 = tf.keras.layers.Activation("relu")(convolution1)

    for i in range(7): blaze_palm_block_128 = blaze_palm_block(x=blaze_palm_block_128, filters=32, strides=1)

    blaze_palm_block_64 = blaze_palm_block(x=blaze_palm_block_128, filters=64, strides=2)

    for i in range(7): blaze_palm_block_64 = blaze_palm_block(x=blaze_palm_block_64, filters=64, strides=1)

    blaze_palm_block_32 = blaze_palm_block(x=blaze_palm_block_64, filters=128, strides=2)

    for i in range(7): blaze_palm_block_32 = blaze_palm_block(x=blaze_palm_block_32, filters=128, strides=1)

    blaze_palm_block_16 = blaze_palm_block(x=blaze_palm_block_32, filters=256, strides=2)

    for i in range(7): blaze_palm_block_16 = blaze_palm_block(x=blaze_palm_block_16, filters=256, strides=1)

    blaze_palm_block_8 = blaze_palm_block(x=blaze_palm_block_16, filters=256, strides=2)

    for i in range(7): blaze_palm_block_8 = blaze_palm_block(x=blaze_palm_block_8, filters=256, strides=1)

    upsample_16 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=2)(blaze_palm_block_8)
    upsample_16 = tf.keras.layers.Activation("relu")(upsample_16)
    upsample_16 = tf.keras.layers.Add()([upsample_16, blaze_palm_block_16])
    upsample_16 = blaze_palm_block(x=upsample_16, filters=256, strides=1)

    # upsample_32 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=2)(upsample_16)
    # upsample_32 = tf.keras.layers.Activation("relu")(upsample_32)
    # upsample_32 = tf.keras.layers.Add()([upsample_32, blaze_palm_block_32])
    # upsample_32 = blaze_palm_block(x=upsample_32, filters=128, strides=1)

    # cls_8 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_32)
    cls_16 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_16)
    cls_32 = tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='same')(blaze_palm_block_8)
    # reshape_cls_8 = tf.keras.layers.Reshape([-1, 1])(cls_8)
    reshape_cls_16 = tf.keras.layers.Reshape([-1, 1])(cls_16)
    reshape_cls_32 = tf.keras.layers.Reshape([-1, 1])(cls_32)
    # conf = tf.keras.layers.Concatenate(axis=1)([reshape_cls_8, reshape_cls_16, reshape_cls_32])
    conf = tf.keras.layers.Concatenate(axis=1)([reshape_cls_16, reshape_cls_32])

    # reg_8 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_32)
    reg_16 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_16)
    reg_32 = tf.keras.layers.Conv2D(filters=108, kernel_size=(1, 1), strides=(1, 1), padding='same')(blaze_palm_block_8)
    # reshape_reg_8 = tf.keras.layers.Reshape([-1, 18])(reg_8)
    reshape_reg_16 = tf.keras.layers.Reshape([-1, 18])(reg_16)
    reshape_reg_32 = tf.keras.layers.Reshape([-1, 18])(reg_32)
    # loc = tf.keras.layers.Concatenate(axis=1)([reshape_reg_8, reshape_reg_16, reshape_reg_32])
    loc = tf.keras.layers.Concatenate(axis=1)([reshape_reg_16, reshape_reg_32])

    if to_64:
        upsample_64 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2)(upsample_32)
        upsample_64 = tf.keras.layers.Activation("relu")(upsample_64)
        upsample_64 = tf.keras.layers.Add()([upsample_64, blaze_palm_block_64])
        upsample_64 = blaze_palm_block(x=upsample_64, filters=64, strides=1)

        cls_8 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_64)
        cls_16 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_32)
        cls_32 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_16)
        cls_64 = tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            blaze_palm_block_8)
        reshape_cls_8 = tf.keras.layers.Reshape([-1, 1])(cls_8)
        reshape_cls_16 = tf.keras.layers.Reshape([-1, 1])(cls_16)
        reshape_cls_32 = tf.keras.layers.Reshape([-1, 1])(cls_32)
        reshape_cls_64 = tf.keras.layers.Reshape([-1, 1])(cls_64)
        conf = tf.keras.layers.Concatenate(axis=1)([reshape_cls_8, reshape_cls_16, reshape_cls_32, reshape_cls_64])

        reg_8 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_64)
        reg_16 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_32)
        reg_32 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), strides=(1, 1), padding='same')(upsample_16)
        reg_64 = tf.keras.layers.Conv2D(filters=108, kernel_size=(1, 1), strides=(1, 1), padding='same')(blaze_palm_block_8)
        reshape_reg_8 = tf.keras.layers.Reshape([-1, 18])(reg_8)
        reshape_reg_16 = tf.keras.layers.Reshape([-1, 18])(reg_16)
        reshape_reg_32 = tf.keras.layers.Reshape([-1, 18])(reg_32)
        reshape_reg_64 = tf.keras.layers.Reshape([-1, 18])(reg_64)
        loc = tf.keras.layers.Concatenate(axis=1)([reshape_reg_8, reshape_reg_16, reshape_reg_32, reshape_reg_64])

    x_out = tf.concat([conf, loc], axis=-1)
    model = tf.keras.Model(inputs=x_in, outputs=x_out)
    return model

