import tensorflow as tf


def binary_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.
    alpha = .25
    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = tf.keras.backend.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(tf.keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = tf.keras.backend.ones_like(y_true) * alpha
    alpha_t = tf.where(tf.keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -tf.math.log(p_t)
    weight = alpha_t * tf.math.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    #     loss = tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=1))
    return loss


def smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.
    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


class SSDLoss:
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]

        classification_loss = binary_focal_loss_fixed(y_true[:, :, 0], y_pred[:, :, 0])
        localization_loss = smooth_L1_loss(y_true[:, :, 1:], y_pred[:, :, 1:])

        negatives = y_true[:, :, 0]
        negatives = -(negatives - 1)
        positives = y_true[:, :, 0]
        n_positive = tf.reduce_sum(positives)

        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        n_negative_keep = tf.minimum(tf.cast(tf.maximum(self.neg_pos_ratio * n_positive,
                                                        self.n_neg_min), tf.int32),
                                     n_neg_losses)

        def f1():
            return tf.zeros([batch_size])

        def f2():
            neg_class_loss_all_1d = tf.reshape(neg_class_loss_all, [-1])
            values, indices = tf.nn.top_k(tf.cast(neg_class_loss_all_1d, tf.int32),
                                          k=tf.cast(n_negative_keep, tf.int32), sorted=False)
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1d))
            negatives_keep = tf.reshape(negatives_keep, [batch_size, n_boxes])
            neg_loss = tf.reduce_sum(classification_loss * tf.cast(negatives_keep, tf.float32), axis=-1)
            return neg_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, tf.cast(n_positive, tf.float32))
        # total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
