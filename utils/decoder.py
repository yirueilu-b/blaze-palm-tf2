import numpy as np
from utils.anchor_config import AnchorsConfig
from utils.anchors_generator import AnchorsGenerator

anchor_config = AnchorsConfig(fixed_anchor_size=False)
anchor_generator = AnchorsGenerator(anchor_config)
anchors = anchor_generator.generate()
ANCHORS = np.array([[anchor.x_center, anchor.y_center, anchor.w, anchor.h] for anchor in anchors])


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def non_max_suppression_fast(boxes, probabilities=None, overlap_threshold=0.3):
    """
    Algorithm to filter bounding box proposals by removing the ones with a too low confidence score
    and with too much overlap.
    Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes: List of proposed bounding boxes
    :param overlap_threshold: the maximum overlap that is allowed
    :return: filtered boxes
    """
    # if there are no boxes, return an empty list
    if boxes.shape[1] == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 2] / [2])  # center x - width/2
    y1 = boxes[:, 1] - (boxes[:, 3] / [2])  # center y - height/2
    x2 = boxes[:, 0] + (boxes[:, 2] / [2])  # center x + width/2
    y2 = boxes[:, 1] + (boxes[:, 3] / [2])  # center y + height/2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = boxes[:, 2] * boxes[:, 3]  # width * height
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probabilities is not None:
        idxs = probabilities

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))
    # return only the bounding boxes that were picked
    return pick


def decode_prediction(prediction, conf_threshold=0.6):
    """
    :param prediction: The raw prediction result from BlazePalm model
    :param conf_threshold: Threshold of confidence on each anchor
    :return: Bounding boxes and key points list. The shape are (#_of_object, 4) and (#_of_object, 7, 2)
    """
    # split conf and loc results
    conf = prediction[:, :, 0][0]
    loc = prediction[:, :, 1:][0]
    # apply sigmoid to conf
    scores = sigmoid(conf)
    # filter with threshold
    loc = loc[scores > conf_threshold]
    conf = conf[scores > conf_threshold]
    # transfer loc back to coordinates (the original one is difference)
    candidate_anchors = ANCHORS[scores > conf_threshold]
    loc[:, :2] += candidate_anchors[:, :2] * 256
    # apply NMS to bounding boxes
    box_ids = non_max_suppression_fast(loc[:, :4], conf)
    # extract final results
    bounding_boxes = loc[box_ids, :4].astype('int')
    key_points_list = loc[box_ids, 4:].reshape(-1, 7, 2)
    # transfer key points back to coordinates (the original one is difference)
    center_offset = candidate_anchors[box_ids, :2] * 256
    for i in range(len(key_points_list)):
        key_points_list[i] = key_points_list[i] + center_offset[i]

    return bounding_boxes, key_points_list
