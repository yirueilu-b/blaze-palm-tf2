import numpy as np


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
    """
    boxes1 and boxes2 should be in the same scale and corner format
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def center_to_corner(center_boxes):
    """
    (cx, cy, w, h) to (y1, x1, y2, x2)
    """
    corner_boxes = np.zeros(center_boxes.shape)
    corner_boxes[:, :2] = center_boxes[:, :2] - center_boxes[:, 2:] / 2
    corner_boxes[:, 2:] = corner_boxes[:, :2] + center_boxes[:, 2:]
    return corner_boxes


def corner_to_center(corner_boxes):
    """
    (x1, y1, x2, y2) to (cx, cy, w, h)
    """
    center_boxes = np.zeros(corner_boxes.shape)
    center_boxes[:, :2] = (corner_boxes[:, :2] + corner_boxes[:, 2:]) / 2
    center_boxes[:, 2:] = corner_boxes[:, 2:] - corner_boxes[:, :2]
    return center_boxes


def encode(matches, anchors):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matches: (tensor) Coords of ground truth for each prior in point-form Shape: [num_priors, 18].
        anchors: (tensor) Prior boxes in center-offset form Shape: [num_priors, 18].
    Return:
        encoded location (tensor), Shape: [num_priors, 18]
    """

    # dist b/t match center and prior's center
    # (x_min, y_min, x_max, y_max) to (center)
    g_cxcy = corner_to_center(matches[:, :4])[:, :2] - anchors[:, :2]
    g_wh = (matches[:, 2:4] - matches[:, :2])
    g_kp = matches[:, 4:].reshape(-1, 7, 2) - anchors[:, :2].reshape(-1, 1, 2)

    return np.concatenate([g_cxcy, g_wh, g_kp.reshape(-1, 14)], 1)  # [num_priors, 18]


def match(annotation, anchors, match_threshold=0.5):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        match_threshold: (float) The overlap threshold used when mathing boxes. default=0.5
        annotation: (list) Ground truth labels and ground truth boxes, Shape: [(num_obj, 1), (num_obj, num_coordinate)].
        anchors: (np.array) Anchor boxes Shape: [n_priors,4].
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    ious = iou(annotation[1][:, :4], center_to_corner(anchors))
    best_prior_overlap, best_prior_idx = np.max(ious, 1), np.argmax(ious, 1)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = np.max(ious, 0), np.argmax(ious, 0)
    best_truth_idx.squeeze()
    best_truth_overlap.squeeze()
    best_prior_idx.squeeze()
    best_prior_overlap.squeeze()
    best_truth_overlap[best_prior_idx] = 2
    for j in range(len(best_prior_idx)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = annotation[1][best_truth_idx]

    # Shape: [num_priors]
    conf = annotation[0][best_truth_idx]
    # label as background
    conf[best_truth_overlap < match_threshold] = 0
    loc = encode(matches, anchors)

    # [num_priors,4] encoded offsets to learn
    # loc_truth[batch_id] = encoded_loc
    # [num_priors] top class label for each prior
    # conf_truth[batch_id] = conf
    return conf, loc
