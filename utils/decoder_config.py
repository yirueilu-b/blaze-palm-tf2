class DecoderConfig:
    def __init__(self,
                 num_classes=1,
                 num_boxes=2944,
                 num_coords=18,
                 box_coord_offset=0,
                 keypoint_coord_offset=4,
                 num_keypoints=7,
                 num_values_per_key_point=2,
                 sigmoid_score=True,
                 score_clipping_thresh=100.0,
                 x_scale=256.0,
                 y_scale=256.0,
                 h_scale=256.0,
                 w_scale=256.0,
                 min_score_thresh=0.7):
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.num_coords = num_coords
        self.box_coord_offset = box_coord_offset
        self.keypoint_coord_offset = keypoint_coord_offset
        self.num_keypoints = num_keypoints
        self.num_values_per_key_point = num_values_per_key_point
        self.sigmoid_score = sigmoid_score
        self.score_clipping_thresh = score_clipping_thresh
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.h_scale = h_scale
        self.w_scale = w_scale
        self.min_score_thresh = min_score_thresh

    def __str__(self):
        info = "Decode Config:\n===============\n" \
               "\tnum_classes: %s\n" \
               "\tnum_boxes: %s\n" \
               "\tnum_coords: %s\n" \
               "\tbox_coord_offset: %s\n" \
               "\tkeypoint_coord_offset: %s\n" \
               "\tnum_keypoints: %s\n" \
               "\tnum_values_per_key_point: %s\n" \
               "\tsigmoid_score: %s\n" \
               "\tscore_clipping_thresh: %s\n" \
               "\tx_scale: %s\n" \
               "\ty_scale: %s\n" \
               "\th_scale: %s\n" \
               "\tw_scale: %s\n" \
               "\tmin_score_thresh: %s\n" % (
                   self.num_classes,
                   self.num_boxes,
                   self.num_coords,
                   self.box_coord_offset,
                   self.keypoint_coord_offset,
                   self.num_keypoints,
                   self.num_values_per_key_point,
                   self.sigmoid_score,
                   self.score_clipping_thresh,
                   self.x_scale,
                   self.y_scale,
                   self.h_scale,
                   self.w_scale,
                   self.min_score_thresh)
        return info
