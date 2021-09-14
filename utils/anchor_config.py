class AnchorsConfig:
    def __init__(self,
                 num_layers=4,
                 min_scale=0.1171875,
                 max_scale=0.75,
                 input_size_height=256,
                 input_size_width=256,
                 anchor_offset_x=0.5,
                 anchor_offset_y=0.5,
                 strides=(16, 32, 32, 32),
                 aspect_ratios=(1.0,),
                 fixed_anchor_size=False,
                 reduce_boxes_in_lowest_layer=False,
                 interpolated_scale_aspect_ratio=1.0):
        self.num_layers = num_layers
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.input_size_height = input_size_height
        self.input_size_width = input_size_width
        self.anchor_offset_x = anchor_offset_x
        self.anchor_offset_y = anchor_offset_y
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        self.fixed_anchor_size = fixed_anchor_size
        self.strides_size = len(strides)
        self.aspect_ratios_size = len(aspect_ratios)
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio

    def __str__(self):
        info = "%sAnchor Config:\n%s" \
               "num_layers: %s\n" \
               "min_scale: %s\n" \
               "max_scale: %s\n" \
               "input_size_height: %s\n" \
               "input_size_width: %s\n" \
               "anchor_offset_x: %s\n" \
               "anchor_offset_y: %s\n" \
               "strides: %s\n" \
               "aspect_ratios: %s\n" \
               "fixed_anchor_size: %s\n" \
               "strides_size: %s\n" \
               "aspect_ratios_size: %s\n" \
               "reduce_boxes_in_lowest_layer: %s\n" \
               "interpolated_scale_aspect_ratio: %s\n%s" % (
                   "=" * 40 + "\n",
                   "-" * 40 + "\n",
                   self.num_layers,
                   self.min_scale,
                   self.max_scale,
                   self.input_size_height,
                   self.input_size_width,
                   self.anchor_offset_x,
                   self.anchor_offset_y,
                   self.strides,
                   self.aspect_ratios,
                   self.fixed_anchor_size,
                   self.strides_size,
                   self.aspect_ratios_size,
                   self.reduce_boxes_in_lowest_layer,
                   self.interpolated_scale_aspect_ratio,
                   "=" * 40 + "\n")
        return info


class Anchor:
    def __init__(self, x_center, y_center, h, w):
        self.x_center = x_center
        self.y_center = y_center
        self.h = h
        self.w = w
