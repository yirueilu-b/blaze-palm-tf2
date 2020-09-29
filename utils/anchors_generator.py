import math
import pandas as pd
from utils.anchor_config import Anchor, AnchorsConfig


class AnchorsGenerator:
    def __init__(self, config):
        """
        :param config: Anchor configs see `anchor_config.py`
        """
        self.config = config

    def generate(self, save_path=None):
        """
        :param save_path: path for saving result as csv, default = None
        :return: a list of `Anchor` object see `anchor_config.py` for detail
        """
        result_anchors = []
        # Verify the options.
        if self.config.strides_size != self.config.num_layers:
            print("strides_size and num_layers must be equal.")
            return []

        layer_id = 0
        while layer_id < self.config.strides_size:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []
            # For same strides, we merge the anchors in the same order.
            last_same_stride_layer = layer_id
            while (last_same_stride_layer < self.config.strides_size and self.config.strides[last_same_stride_layer] ==
                   self.config.strides[layer_id]):
                scale = self.config.min_scale + (
                        self.config.max_scale - self.config.min_scale) * 1.0 * last_same_stride_layer / (
                                self.config.strides_size - 1.0)
                if last_same_stride_layer == 0 and self.config.reduce_boxes_in_lowest_layer:
                    # For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for aspect_ratio_id in range(self.config.aspect_ratios_size):
                        aspect_ratios.append(self.config.aspect_ratios[aspect_ratio_id])
                        scales.append(scale)

                    if self.config.interpolated_scale_aspect_ratio > 0.0:
                        scale_next = 1.0 if last_same_stride_layer == self.config.strides_size - 1 else self.config.min_scale + (
                                self.config.max_scale - self.config.min_scale) * 1.0 * (last_same_stride_layer + 1) / (
                                                                                                                self.config.strides_size - 1.0)
                        scales.append(math.sqrt(scale * scale_next))
                        aspect_ratios.append(self.config.interpolated_scale_aspect_ratio)
                last_same_stride_layer += 1
            for i in range(len(aspect_ratios)):
                ratio_sqrts = math.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)

            stride = self.config.strides[layer_id]
            feature_map_height = math.ceil(1.0 * self.config.input_size_height / stride)
            feature_map_width = math.ceil(1.0 * self.config.input_size_width / stride)

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + self.config.anchor_offset_x) * 1.0 / feature_map_width
                        y_center = (y + self.config.anchor_offset_y) * 1.0 / feature_map_height
                        if self.config.fixed_anchor_size:
                            w = 1.0
                            h = 1.0
                        else:
                            w = anchor_width[anchor_id]
                            h = anchor_height[anchor_id]
                        new_anchor = Anchor(x_center, y_center, h, w)
                        result_anchors.append(new_anchor)
            layer_id = last_same_stride_layer
        if save_path:
            df = pd.DataFrame([[anchor.x_center,
                                anchor.y_center,
                                int(anchor.w),
                                int(anchor.h)] for anchor in result_anchors])
            df.to_csv(save_path, index=False)

        return result_anchors


if __name__ == '__main__':
    anchors_config = AnchorsConfig(fixed_anchor_size=False)
    anchors_generator = AnchorsGenerator(anchors_config)
    print(anchors_generator.config)
    test_anchors = anchors_generator.generate("./anchors.csv")
    print(pd.DataFrame([[anchor.x_center, anchor.y_center, anchor.w, anchor.h] for anchor in test_anchors]).head())
