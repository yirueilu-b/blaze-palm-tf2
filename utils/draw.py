import cv2
from utils.encoder import center_to_corner


def draw_box(image, box):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return None


def draw_box_list(image, box_list):
    box_list = center_to_corner(box_list)
    for box in box_list:
        draw_box(image, box.astype('int'))
    return None


def draw_key_points(image, key_points):
    for point in key_points:
        cv2.circle(image, (point[0], point[1]), 5, (255, 0, 0), -1)
    return None


def draw_key_points_list(image, key_points_list):
    for key_points in key_points_list:
        draw_key_points(image, key_points.astype('int'))
    return None
