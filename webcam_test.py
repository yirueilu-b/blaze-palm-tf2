import cv2
from utils.decoder import decode_prediction
from utils.inference import PalmDetector
from utils.draw import draw_key_points_list, draw_box_list

camera = cv2.VideoCapture(1)
palm_detector = PalmDetector()

while True:
    ret, original_frame = camera.read()
    if not ret: print("Cannot read frame from camera")

    show_frame = original_frame.copy()

    prediction = palm_detector.detect(original_frame)
    bounding_boxes, key_points_list = decode_prediction(prediction, conf_threshold=0.95)
    bounding_boxes, key_points_list = palm_detector.rescale_result(bounding_boxes, key_points_list)

    draw_key_points_list(show_frame, key_points_list)
    draw_box_list(show_frame, bounding_boxes)

    cv2.imshow('Palm Detector Demo', show_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
