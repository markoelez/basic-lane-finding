#!/usr/bin/env python3

import cv2
import numpy as np
from display import Display 
from image_processor import ImageProcessor 
from line import Line


processor = ImageProcessor()

def process_frame(frame, ROS):
    edges = processor.get_canny(frame)
    edges = processor.mask_roi(edges, ROS)

    left, right, pts = processor.detect_lanes(edges)

    (x, y) = left.get_intersection(right)

    lines = processor.get_lines(edges)
    edges = processor.add_channel(edges)

    overlay = np.zeros_like(frame)

    left.clip_to_point(x, y)
    right.clip_to_point(x, y)
    
    # draw lanes and fill poly
    overlay = processor.draw_poly(overlay, pts)
    left.draw(overlay, color=(0, 255, 0))
    right.draw(overlay, color=(0, 255, 0))

    f = processor.weighted_img(frame, overlay)

    return f 

def handle_img(img_path):

    frame = cv2.imread(img_path)

    H, W, _ = frame.shape
    ROS = np.array([[(50, H), (450, 310), (500, 310), (W - 50, H)]], dtype=np.int32)

    frame = process_frame(frame, ROS)

    display = Display(W, H)

    display.imshow(frame)

def handle_vid(vid_path):
    
    cap = cv2.VideoCapture(vid_path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ROS = np.array([[(50, H), (450, 310), (500, 310), (W - 50, H)]], dtype=np.int32)

    display = Display(W, H, use_pygame=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (W, H))

        frame = process_frame(frame, ROS)

        display.blit(frame)


if __name__ == "__main__":
    
    #handle_img("data/img/solidWhiteCurve.jpg")

    handle_vid("data/vid/solidWhiteRight.mp4")

