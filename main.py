#!/usr/bin/env python3

import cv2
import numpy as np
from display import Display 
from image_processor import ImageProcessor 


def cvtgray(img):
    # if grayscale, add channels
    if len(img.shape) == 2:
        return np.stack((img,) * 3, axis=-1)
    return img

if __name__ == "__main__":

    processor = ImageProcessor()

    frame = cv2.imread("data/img/solidWhiteCurve.jpg")

    H, W, _ = frame.shape

    ros = np.array([[(50, H), (450, 310), (500, 310), (W - 50, H)]], dtype=np.int32)

    display = Display(W, H, pygame=False)

    f2 = np.copy(frame)

    canny = processor.get_canny(frame)
    canny = processor.mask_ros(canny, ros)
    #left, right = processor.detect_lanes(f2)

    #lines = processor.get_lines(canny)

    lines = cv2.HoughLinesP(canny,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

    canny = cvtgray(canny)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(canny,(x1,y1),(x2,y2),(0,255,0),2)


    #for line in lines:

        #line.draw(canny, color=(0, 255, 0))

    #print(left.get_coords())
    #print(right.get_coords())
    #left.draw(f2, color=(255, 255, 0))
    #right.draw(f2, color=(255, 0, 0))
    #print(right.get_length())
    #print(left.get_length())

    display.imshow(canny)
    #display.imshow(f2, window="2")
