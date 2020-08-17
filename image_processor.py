#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from line import Line


INT_MIN = -1e8

class ImageProcessor:

    def mask_ros(self, img, vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            bg = (255,) * img.shape[2]
        else:
            bg = 255

        cv2.fillPoly(mask, vertices, bg)

        masked = cv2.bitwise_and(img, mask)

        return masked

    def hough_lines(self, img, rho=2, theta=np.pi/180, threshold=1, min_line_len=15, max_line_gap=5):
        return cv2.HoughLinesP(img, 1, np.pi/180, 30, maxLineGap=200)
        return cv2.HoughLinesP(img, rho,
                theta,
                threshold,
                np.array([]),
                minLineLength=min_line_len,
                maxLineGap=max_line_gap)

    def get_canny(self, img):
        f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = cv2.GaussianBlur(f, (17, 17), 0)
        f = cv2.Canny(f, threshold1=50, threshold2=80)
        return f

    def get_lines(self, img):
        f = img
        lines = self.hough_lines(f)
        return [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]

    def detect_lanes(self, img):
        lines = self.get_lines(img)
        return self.filter_lines(lines, img.shape)

    def filter_lines(self, lines, shape):
        pos = [l for l in lines if l.slope > 0]
        neg = [l for l in lines if l.slope < 0]

        lb = np.median([l.bias for l in neg]).astype(int)
        ls = np.median([l.slope for l in neg])
        x1, y1 = 0, lb 
        x2, y2 = -np.int32(np.round(lb / ls)), 0
        if y1 <= INT_MIN: y1 = 0
        if x2 <= INT_MIN: x2 = 0
        ll = Line(x1, y1, x2, y2)

        rb = np.median([l.bias for l in pos]).astype(int)
        rs = np.median([l.slope for l in pos])
        x1, y1 = 0, rb 
        x2, y2 = np.int32(np.round((shape[0] - rb) / rs)), shape[0]
        if x2 <= INT_MIN: x2 = 0
        if y1 <= INT_MIN: y1 = 0
        rl = Line(x1, y1, x2, y2)

        return (ll, rl)
    
