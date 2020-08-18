#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from line import Line


class ImageProcessor:

    def mask_roi(self, img, vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            bg = (255,) * img.shape[2]
        else:
            bg = 255

        cv2.fillPoly(mask, vertices, bg)
        return cv2.bitwise_and(img, mask)

    def hough_lines(self, img, rho=1, theta=np.pi/180, threshold=10, min_line_len=20, max_line_gap=100):
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
        # separate by gradient
        pos = [l for l in lines if l.slope > 0]
        neg = [l for l in lines if l.slope < 0]

        # remove outliers
        pos = [l for l in pos if l.slope > 0.3]
        neg = [l for l in neg if l.slope < -0.3]

        # get gradients
        rslopes = [l.slope for l in pos]
        lslopes = [l.slope for l in neg]

        # get biases
        rbiases = [l.bias for l in pos]
        lbiases = [l.bias for l in neg]

        # get average m and b
        rslope = np.mean(rslopes[-30:])
        rbias = np.mean(rbiases[-30:])
        lslope = np.mean(lslopes[-30:])
        lbias = np.mean(lbiases[-30:])

        # fix for inf values
        rbias = rbias if rbias != float("-inf") else -1e6

        # get points
        lx1 = int((0.65 * shape[0] - lbias)/lslope)
        ly1 = int(0.65 * shape[0])
        lx2 = int((shape[0] - lbias)/lslope)
        ly2 = int((shape[0]))
        
        rx1 = int((0.65 * shape[0] - rbias)/rslope)
        ry1 = int(0.65 * shape[0])
        rx2 = int((shape[0] - rbias)/rslope)
        ry2 = int(shape[0])

        ll = Line(lx1, ly1, lx2, ly2)
        rl = Line(rx1, ry1, rx2, ry2)

        pts = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]])

        return (ll, rl, pts)

    def draw_poly(self, img, pts, color=(0, 0, 255)):
        pts = pts.reshape((-1, 1, 2))
        return cv2.fillPoly(img, [pts], color)

    def add_channel(self, img):
        # if grayscale, add channels
        if len(img.shape) == 2:
            return np.stack((img,) * 3, axis=-1)
        return img
       
    def overlay(self, img, overlay, a=0.8, b=0.1, l=0):
        return cv2.addWeighted(img, a, overlay, b, l)
    
