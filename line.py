#!/usr/bin/env python3

import math
import cv2
import numpy as np


class Line:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = np.float32(x1)
        self.x2 = np.float32(x2)
        self.y1 = np.float32(y1)
        self.y2 = np.float32(y2)

        self.slope = self.get_slope()
        self.bias = self.get_bias()
        self.length = self.get_length()
        self.coords = self.get_coords()

    def get_slope(self):
        if self.x2 - self.x1 == 0: return float("inf")
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def get_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def get_p1(self):
        return (self.x1, self.y1)

    def get_p2(self):
        return (self.x2, self.y2)

    def draw(self, frame, color=(0, 128, 255), thickness=3):
        cv2.line(frame, self.get_p1(), self.get_p2(), color, thickness)
    
    def get_length(self):
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    def get_intersection(self, other):
        m1, m2 = self.get_slope(), other.get_slope()
        b1, b2 = self.get_bias(), other.get_bias()
        # check not parallel
        assert m1 != m2
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return (x, y)

    def clip_to_point(self, x, y):
        if self.y1 < y:
            self.x1 = x
            self.y1 = y
        elif self.y2 < y:
            self.x2 = x
            self.y2 = y

