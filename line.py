#!/usr/bin/env python3

import math
import cv2
import numpy as np


class Line:

    def __init__(self, x1, x2, y1, y2):
        self.x1 = np.float32(x1)
        self.x2 = np.float32(x2)
        self.y1 = np.float32(y1)
        self.y2 = np.float32(y2)


        self.slope = self.get_slope()
        self.bias = self.get_bias()

    def get_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def get_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def get_p1(self):
        return (self.x1, self.y1)

    def get_p2(self):
        return (self.x2, self.y2)

    def draw(self, frame, color=(0, 128, 255), thickness=2):
        cv2.line(frame, self.get_p1(), self.get_p2(), color, thickness)
    
    def get_length(self):
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

