#!usr/bin/env python3

import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF


class Display:
    def __init__(self, W, H, pygame=True):
        if pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
            self.surface = pygame.Surface(self.screen.get_size()).convert()

    def imshow(self, img, window="image"):
        img = self.cvtgray(img)
        cv2.imshow(window, img)
        cv2.waitKey(0)

    def blit(self, img):
        for _  in pygame.event.get():
            pass

        img = self.cvtgray(img)

        # bgr -> rgb
        img = img.swapaxes(0, 1)[..., ::-1]

        pygame.surfarray.blit_array(self.surface, img)
        self.screen.blit(self.surface, (0, 0))

        pygame.display.flip()
    
    def cvtgray(self, img):
        # if grayscale, add channels
        if len(img.shape) == 2:
            return np.stack((img,) * 3, axis=-1)
        return img