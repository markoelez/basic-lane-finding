#!usr/bin/env python3

import sys
import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF


class Display:
    def __init__(self, W, H, use_pygame=False):
        if use_pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
            self.surface = pygame.Surface(self.screen.get_size()).convert()
            self.clock = pygame.time.Clock()
            pygame.display.flip()

    def imshow(self, img, window="image"):
        img = self.cvtgray(img)
        cv2.imshow(window, img)
        cv2.waitKey(0)

    def blit(self, img):
        self.clock.tick(30)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            if e.type == pygame.KEYDOWN:
                sys.exit(0)

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

