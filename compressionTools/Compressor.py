import cv2
import time
from printoverride import *
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class _Compressor():
    '''Template class for compression types'''
    def __init__(self, source, visual = False):
        self.compressed = None
        self.img = source
        self.visual = visual
        if type(source) == str:
            self._load(source)


    def _load(self, source): # Override this
        # Hidden as init calls this
        self.img = None


    def save(self, destination): # Override this
        pass


    def compress(self):
        print("Compressing image file")
        if self.compressed is not True: # Protects against double compressing
            self.compressed = True
            self._compress()


    def decompress(self):
        print("Decompressing image file")
        if self.compressed is not False: # Protects against double decompressing
            self.compressed = False
            self._decompress()


    def _compress(self): # Override this
        # Hidden as public compress function calls this
        pass


    def _decompress(self): # Override this
        # Hidden as public decompress function calls this
        pass


