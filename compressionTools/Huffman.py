from .Compressor import _Compressor
import cv2
import time
from printoverride import *
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Huffman(_Compressor):
    def _load(self, source):
        self.img = source


    def save(self, destination):

        cv2.imwrite(destination, self.img)
        print("saved image")


    def _compress(self):
        from .huffman.huffman_compression import compress_image
        #from huffman.Huffman_IO import *

        compress_image(self.img, 'huffman.txt')   # TODO sort out this crap, get it to save and take in text docs


    def _decompress(self):
        from .huffman.huffman_decompression import decompress_image


        self.img = decompress_image('huffman.txt')
