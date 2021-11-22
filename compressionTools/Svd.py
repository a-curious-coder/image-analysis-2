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

class Svd(_Compressor):
    '''
    Uses the svd function (singular value decomposition )
    '''
    def _load(self, source):
        # load in the needed variables
        self.u = np.loadtxt(source + '.U-data')
        self.sigma = np.loadtxt(source + '.sigma-data')
        self.v = np.loadtxt(source + '.V-data')


    def save(self, destination):
        np.savetxt(destination + '.U-data', self.u)
        np.savetxt(destination + '.sigma-data', self.sigma)
        np.savetxt(destination + '.V-data', self.v)
        print("saved image")


    def _compress(self):
        from stegano import lsb
        import json

        #open up image
        img = self.img

        # convert to grey scale and show
        imggrey = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # change type and shape of the image
        imgmat = np.array(list(imggrey), float)
        imgmat.shape = imggrey.shape
        imgmat = np.matrix(imgmat)

        # compute singular value decomposition
        U, sigma, V = np.linalg.svd(imgmat)

        # Writes out data to files to be read by decompression
        self.u = U
        self.sigma = sigma
        self.v = V

        if not self.visual:
            return

        # compute approximation of the image for different ranges and show
        for i in range(1, 4):
            reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

        self.img = reconstimg

        # show the image
        plt.imshow(self.img, cmap='gray')
        title = "n = %s" % i
        plt.title(title)
        plt.show()


    def _decompress(self):
        sigma = self.sigma
        U = self.u
        V = self.v

        # assign the image object
        reconstimg = self.img

        # reconstruct the image
        for i in range(5, 200, 5):
            reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

        self.img = reconstimg

        if not self.visual:
            return
        
        # show the image
        plt.imshow(self.img, cmap='gray')
        title = "n = %s" % i
        plt.title(title)
        plt.show()

