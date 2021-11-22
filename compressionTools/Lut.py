from .Compressor import _Compressor
import cv2
import numpy as np
from printoverride import *

class Lut(_Compressor):
    '''Lossy idea
    Build array of sample pixels up to 256 of them
    Every pixel references a sample and adds to its value slightly
    Sample pixels are full 8bit 3 channel
    Image pixel 8bits for reference + 4bit 3 channel modifier
    At large scale, may reduce image size by half
    '''
    def _load(self, source):
        with open(source, "rb") as file:
            self.height = int.from_bytes(file.read(2), "big")
            self.width = int.from_bytes(file.read(2), "big")
            maxi = int.from_bytes(file.read(2), "big")

            samplePixels = []
            for i in range(maxi):
                col = []
                for c in range(3):
                    col.append(int.from_bytes(file.read(1), "big"))
                samplePixels.append(tuple(col))

            dat = []
            for y in range(self.height):
                d = []
                for x in range(self.width):
                    d.append(int.from_bytes(file.read(1), "big"))
                dat.append(d)

            self.img = dat
            self.samplePixels = samplePixels
            self.maxi = maxi


    def save(self, destination):
        with open(destination, "wb") as file:
            if len(self.samplePixels) < self.maxi:
                self.maxi = len(self.samplePixels)
            
            file.write(self.height.to_bytes(2, "big"))
            file.write(self.width.to_bytes(2, "big"))
            file.write(self.maxi.to_bytes(2, "big"))

            for i in self.samplePixels:
                for c in i:
                    file.write(int(c).to_bytes(1, "big"))

            for y in range(self.height):
                for x in range(self.width):
                    file.write(int(self.img[y][x]).to_bytes(1,"big"))

            #Todo write image pixels


    def _compress(self):
        #Map all pixel in sorted array
        #Calculate regional demand
        #Place sample pixels to supply data fulfilling as much demand as possible
        from collections import defaultdict as defdict
        
        img = self.img
        h, w, d = img.shape
        stream = img.reshape((h * w, d))
        divfact = 7 # Forced point spread
        maxi = 256 # Maximum number of points
        # 256 8bit, 4096 16bit

        colours = defdict(int)
        for i in stream:
            i //= divfact
            i *= divfact
            colours[tuple(i)] += 1
        colour = [i for i in colours.items()]
        colour.sort(key=lambda a:a[1],reverse=True)
        colour = colour[:maxi]
        colour = [i[0] for i in colour]
        
        data = np.zeros((h,w))
        
        for y in range(h):
            for x in range(w):
                v = img[y][x]
                c = (v // divfact) * divfact
                try:
                    r = colour.index(tuple(c))
                    data[y][x] = r
                except:
                    #TODO find closest
                    pass

        self.samplePixels = colour
        self.img = data
        self.height, self.width = h, w
        self.maxi = maxi


    def _decompress(self):
        img = self.img
        pix = self.samplePixels
        h, w = self.height, self.width

        fin = np.zeros((h,w,3))
        for y in range(h):
            for x in range(w):
                d = img[y][x]
                val = np.array(pix[int(d)])
                fin[y][x] = val
        self.img = fin

