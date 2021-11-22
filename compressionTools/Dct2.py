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

class Dct2(_Compressor):
    def _load(self, source):
        with open(source, "rb") as file:
            blockSize = int.from_bytes(file.read(2), "big")
            blocksX = int.from_bytes(file.read(2), "big")
            blocksY = int.from_bytes(file.read(2), "big")

            data = []
            for i in range(blocksX * blocksY):
                rg = []
                for rgb in range(3):
                    rows = []
                    for row in range(blockSize//2):
                        col = []
                        for cell in range(blockSize//2):
                            col.append(int.from_bytes(file.read(2), "little"))
                        rows.append(col)
                    rg.append(rows)
                data.append(rg)

            data = np.array(data)
            data = np.int16(data)
            self.img = data
            self.blockSize = blockSize
            self.blocksX = blocksX
            self.blocksY = blocksY


    def save(self, destination):
        with open(destination, "wb") as file:
            file.write(self.blockSize.to_bytes(2, "big"))
            file.write(self.blocksX.to_bytes(2, "big"))
            file.write(self.blocksY.to_bytes(2, "big"))

            for block in self.img:
                for rgb in block:
                    for x in rgb:
                        for y in x:
                            file.write(y.tobytes())
                            # np gens little endian


    def _compress(self):
        # HLS or HSV would be better than rgb here
        # It would remove risk of colour tearing
        # Also I'm starting to think taking 3/4 of the dct data was too much
        # You can see the block boundaries in the image
        blockSize = 8
        img = self.img
        height, width, t = img.shape

        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]

        def getChunk(x, y, b, dat):
            x *= b
            y *= b
            dat = dat[y: y + b, x: x + b]
            dat = np.float32(dat)
            b = b // 2
            dat = cv2.dct(dat)[:b, :b]
            #Remove all but a quarter of the dct data
            dat = np.int16(dat) #int8 wont work :(
            return dat

        data = []
        for j in range(height//blockSize):
            for i in range(width//blockSize):
                data.append([getChunk(i,j,blockSize,r),
                             getChunk(i,j,blockSize,g),
                             getChunk(i,j,blockSize,b)])

        self.img = data
        self.blockSize = blockSize
        self.blocksX = i+1
        self.blocksY = j+1


    def _decompress(self):
        img = self.img
        print(type(img))
        blockSize = self.blockSize
        blocksX = self.blocksX
        blocksY = self.blocksY

        def getdat(b, chunk):
            t = np.zeros((b,b))
            t[:chunk.shape[0], :chunk.shape[1]] = chunk
            chunk = t
            chunk = np.float32(chunk)
            return cv2.idct(chunk)

        fin = np.zeros((blockSize * blocksY, blockSize * blocksX, 3))
        for j in range(blocksY):
            for i in range(blocksX):
                block = img[j * blocksX + i]
                t = np.zeros((blockSize,blockSize))

                r = getdat(blockSize, block[0])
                g = getdat(blockSize, block[1])
                b = getdat(blockSize, block[2])

                block = np.stack((r,g,b),axis=2)

                x = i * blockSize
                y = j * blockSize
                fin[y: y + blockSize,x: x + blockSize] = block

        self.img = fin
