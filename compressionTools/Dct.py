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


# https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/transforms.html#discrete-cosine-transform-dct
class Dct(_Compressor):
    '''Discrete cosine transform (JPEG style)'''
    ''' Lossless Compression & Decompression '''

    def _load(self, source):
        self.img = source


    def save(self, destination):
        cv2.imwrite(destination, self.img)
        print("saved image")


    def _compress(self):
        # Splits image into blocks of 8x8
        blockSize = 8
        original = self.img
        height, width, t = self.img.shape
        
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        allBlocks = []

        # Height / Width of image in regards to blocks that fit in image.
        h ,w = np.array(grayscale.shape[:2]) / blockSize * blockSize
        # https://stackoverflow.com/questions/20733156/slice-indices-must-be-integers-or-none-or-have-index-method
        # Height and width measured in blocks need to be integers
        h = int(h)
        w = int(w)
        # 2D array of this current image grayscale in pixels
        grayscale = grayscale[:h, :w]
        # Number of blocks Vertical/Horizontal
        blocksV = int(h/blockSize)
        blocksH = int(w/blockSize)
        # All the values are set to zero for this block.
        vis0 = np.zeros((h ,w), np.float32)
        # vis0 stores the grayscale image
        vis0[:h, :w] = grayscale
        transform = np.zeros((h ,w), np.float32)
        ##### Applying DCT to all pixel blocks from image to get DCT coefficients #####
        for row in range(blocksV):
                for col in range(blocksH):
                        # Gets block from image, subtracts 128 from every value in the block and applies Discrete Cosine Transform
                        currentblock = cv2.dct(vis0[ row*blockSize : (row+1)*blockSize ,col*blockSize : (col+1)*blockSize]-np.ones((blockSize, blockSize))*128)
                        # Stores currentBlock to transform.
                        transform[ row*blockSize : (row+1)*blockSize, col*blockSize : (col+1)*blockSize ] = currentblock
                        allBlocks.append(currentblock)
        # Prints the top row of values within the first block. 
        print(f"Values of one row in one block before Quantisation:\n\t{allBlocks[0][0]}")
        # Prints all pixel values within the block after DCT is applied
        # print(allBlocks[0])
        print(f"There are {len(allBlocks)} blocks for this image.")
        print(f"Each block is {allBlocks[0].shape} in size")
        # Converts array to image after transform with dct
        transformImage = Image.fromarray(transform)



        ##### Apply quantisation - takes between 1.5 - 2 minutes #####
        tic = time.perf_counter()
        quantised = np.zeros((h ,w), np.float32)
        selectedQMatrix = self.selectQMatrix("Q50")
        for ndct in allBlocks:
            for i in range(blockSize):
                for j in range(blockSize):
                    ndct[i,j] = np.around(ndct[i,j]/selectedQMatrix[i,j])
                    quantised[i*blockSize : (i+1)*blockSize, j*blockSize : (j+1)*blockSize] = ndct[i, j]
        print(f"Values after Quantisation:\t{allBlocks[0][0]}")
        toc = time.perf_counter()
        print(f"Quantisation took: {toc-tic:0.4f} seconds")
        quantisedImage = Image.fromarray(quantised)

        ##### Apply inverse Discrete Cosine Transform to get image back from these new values in each block #####
        invertedBlocks = []
        for blockToInvert in allBlocks:
            curriDCT = cv2.idct(blockToInvert)
            invertedBlocks.append(curriDCT+np.ones((blockSize, blockSize))*128)
        invertedBlocks[0][0]
    
        ##### Stitches all inverted blocks back together to reconstruct image in compressed state #####
        row = 0
        rowNcol = []
        for j in range(int(width/blockSize),len(invertedBlocks)+1,int(width/blockSize)):
            if row == 0:
                print(j)
            rowNcol.append(np.hstack((invertedBlocks[row:j])))
            row = j
        compressed = np.vstack((rowNcol))
        # print(compressed.shape)
        # print(compressed)

        ###### Point represents where on the image the 8x8 pixel block is retreived from #####
        # point = plt.ginput(1)
        # Hardcoded point in image
        # point = [(2064, 1630)]
        # Captures the block using the point user clicked on
        # block = np.floor(np.array(point)/blockSize) # First component is col, second component is row
        # print(block)
        # col = block[0,0]
        # row = block[0,1]
        # plt.plot([blockSize*col,blockSize*col+blockSize,blockSize*col+blockSize,blockSize*col,blockSize*col],
        # [blockSize*row,blockSize*row,blockSize*row+blockSize,blockSize*row+blockSize,blockSize*row])
        # plt.axis([0,w,h,0])

        ###### Shows a 8x8 pixel block selected from the image #####
        # print("[*]\tShows block user selected before and after DCT")
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # selectedImg=grayscale[int(row*blockSize):int((row+1)*blockSize), int(col*blockSize):int((col+1)*blockSize)]
        # N255 = Normalize(0,255) # Normalization object, used by imshow()
        # plt.imshow(selectedImg, cmap="gray", norm=N255, interpolation='nearest')
        # plt.title(f"Before DCT")

        ###### Shows the Discrete Cosine Transform on that particular block #####
        # plt.subplot(1, 2, 2)
        # # print(len(eachTransform))
        # newtrans = transform
        # selectedTrans=newtrans[int(row*blockSize) : int((row+1)*blockSize) , int(col*blockSize) : int((col+1)*blockSize) ]
        # plt.imshow(selectedTrans, cmap=cm.jet, interpolation='nearest')
        # plt.colorbar(shrink=0.5)
        # plt.title("After DCT")
        
        ##### Converts BGR image to RGB #####
        # b, g, r = cv2.split(original)
        # original = cv2.merge([r,g,b])

        ###### COMPARE GRAYSCALE WITH COMPRESSED GRAYSCALE #####
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(grayscale, cmap=cmap)
        # plt.axis('off')
        # plt.title("DCT on entire image")
        # plt.subplot(1,2,2)
        # plt.imshow(compressed, cmap=cmap)
        # plt.axis('off')
        # plt.title("DCT and Quantisation on entire image")
        # Compares image at various stages of compression process.
        ###### COMPARE ORIGINAL WITH COMPRESSED GRAYSCALE #####
        cmap = 'gray'
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(grayscale, cmap = cmap)
        plt.axis('off')
        plt.title("Original image")
        plt.subplot(1,2,2)
        plt.imshow(compressed, cmap = cmap)
        plt.axis('off')
        plt.title("Compressed image")
        # Fullscreen
        # print(grayscale)
        self.img = compressed
        plt.show()
        


    def _decompress(self):
        compressed = self.img
        originalCompressed = self.img

        print(type(self.img))
        blockSize = 8
        allCBlocks = []

        # Height / Width of image in regards to blocks that fit in image.
        h ,w = np.array(compressed.shape[:2]) / blockSize * blockSize
        # https://stackoverflow.com/questions/20733156/slice-indices-must-be-integers-or-none-or-have-index-method
        # Height and width measured in blocks need to be integers
        h = int(h)
        w = int(w)
        # 2D array of this current image grayscale in pixels
        grayscale = grayscale[:h, :w]
        # Number of blocks Vertical/Horizontal
        blocksV = int(h/blockSize)
        blocksH = int(w/blockSize)
        # All the values are set to zero for this block.
        vis0 = np.zeros((h ,w), np.float32)
        # vis0 stores the grayscale image
        vis0[:h, :w] = grayscale
        compressedImageBlocks = np.zeros((h ,w), np.float32)
        ##### Applying DCT to all pixel blocks from image to get DCT coefficients #####
        for row in range(blocksV):
                for col in range(blocksH):
                        # Gets block from image, subtracts 128 from every value in the block and applies Discrete Cosine Transform
                        currentblock = vis0[ row*blockSize : (row+1)*blockSize ,col*blockSize : (col+1)*blockSize]
                        # Stores currentBlock to transform.
                        compressedImageBlocks[ row*blockSize : (row+1)*blockSize, col*blockSize : (col+1)*blockSize ] = currentblock
                        allCBlocks.append(currentblock)

        tic = time.perf_counter()
        dequantised = np.zeros((h ,w), np.float32)
        selectedQMatrix = self.selectQMatrix("Q50")
        for ndct in allBlocks:
            for i in range(blockSize):
                for j in range(blockSize):
                    ndct[i,j] = np.around(ndct[i,j]*selectedQMatrix[i,j])
                    dequantised[i*blockSize : (i+1)*blockSize, j*blockSize : (j+1)*blockSize] = ndct[i, j]
        print(f"Values after Deuqantisation:\t{allCBlocks[0][0]}")
        toc = time.perf_counter()
        print(f"Dequantisation took: {toc-tic:0.4f} seconds")
        dequantisedImage = Image.fromarray(dequantised)

        ##### Apply inverse Discrete Cosine Transform to get image back from these new values in each block #####
        invertedBlocks = []
        for blockToInvert in allCBlocks:
            curriDCT = cv2.idct(blockToInvert)
            invertedBlocks.append(curriDCT+np.ones((blockSize, blockSize))*128)
        invertedBlocks[0][0]

        row = 0
        rowNcol = []
        for j in range(int(width/blockSize),len(invertedBlocks)+1,int(width/blockSize)):
            if row == 0:
                print(j)
            rowNcol.append(np.hstack((invertedBlocks[row:j])))
            row = j
        decompressed = np.vstack((rowNcol))


        cmap = 'gray'
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(originalCompressed, cmap = cmap)
        plt.axis('off')
        plt.title("Original Compressed Image")
        plt.subplot(1,2,2)
        plt.imshow(decompressed, cmap = cmap)
        plt.axis('off')
        plt.title("Decompressed image")
        # Fullscreen
        self.img = decompressed
        plt.show()



    # Quantisation matrices
    def selectQMatrix(self, qName):
        Q10 = np.array([[80,60,50,80,120,200,255,255],
                    [55,60,70,95,130,255,255,255],
                    [70,65,80,120,200,255,255,255],
                    [70,85,110,145,255,255,255,255],
                    [90,110,185,255,255,255,255,255],
                    [120,175,255,255,255,255,255,255],
                    [245,255,255,255,255,255,255,255],
                    [255,255,255,255,255,255,255,255]])

        Q50 = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,130,99]])

        Q90 = np.array([[3,2,2,3,5,8,10,12],
                        [2,2,3,4,5,12,12,11],
                        [3,3,3,5,8,11,14,11],
                        [3,3,4,6,10,17,16,12],
                        [4,4,7,11,14,22,21,15],
                        [5,7,11,13,16,12,23,18],
                        [10,13,16,17,21,24,24,21],
                        [14,18,19,20,22,20,20,20]])
        if qName == "Q10":
            return Q10
        elif qName == "Q50":
            return Q50
        elif qName == "Q90":
            return Q90
        else:
            return np.ones((8,8)) #it suppose to return original image back
