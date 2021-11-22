import cv2
import numpy as np

'''Compares the two source images provided'''

def main(source1, source2, *args):
    """
    Takes in the two images and then runs them through the compare function
    """
    print(source1, source2, args)

    img1 = cv2.imread(source1)
    if img1 is None:
            raise Exception("Image could not be loaded")

    img2 = cv2.imread(source2)
    if img2 is None:
        raise Exception("Image could not be loaded")

    if args.count("v"):
        visualcompare(img1, img2)

    if args.count("s"):
        diff = generateDeltaImage(img1, img2)
        cv2.imwrite("deltaImage.bmp", diff)
        #Hard coded for now

    sizeDifferences(source1, source2)
    results = statisticalcompare(img1, img2)
    print(results)


def generateDeltaImage(img1, img2):
    if img1.shape != img2.shape:
        raise Exception("Images not the same size")

    # Store the differences to a 'difference' image
    d1 = cv2.subtract(img1, img2) #Delta positive
    d2 = cv2.subtract(img2, img1) #Delta negative
    difference = cv2.add(d1, d2) #Unsigned delta

    return difference


def visualcompare(img1, img2):
    """
    Takes in two images, reshapes them
    """
    difference = generateDeltaImage(img1, img2)

    #difference = cv2.resize(difference, (0,0), None, 2, 2, cv2.INTER_NEAREST)
    #Upscale to 200% using nearest-neighbour (No blur)

    # Stacks images the before and after image enhancement
    comparison = np.hstack((img1, img2, difference))

    # Stacks the comparison image and a bigger image representing the differences between the two
    #ver = np.vstack((comparison, difference))

    print("[*] Showing images together with difference")

    cv2.imshow('Comparison of images', comparison)
    cv2.waitKey(0)


def statisticalcompare(original, denoised):
    if original.shape != denoised.shape:
        raise Exception("Images not the same size")

    sse = 0
    greatestSquareError = 0

    for rowOriginal, rowDenoised in zip(original, denoised):
        for pixelOriginal, pixelDenoised in zip(rowOriginal, rowDenoised):
            #Assuming greyscale only

            error = int(pixelOriginal[0]) - int(pixelDenoised[0])
            error **= 2

            if error > greatestSquareError:
                greatestSquareError = error

            sse += error

    mse = sse / original.size

    return {"sse":sse, "mse":mse, "gse":greatestSquareError} #More to be added

def sizeDifferences(source1, source2):
    # Gets size in bytes and divide 1000000 to get megabytes
    img1Size = (os.stat(source1).st_size) / 1000000
    img2Size = (os.stat(source2).st_size) / 1000000
    print(f"Original Image size:\t\t{img1Size:0.2f} megabytes")
    print(f"Decompressed Image size:\t{img2Size:0.2f} megabytes")
    result = 100 - ((img2Size/img1Size) * 100)
    print(f"The decompressed image is {result:0.2f} % smaller than the original image")

if __name__ == "__main__":
    import sys

    # check for both input arguments
    if len(sys.argv) < 3:
        print(">> ERR: No Source2 Image provided")
        if len(sys.argv) < 2:
            print(">> ERR: No Source1 Image provided")
        exit(1)

    main(*sys.argv[1:])
