import os

def sizeDifferences(source1, source2):
    # Gets size in bytes and divide 1000000 to get megabytes
    img1Size = (os.stat(source1).st_size) / 1000000
    img2Size = (os.stat(source2).st_size) / 1000000
    print(f"Original Image size: {img1Size:0.2f} megabytes")
    print(f"Compressed Image size: {img2Size:0.2f} megabytes")
    result = (img2Size/img1Size) * 100
    print(f"The Compressed image is {result:0.2f}% the size of the original image")

if __name__ == "__main__":
    import sys

    # check for both input arguments
    if len(sys.argv) < 3:
        print(">> ERR: No Source2 Image provided")
        if len(sys.argv) < 2:
            print(">> ERR: No Source1 Image provided")
        exit(1)

    sizeDifferences(*sys.argv[1:3])
