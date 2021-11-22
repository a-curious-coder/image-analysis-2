import cv2
import os
from printoverride import *
import compressionTools as ct


'''Compress or decompress the source image and saves the product at the destination'''

def main(source, destination, tool, **nargs):
    #os.system('cls' if os.name == 'nt' else 'clear')
    print("Image Compression / Decompression Program")
    print(f'Source: "{source}"', 1)
    print(f'Destination: "{destination}"', 1)
    print(f'Tool: "{tool}"', 1)
    print(f'Arguments: "{nargs}', 1)

    compress = not nargs.get("d")
    compressor = keymap[tool]

    if compress:
        img = cv2.imread(source)
        if img is None:
            print("Image could not be loaded")
            exit(1)

        compressor = compressor(img, nargs.get("v"))
        compressor.compress()

        compressor.save(destination)
        # The compressed image does not need to be in a common file format
        # Have the compressor save it as pure binary if you want

    else:
        compressor = compressor(source, nargs.get("v"))
        compressor.decompress()
        cv2.imwrite(destination, compressor.img)
        # Decompression should return an np array that cv2 will understand
        # as an image

##    if "-c" in args or "-compare" in args:
##        # Calculate input size against output size
##        # Maybe include size calc in compressor?
##        # Locally decompress
##        # Display input next to decompressed output
##        # Show delta between image like from Cw-1

keymap = ct.names

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Input file path')
    parser.add_argument('destination', help='Output file path')
    parser.add_argument('tool', choices=keymap.keys(), help='Compression tool')
    boolinp = {"nargs":'?', "type":bool, "const":True, "default":False}
    parser.add_argument('-d', '-de', '-decompress', help='Decompress tag', **boolinp)
    parser.add_argument('-v', '-visual', help='Visualisation tag', **boolinp)
    args = parser.parse_args()

    main(**args.__dict__)
