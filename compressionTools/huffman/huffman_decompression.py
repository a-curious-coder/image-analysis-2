from PIL import Image
import numpy as np
import cv2

from .Huffman_IO import *

def raw_size(width, height):
    header_size = 2 * 16 # height and width as 16 bit values
    pixels_size = 3 * 8 * width * height # 3 channels, 8 bits per channel
    return (header_size + pixels_size) / 8

def images_equal(file_name_a, file_name_b):
    image_a = Image.open(file_name_a)
    image_b = Image.open(file_name_b)

    diff = ImageChops.difference(image_a, image_b)

    return diff.getbbox() is None

def decode_header(bitstream):
    height = from_binary_list(bitstream.read_bits(16))
    width = from_binary_list(bitstream.read_bits(16))
    return (height, width)

# https://stackoverflow.com/a/759766/3962537
def decode_tree(bitstream):
    flag = bitstream.read_bits(1)[0]
    if flag == 1: # Leaf, read and return symbol
        return from_binary_list(bitstream.read_bits(8))
    left = decode_tree(bitstream)
    right = decode_tree(bitstream)
    return (left, right)

def decode_value(tree, bitstream):
    bit = bitstream.read_bits(1)[0]
    node = tree[bit]
    if type(node) == tuple:
        return decode_value(node, bitstream)
    return node

def decode_pixels(height, width, tree, bitstream):
    pixels = bytearray()
    for i in range(height * width * 3):
        pixels.append(decode_value(tree, bitstream))
    return Image.frombytes('RGB', (width, height), bytes(pixels))

def decompress_image(in_file_name):
    #print('Decompressing "%s" -> "%s"' % (in_file_name, out_file_name))

    print('Reading...')
    stream = InputBitStream(in_file_name)
    print('* Header offset: %d' % stream.bytes_read)
    height, width = decode_header(stream)
    stream.flush() # Ensure next chunk is byte-aligned
    print('* Tree offset: %d' % stream.bytes_read)
    trimmed_tree = decode_tree(stream)
    stream.flush() # Ensure next chunk is byte-aligned
    print('* Pixel offset: %d' % stream.bytes_read)
    image = decode_pixels(height, width, trimmed_tree, stream)
    stream.close()
    print('Read %d bytes.' % stream.bytes_read)

    print('Image size: (height=%d, width=%d)' % (height, width))
    print('Trimmed tree: %s' % str(trimmed_tree))
    #image.save(out_file_name)

    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
