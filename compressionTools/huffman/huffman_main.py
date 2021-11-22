from PIL import ImageChops
import cv2

from huffman_compression import compress_image
from huffman_decompression import decompress_image


# def raw_size(width, height):
#     header_size = 2 * 16 # height and width as 16 bit values
#     pixels_size = 3 * 8 * width * height # 3 channels, 8 bits per channel
#     return (header_size + pixels_size) / 8
#
# def images_equal(file_name_a, file_name_b):
#     image_a = Image.open(file_name_a)
#     image_b = Image.open(file_name_b)
#
#     diff = ImageChops.difference(image_a, image_b)
#
#     return diff.getbbox() is None

if __name__ == '__main__':
    img = cv2.imread('IC1.bmp')

    compress_image(img, 'answer.txt')

    img = decompress_image('answer.txt')

    cv2.imwrite("out.bmp", img)
