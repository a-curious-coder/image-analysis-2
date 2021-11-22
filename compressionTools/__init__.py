from .Compressor import *
from .Dct import *
from .Dct2 import *
from .Huffman import *
from .Svd import *
from .Lut import *

names = {"dct": Dct, "dct2": Dct2, "svd": Svd,
         "huf": Huffman, "lut": Lut}
