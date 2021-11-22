import compressionTools as ct
import cv2

original = "Images/IC1.bmp"
algo = "dct2"

a = cv2.imread(original)

alg = ct.names[algo]

comp = alg(a)
comp.compress()
comp.save("out")

comp = alg("out")
comp.decompress()
cv2.imwrite("out.bmp",comp.img)

from compareFiles import sizeDifferences

print("Using " + algo + " on the image " + original)
sizeDifferences(original, "out")
