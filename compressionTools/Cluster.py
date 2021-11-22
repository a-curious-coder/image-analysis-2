import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.cluster import KMeans

img = cv2.imread("j.jpg")
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.title("RGB Image")
plt.imshow(img)

img = img.reshape((img.shape[1]*img.shape[0], 3))

kmeans = KMeans(n_clusters=4)
s=kmeans.fit(img)
labels = kmeans.labels_
centroid = kmeans.cluster_centers_
print(centroid)
labels =list(labels)
percent = []
tic = time.perf_counter()
for i in range(len(centroid)):
    j= labels.count(i)
    j=j/(len(labels))
    percent.append(j)
toc = time.perf_counter()
print(f"Clustering took: {toc-tic:0.4f} seconds")

plt.subplot(1,2,2)

plt.pie(percent, colors=np.array(centroid/255), labels=np.arange(len(centroid)), autopct='%1.0f%%', pctdistance=1.5, labeldistance=1.2)

plt.show()