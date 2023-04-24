import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import cv2


img = plt.imread('nature.jpg')

img_width = img.shape[0]
img_height = img.shape[1]
img_type = img.shape[2]

img = img.reshape(img_width*img_height, img_type)
#print(img.shape)
#print(img)
kmeans = KMeans(n_clusters=4).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
#print(labels)
#print(clusters)
img2 = np.zeros_like(img)
#print(img2)

for i in range(len(img2)):
    img2[i] = clusters[labels[i]]

print(img_width)
print(img_height)
print(img_type)


img2 = img2.reshape((img_width, img_height, img_type))
print(img2.shape)
print(img2)
plt.imshow(img2)

plt.show()

#img2 = img2.reshape(img_width*img_height, img_type)
cv2.imwrite("ns.jpg", img2)