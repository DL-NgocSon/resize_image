from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = plt.imread("nature.jpg")

# print(img.shape)

img_height = img.shape[0]
img_width = img.shape[1]
img_type = img.shape[2]

# print(img_height)
# print(img_width)


img = img.reshape(img_height*img_width, img_type)
kmeans = KMeans(n_clusters=4).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

# print(labels)
# print(clusters)

img_2 = np.zeros((img_height, img_width, img_type), dtype=np.uint8)

# plt.imshow(img_2)
# plt.show()

index = 0
for i in range(img_height):
    for j in range(img_width):
        label_of_pixel = labels[index]
        img_2[i][j] = clusters[label_of_pixel]
        index = index + 1

plt.imshow(img_2)
plt.show()
#cv2.imwrite('ns.jpg', img_2)