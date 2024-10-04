from ctypes import sizeof
import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage
import cv2 as cv
from PIL import Image

image_file1 = "images/image1.jpg"
image_file2 = "images/image2.jpg"
image_file3 = "images/image3.jpg"
image_file4 = "images/image4.jpg"
image_file5 = "images/image5.jpg"

img1 = np.array(Image.open(image_file1))
img2 = np.array(Image.open(image_file2))
img3 = np.array(Image.open(image_file3))
img4 = np.array(Image.open(image_file4))
img5 = np.array(Image.open(image_file5))


def norm(img):
    r_ = img[:,:,0]/255
    g_ = img[:,:,1]/255
    b_ = img[:,:,2]/255

    r = r_/(r_+g_+b_)
    g = g_/(r_+g_+b_)
    b = b_/(r_+g_+b_)

    return r,g,b

def get_CIVE_band(img):
    r,g,b = norm(img)
    img = cv.GaussianBlur(img,(35,35),0)
    CIVE_band = 0.441*img[:,:,0] - 0.881*img[:,:,1] + 0.385*img[:,:,2] + 18.787
    normalized_CIVE_band = (((CIVE_band+abs(CIVE_band.min()))/CIVE_band.max())).astype(np.uint8)
    return CIVE_band

def apply_mask(img, vegetation_index_band):
    ret, otsu = cv.threshold(vegetation_index_band,2.78,1,cv.THRESH_BINARY)
    masked_img = cv.bitwise_and(img,img,mask = otsu.astype(np.uint8))
    return masked_img

fig, axes = plt.subplots(nrows=5, ncols=2, figsize = (9,7))

#-------------img1---------------

ax = plt.subplot(5, 3, 1)
ax.set_title("img1")
plt.imshow(img1)
print(img1.size)

CIVE_band = get_CIVE_band(img1)
ax = plt.subplot(5, 3, 2)
ax.set_title("CIVE band")
print(CIVE_band.size)
plt.imshow(CIVE_band)


masked_img = apply_mask(img1, CIVE_band)


ax = plt.subplot(5, 3, 3)
ax.set_title("CIVE+Otsu")
plt.imshow(masked_img)

#-------------img2---------------

ax = plt.subplot(5, 3, 4)
ax.set_title("img2")
plt.imshow(img2)

CIVE_band = get_CIVE_band(img2)
ax = plt.subplot(5, 3, 5)
ax.set_title("CIVE band")
plt.imshow(CIVE_band)

masked_img = apply_mask(img2, CIVE_band)
ax = plt.subplot(5, 3, 6)
ax.set_title("CIVE+Otsu")
plt.imshow(masked_img)

#-------------img3---------------

ax = plt.subplot(5, 3, 7)
ax.set_title("img3")
plt.imshow(img3)

CIVE_band = get_CIVE_band(img3)
ax = plt.subplot(5, 3, 8)
ax.set_title("CIVE band")
plt.imshow(CIVE_band)

masked_img = apply_mask(img3, CIVE_band)
ax = plt.subplot(5, 3, 9)
ax.set_title("CIVE+Otsu")
plt.imshow(masked_img)

#-------------img4---------------

ax = plt.subplot(5, 3, 10)
ax.set_title("img4")
plt.imshow(img4)

CIVE_band = get_CIVE_band(img4)
ax = plt.subplot(5, 3, 11)
ax.set_title("CIVE band")
plt.imshow(CIVE_band)

masked_img = apply_mask(img4, CIVE_band)
ax = plt.subplot(5, 3, 12)
ax.set_title("CIVE+Otsu")
plt.imshow(masked_img)

#-------------img5---------------

ax = plt.subplot(5, 3, 13)
ax.set_title("img5")
plt.imshow(img5)

CIVE_band = get_CIVE_band(img5)
ax = plt.subplot(5, 3, 14)
ax.set_title("CIVE band")
plt.imshow(CIVE_band)

masked_img = apply_mask(img5, CIVE_band)
ax = plt.subplot(5, 3, 15)
ax.set_title("CIVE+Otsu")
plt.imshow(masked_img)

fig.tight_layout()
#plt.savefig('output/cive_output')
plt.show()
