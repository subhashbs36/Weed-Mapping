
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import cv2 as cv
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio as rio


image_org = "Images/farm_weedmap_img.jpg"
image_mask = "images/out.png"

def get_CIVE_band(img):
    img = cv.GaussianBlur(img,(35,35),0)
    CIVE_band = 0.441*img[:,:,0] - 0.881*img[:,:,1] + 0.385*img[:,:,2] + 18.787
    return CIVE_band

def apply_mask(img, vegetation_index_band):

    red_array = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    red_array[0:img.shape[0], 0:img.shape[1]] = (171, 42, 42)

    orange_array = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    orange_array[0:img.shape[0], 0:img.shape[1]] = (255, 160, 30)

    ret, otsu = cv.threshold(vegetation_index_band,1.28,255,cv.THRESH_BINARY)
    soil_img = cv.bitwise_and(img,img,mask = otsu.astype(np.uint8))

    ret, otsu1 = cv.threshold(vegetation_index_band,1.28,255,cv.THRESH_BINARY_INV)
    orange_out = cv.bitwise_and(orange_array,orange_array,mask = otsu1.astype(np.uint8))
    orange_out[:,:,1] = cv.bitwise_and(orange_out[:,:,1] + (vegetation_index_band.astype(np.uint8)),orange_out[:,:,1],mask=None)


    ret, otsu2 = cv.threshold(vegetation_index_band,-4.78,255,cv.THRESH_BINARY_INV)
    red_out = cv.bitwise_and(red_array,red_array,mask = otsu2.astype(np.uint8))
    red_out[:,:,1] = cv.bitwise_and(red_out[:,:,1] + (0.85*vegetation_index_band).astype(np.uint8),red_out[:,:,1],mask=None)

    ret, otsu3 = cv.threshold(vegetation_index_band,-4.78,255,cv.THRESH_BINARY)
    orange_inv = cv.bitwise_and(orange_out,orange_out,mask = otsu3.astype(np.uint8))

    mask = cv.bitwise_or(red_out, orange_inv, mask = None)

    kernel1 = np.ones((2,2), np.float32)/4
    smooth_mask = cv.filter2D(src = mask, ddepth = -1, kernel = kernel1)

    final = cv.bitwise_or(smooth_mask, soil_img, mask = None)

    return final

def generate_heatmap(image_org, image_mask):

    org_img = np.array(Image.open(image_org))
    mask_img = np.array(Image.open(image_mask))

    CIVE_band = get_CIVE_band(mask_img)
    masked = apply_mask(org_img, CIVE_band)

    return org_img, masked


org, masked = generate_heatmap(image_org, image_mask)


#fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (9,7))

# ax = plt.subplot(1, 2, 1)
# ax.set_title("original")
# plt.imshow(org)

# ax = plt.subplot(1, 2, 2)
# ax.set_title("soil")
# plt.imshow(masked)

ax = plt.subplot(1, 1, 1)
ax.set_title("original")
plt.imshow(masked)
plt.savefig('output/final.png')#'output/final.png')

plt.savefig('output/final')#'output/fina')
plt.show()
