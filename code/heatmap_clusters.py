
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
import rasterio as rio

image_file1 = "images/image1.jpg"
image_file2 = "images/image2.jpg"
image_file3 = "images/image3.jpg"
image_file4 = "images/image4.jpg"
image_file5 = "images/image5.jpg"
image_org = "images/sliced.jpg"
image_mask = "images/out.png"
tiff_image = "Images/sliced.tif"
ground_truth = "images/GT.jpg"

def geoCordinates(img_path, x, y):
    # Open the TIFF file
    with rio.open(img_path) as src:
        # Get the georeferencing information
        transform = src.transform
        # Convert the x, y coordinates to latitude and longitude
        lon, lat = transform * (x, y)
        return lon, lat 

def draw_countors(img):
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours,hierarchy= cv.findContours(grey_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mid_points=[]

    #print(hierarchy)

    filtered_contours=[]
    for idx, contour in enumerate(contours):
        area=int(cv.contourArea(contour))
        if area>500:
            #print((contours[idx][1]))
            filtered_contours.append(contour)
            _,arr=cv.minEnclosingTriangle(contour)
            arr1=arr.tolist()
            #print(arr1[1][0][0])
            x1=arr1[0][0][0]
            y1=arr1[0][0][1]
            x2=arr1[1][0][0]
            y2=arr1[1][0][1]
            x3=arr1[2][0][0]
            y3=arr1[2][0][1]
            cent_x=(x1+x2+x3)/3
            cent_y=(y1+y2+y3)/3
            mid_points.append([cent_x,cent_y])
            #print([cent_x,cent_y])
            lat_coord,lon_coord=geoCordinates(tiff_image,cent_x,cent_y)
            print([cent_x,cent_y],[lat_coord,lon_coord])


    cv.drawContours(img,filtered_contours, -1, (0,255,0), 3)
    return img

def get_CIVE_band(img):
    img = cv.GaussianBlur(img,(35,35),0)
    CIVE_band = 0.441*img[:,:,0] - 0.881*img[:,:,1] + 0.385*img[:,:,2] + 18.787
    return CIVE_band

def apply_mask(img, vegetation_index_band):
    red_array = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    red_array[0:img.shape[0], 0:img.shape[1]] = (171, 42, 42)

    orange_array = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    orange_array[0:img.shape[0], 0:img.shape[1]] = (255, 160, 30)

    ret, otsu = cv.threshold(vegetation_index_band,0,255,cv.THRESH_BINARY)
    soil_img = cv.bitwise_and(img,img,mask = otsu.astype(np.uint8))

    ret, otsu1 = cv.threshold(vegetation_index_band,0,255,cv.THRESH_BINARY_INV)
    orange_out = cv.bitwise_and(orange_array,orange_array,mask = otsu1.astype(np.uint8))
    org_mask = orange_out
    orange_out[:,:,1] = cv.bitwise_and(orange_out[:,:,1] + (vegetation_index_band.astype(np.uint8)),orange_out[:,:,1],mask=None)


    ret, otsu2 = cv.threshold(vegetation_index_band,-7.78,255,cv.THRESH_BINARY_INV)
    red_out = cv.bitwise_and(red_array,red_array,mask = otsu2.astype(np.uint8))
    red_out[:,:,1] = cv.bitwise_and(red_out[:,:,1] + (0.85*vegetation_index_band).astype(np.uint8),red_out[:,:,1],mask=None)

    ret, otsu3 = cv.threshold(vegetation_index_band,-7.78,255,cv.THRESH_BINARY)
    orange_inv = cv.bitwise_and(orange_out,orange_out,mask = otsu3.astype(np.uint8))

    mask = cv.bitwise_or(red_out, orange_inv, mask = None)

    mask = draw_countors(mask)

    kernel1 = np.ones((3,3), np.float32)/9
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
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (9,7))

ax = plt.subplot(1, 1, 1)
ax.set_title("original")
plt.imshow(masked)
plt.show()
