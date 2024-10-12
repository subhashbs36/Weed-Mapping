import rasterio as rio

def check_geolocation(image_path):
    with rio.open(image_path) as src:
        if src.crs is not None:
            return True, src.crs
        else:
            return False, None

# Provide the path to your TIFF image here
tiff_image_path = "Images/test2.tif"
# tiff_image_path = "Images/yp_upload3.tif"

has_geolocation, crs = check_geolocation(tiff_image_path)

if has_geolocation:
    print("The TIFF image contains geolocation data.")
    print("Coordinate Reference System (CRS):", crs)
else:
    print("The TIFF image does not contain geolocation data.")
