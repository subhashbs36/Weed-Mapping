import rasterio as rio
from rasterio.windows import Window

def crop_tiff_image(input_path, output_path, x1, y1, x2, y2):
    # Open the original TIFF image
    with rio.open(input_path) as src:
        # Calculate the crop window coordinates
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        # Determine the crop window using Window.from_slices
        window = Window.from_slices((top, bottom+1), (left, right+1))

        # Read the data from the crop window
        cropped_data = src.read(window=window)

        # Get the metadata for the cropped data
        cropped_meta = src.meta.copy()
        cropped_meta['height'], cropped_meta['width'] = cropped_data.shape[-2:]

        # Update the transform to reflect the new cropping
        cropped_transform = rio.windows.transform(window, src.transform)
        cropped_meta['transform'] = cropped_transform

        # Write the cropped data to a new GeoTIFF file
        with rio.open(output_path, 'w', **cropped_meta) as dst:
            dst.write(cropped_data)

# Example usage:
input_tiff_path = "Images/Tiff Images/test4.tif"
output_tiff_path = "Images/test2.tif"
x1, y1 = 20.598068598068494 ,  1557.9415584415583
x2, y2 = 2471.4805194805185 ,  4814.684815184814
crop_tiff_image(input_tiff_path, output_tiff_path, x1, y1, x2, y2)
