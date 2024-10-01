import rasterio
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array

# Load the trained model
model_path = f'./demo/random_forest_model.joblib'
model = load(model_path)

# Path to the input TIFF image
tif_path = f'./demo/img_demo_RF_small03.tif'

# Load the image using gdal
img_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)

# Read the image bands into a numpy array
img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

# Get image dimensions and number of bands
rows, cols, bands = img.shape
print(f'Image extent: {rows} x {cols} (row x col)')
print(f'Number of Bands: {bands}')

# Reshape the image to 2D array (pixels x bands)
img_reshaped = img.reshape(-1, bands)

# Predict the class for each pixel using the loaded model
class_prediction = model.predict(img_reshaped)

# Reshape the predictions back to the image dimensions
class_prediction = class_prediction.reshape(rows, cols)

# Generate a mask from the first band (assuming it corresponds to valid pixels)
mask = img[:, :, 0] > 0

# Apply the mask to the classification result
class_prediction_masked = np.where(mask, class_prediction, np.nan)

# # Plot the masked classification result
# plt.subplot(121)
# plt.imshow(class_prediction, cmap=plt.cm.Spectral)
# plt.title('Classification Unmasked')

# plt.subplot(122)
# plt.imshow(class_prediction_masked, cmap=plt.cm.Spectral)
# plt.title('Classification Masked')

# plt.show()

# Path to save the classified image
classification_image = f'./demo/classified_image_img_demo_RF_small04.tif'

# Create a new TIFF file to save the classification result
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(classification_image, cols, rows, 1, gdal.GDT_Float32)

# Copy geotransform and projection from the original image
out_ds.SetGeoTransform(img_ds.GetGeoTransform())
out_ds.SetProjection(img_ds.GetProjection())

# Write the classification result to the output file
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(class_prediction_masked)
out_band.SetNoDataValue(-9999)
out_band.FlushCache()

# Close the dataset
out_ds = None

print(f'Classification saved to {classification_image}')

# Open and display the classified image
out_ds = gdal.Open(classification_image)
out_classification = out_ds.GetRasterBand(1).ReadAsArray()

# plt.imshow(out_classification, cmap=plt.cm.Spectral)
# plt.title('Resulting Classification')
# plt.show()

# Close the file
out_ds = None
