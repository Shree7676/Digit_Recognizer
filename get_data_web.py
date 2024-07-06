import numpy as np
import pandas as pd
from PIL import Image

# 1. Read the image (assuming it's already loaded into a numpy array)
# Replace 'image_path' with the path to your image
image_path = "saved_image.png"
image = Image.open(image_path).convert(
    "L"
)  # Load image and convert to grayscale ('L' mode)
image_array = np.array(image)  # Convert image to numpy array

# 2. Flatten the image array
flattened_array = image_array.flatten()

# 3. Create a pandas Series
image_series = pd.Series(flattened_array)

# Optionally, you can add labels/index if needed:
# image_series.index = pd.RangeIndex(start=0, stop=len(image_series))

# Print the Series (optional)
print(image_series)
print(image_series.columns)
