""" The Code is for extracting features from the image dataset and storing then into a database. """

# Importing the required modules
from FeatureExtractor import featureExtractorClass
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


# Checking for the GPU for faster computation
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())


# Initializing the Variables
features = []
fe = featureExtractorClass()
feature_path = "./featureDatabase.npy"

# Iterate through images for feature extraction
# And storing the images path in a list
img_paths = []
for img_path in sorted(Path("./combined_datasets").glob("*.jpg")):
    print(img_path)
    img_paths.append(img_path)

    # Extract Features
    feature = fe.extract(img = Image.open(img_path))
    features.append(feature)

# Save the Numpy array (.npy) on designated path
np.save(feature_path, features)

# Saving the image paths in a text file for further use
with open("imgPaths.txt", 'w') as f :
    for img_path in img_paths :
        f.write(str(img_path) + '\n')