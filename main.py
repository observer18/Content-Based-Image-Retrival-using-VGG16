""" This code is generally for testing the project, whether it is giving perfect result or not. """

# Importing the required modules
from FeatureExtractor import featureExtractorClass
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
from fileClass import files

# Loading the feature Database
feature_path = "./featureDatabase.npy"
features = np.load(feature_path)

# Insert the image query
path = files().openFile()

# Just taking care weather the query image is in jpg or png
try :
    img = Image.open(path.name)
except FileNotFoundError :
    print("The particular Image is missing from the directory...")

# Extracting the features of query image
fe = featureExtractorClass()
query = fe.extract(img)

# Calculate the similarity between query image and images in database
dists = scipy.linalg.norm(features - query, axis = 1)

# Loading the image paths in a List for proper indexing
with open("imgPaths.txt", 'r') as f :
    img_paths = [line.rstrip('\n') for line in f]

# Extract 30 images that have the lowest distance
n = 30
ids = np.argsort(dists)[:n]
# print(ids)
scores = [(dists[id], img_paths[id]) for id in ids]

# Visualize the result
axes = []
fig = plt.figure(figsize = (8,8))

# Plotting the similar images in window for simplicity
for a in range(n) :
    # Printing the similar images path
    score = scores[a]
    print(scores[a][1])

    # Adjusting the similar images in a single window
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title = str(score[0])
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))

fig.tight_layout()
plt.show()