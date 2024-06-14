""" This particular code is a Local class which generalize the process of the Feature extraction of any image
 with a pre-trained neural network know as VGG16. We will be extracting the feature from the fully-connected
 layer (fc1) of VGG16 instead of going to the classification layer. """

# Importing the required modules
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from keras.applications.inception_v3 import InceptionV3

class featureExtractorClass :
    """ The main feature extractor class. This class extract features of an image using pre-trained VGG16. """

    def __init__(self) :
        """ Initializing function. """
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights = 'imagenet')

        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs = base_model.input, outputs = base_model.get_layer('fc1').output)

    def extract(self, img) :
        """ Extracting the features from the images using the model initialized in the __init__ section. """
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space to RGB
        img = img.convert('RGB')

        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

    def __del__(self) :
        """ Destructor function. """
        pass
