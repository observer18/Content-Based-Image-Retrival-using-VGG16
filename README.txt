******************************************************  README  ******************************************************



This is a simple Image based Search Engine. It is capable of extracting features from image database and then storing
them into a .npy file which is in numpy ndarray format. In the other script, it tries to extract feature from query
image, and compares them with the other extracted features stored in the database file. Here we have already stored
the database file named featureDatabase.npy in the directory itself. After comparison, it returns 30 images related to
query image from the database. In this way the 'Image Search Engine' works. In the project, we have particularly used a
pre-trained Neural Network named VGG16, which was pre-trained with the huge imagenet dataset.
