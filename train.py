#!/usr/bin/env python
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle 

# set the matplotlib backend so that figures can be saved in the background 
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import cv2
import os

# construct the arguement parse and parse the arguements
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                 help="/Users/lukeclarke/Documents/BTS_MBDS_2021/projects/keras_multi_label_classification/working/dataset")
ap.add_argument("-m", "--model", required=True,
                 help="/Users/lukeclarke/Documents/BTS_MBDS_2021/projects/keras_multi_label_classification/working")
ap.add_argument("-l", "--labelbin", required=True,
                 help="/Users/lukeclarke/Documents/BTS_MBDS_2021/projects/keras_multi_label_classification/working")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                 help="/Users/lukeclarke/Documents/BTS_MBDS_2021/projects/keras_multi_label_classification/working")

args = vars(ap.parse_args())


# initialize the number of epochs to train for, initial learning rate, batch size and image dimensions

EPOCHS = 75
INIT_LR = 1e-3               # default for the Adam optimizer
BS = 32                      # batch size of 32
IMAGE_DIMS = (96, 96, 3)


#LOADING & PREPROCESSING 
#grab the image paths and randomly shuffle them
print("Loading Images ...")
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)


#initialize data and labels
data = []
labels = []


#loop over input images
for imagePath in imagePaths:
    # load the image, preprocess it, and store in a data list 
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0])) # resizing to 96*96
    image = img_to_array(image)     # Converts a PIL Image instance to a Numpy array.
    data.append(image)

    #extract class labels from the image path and update the labels list
    #handle splitting the image path into multiple labels for our multi-label classification task
    l = label = imagePath.split(os.path.sep)[-2].split("_") 
    labels.append(l)


# scale the raw pixel intensities to a range between [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("Data matrix: {} images ({:.2f}MB)".format(len(imagePaths),
                                                        data.nbytes / (1024 * 1000)))


# binarize the labels using scikit learns special multi label binarizer implementation
# transform our human-readable labels into a vector that encodes which class(es) are present in the image
# like one hot encoder, but this is a case of two hot encoding 
print("Class Labels: ")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)     

#
for (i, label) in enumerate(mlb.classes_):
    print("{}, {}".format(i+1, label))




# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# construct the image generator for data augmentation
# Data augmentation is a best practice and a most-likely a “must” if you are working with less than 1,000 images per class.
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")


# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("Compiling model...")
model = SmallerVGGNet.build(width = IMAGE_DIMS[1], height = IMAGE_DIMS[0], depth = IMAGE_DIMS[2],
    classes = len(mlb.classes_), finalAct = "sigmoid")


# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


# compile the model using binary cross-entropy rather than categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here is to treat
# each output label as an independent Bernoulli distribution
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


#train the network 
print("Train network...")
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=BS), 
             validation_data=(X_test, y_test), 
             steps_per_epoch=len(X_train)//BS, 
             epochs=EPOCHS, verbose=1)


print("Serializing network...")
model.save(args["model"])

# save the multilabelbinarizer to a disk 
print("Serializing label binarizer...")
f = open(args['labelbin'], "wb")
f.write(pickle.dumps(mlb))
f.close


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])









