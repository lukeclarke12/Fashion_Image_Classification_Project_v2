#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# load the image
image = cv2.imread(args['image'])
output = imutils.resize(image, width=400)

# pre-process image for classification in the same manner as training data
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# load the trained convolutional neural network and the multi-label binarizer
print("Loading network...")
model = load_model(args['model'])
mlb = pickle.loads(open(args['labelbin'], "rb").read())



# We classify the (preprocessed) input image and extract the top two class labels indices by
# Sorting the array indexes by their associated probability in descending order
# Grabbing the first two class label indices which are thus the top-2 predictions from our network
print("Classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]


# Now we prepare the class labels + associated confidence values for overlay on the output image
# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

