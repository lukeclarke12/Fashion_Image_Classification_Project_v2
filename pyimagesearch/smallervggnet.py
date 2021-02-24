#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        # height, widthm depth refer to the shape of the image we input into the model
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL
        # 32 filters and a 3*3 convolutional kernal with one layer of padding
        # Batch normalization is a technique to automatically standardize the inputs to the neural network
        # Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis
        # Our POOL  layer uses a 3 x 3  POOL  size to reduce spatial dimensions quickly from 96 x 96
        # to 32 x 32 (we’ll be using  96 x 96 x 3 input images to train our network as we’ll see in the next section)
        # Dropout kills neurons and helps to reduce overfitting
		model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
        # We’re increasing our filter size from 32  to 64 . The deeper we go in
        # the network, the smaller the spatial dimensions of our volume, and
        # the more filters we learn
        # We decreased how max pooling size from 3 x 3  to 2 x 2  to ensure we do not reduce our spatial dimensions too quickly.
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
        # Adding a dense layer at the end equal to the number of classes we want to classify
		model.add(Dense(classes))
		model.add(Activation(finalAct))

		# return the constructed network architecture
		return model
