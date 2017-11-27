import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

n_classes = 10
# load data
# I used the mnist data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

#uint8 to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#normalize to [0, 1]
X_train /= 255
X_test /= 255
#one-hot-encoding
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

from keras.layers import Flatten, Conv2D, MaxPooling2D
#define model
model_c = Sequential()
#first convolutional layer
model_c.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model_c.add(MaxPooling2D(pool_size=(2, 2)))
#model_c.add(Dropout(0.25))
#second convolutional layer
model_c.add(Conv2D(16, (3, 3), activation = 'relu'))
model_c.add(MaxPooling2D(pool_size=(2, 2)))
#model_c.add(Dropout(0.25))
#fully-connected layer
model_c.add(Flatten())
#model_c.add(Dense(256, activation='relu'))
#model_c.add(Dropout(0.5))
model_c.add(Dense(10))
model_c.add(Activation('softmax'))
# Define loss
model_c.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
				
history_c = model_c.fit(X_train[..., np.newaxis], y_train, 
                        batch_size = 128, epochs=10, verbose=1,
                        validation_data=(X_test[..., np.newaxis], y_test))