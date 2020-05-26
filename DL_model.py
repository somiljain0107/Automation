#!/usr/bin/env python
# coding: utf-8
from keras.models import Sequential
from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, Dropout
from keras.datasets import fashion_mnist
from keras.callbacks import Callback
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
X_train = x_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32')
X_train = X_train/255.0
X_test = x_test.reshape(10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test = x_test/255.0
y_train_new = np_utils.to_categorical(y_train)
y_test_new = np_utils.to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1) ))
model.add(MaxPool2D(pool_size=(2, 2) ))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10,  activation = 'softmax'))          
model.summary()
class myCallback(Callback):
      def on_epoch_end(self, epoch, logs={}):
          
            directory='/root/Automation/accuracy_dl.txt' 
            var=logs.get('accuracy')
            with open(directory, 'w') as write:
                write.write(np.array2string(var))




callbacks = myCallback()

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'] )

no_of_epochs=5
history = model.fit(X_train, y_train, epochs=no_of_epochs, callbacks=[callbacks])

model.save("myDLmodel.h5")
print("Your Model Has Been Saved")


