#!/usr/bin/env python
# coding: utf-8

# In[23]:


from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.optimizers import adam
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.callbacks import Callback
import numpy as np
import os 
model = load_model("myDLmodel.h5")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

X_train = x_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32')
X_train = X_train/255.0

X_test = x_test.reshape(10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test = x_test/255.0

y_train_new = np_utils.to_categorical(y_train)
y_test_new = np_utils.to_categorical(y_test)
class myCallback(Callback):
      def on_epoch_end(self, epoch, logs={}):
            directory='/root/Automation/accuracy_dl.txt' 
            var=logs.get('accuracy')
            with open(directory, 'w') as write:
                write.write(np.array2string(var))
callbacks = myCallback()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'] )
history = model.fit(X_train, y_train, epochs=5, callbacks=[callbacks])


# In[25]:





# In[ ]:




