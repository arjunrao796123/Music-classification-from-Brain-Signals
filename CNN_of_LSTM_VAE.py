#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 5
num_classes = 12
epochs = 12


# In[ ]:


img_rows, img_cols = 512, 512


# In[ ]:


import numpy as np
preds_test = np.load('/content/drive/My Drive/preds_test.npy')
preds = np.load('/content/drive/My Drive/preds.npy')
work_test = np.load('/content/drive/My Drive/work_test.npy')
work_train = np.load('/content/drive/My Drive/work_train.npy')


# In[4]:


x_train = work_train.reshape(work_train.shape[0], img_rows, img_cols, 1)
x_train_vae = preds.reshape(preds.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
print(x_train_vae.shape)


# In[ ]:


y_train = np.load('train_target.npy')
y_test = np.load('test_target.npy')
x_train = x_train.astype('float32')
x_train_vae = x_train_vae.astype('float32')


# In[ ]:



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


class TestCallback(keras.callbacks.Callback):
    def __init__(self, work_test):
        self.x_test = work_test.reshape(work_test.shape[0], img_rows, img_cols, 1)

    def on_epoch_end(self, epoch, logs={}):
        x_test= self.x_test
        y_test = np.load('test_target.npy')
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# In[11]:


x_test = work_test.reshape(work_test.shape[0], img_rows, img_cols, 1)
model.fit(x_train_vae, y_train,
          batch_size=32,
          epochs=25,
          verbose=1,callbacks=[TestCallback((x_test))])


# In[ ]:




