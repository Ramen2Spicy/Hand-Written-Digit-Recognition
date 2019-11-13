#!/usr/bin/env python
# coding: utf-8

# In[61]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


# In[62]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[71]:


print(X_train.shape)
print(X_test.shape)


# In[73]:


print(X_train[0])


# In[74]:


print(y_train[0])


# In[78]:


plt.imshow(X_train[1])


# In[81]:


y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot[1])


# In[82]:


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


# In[83]:


model = Sequential()
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28, 28,1)))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
        


# In[84]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[85]:


hist = model.fit(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), epochs=1)


# In[86]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('How Accurate is it?')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['Train', 'Val_Train'], loc= 'upper left')
plt.show()


# In[87]:


prediction = model.predict(X_test[0:10])
print(prediction)


# In[89]:


for i in range(1,10):
    image = X_test[i-1]
    image = np.array(image, dtype= 'float')
    pixels = image.reshape(28, 28)
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print((np.argmax(prediction[i-1])))


# In[ ]:




