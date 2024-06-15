#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# # Making The Path
# 

# In[2]:


#PATH ALLOCATION
data_dir=os.path.join('Data_language/data')
classes=os.listdir(data_dir)
classes


# # Removing Unwanted Extensions

# In[3]:


#REMOVING DOGGY IMAGES
import cv2
import imghdr                                                              
                                                                                
img_ext=['jpeg','jpg']                                              

for img_c in classes:
    for img_v in os.listdir(os.path.join(data_dir,img_c)):
        img_p=os.path.join(data_dir,img_c,img_v)
        try:
            #img_mat=cv2.imread(img_p)
            img_mat=plt.imread(img_p)
            img_tip=imghdr.what(img_p)
            if(img_tip not in img_ext):
                print('Image not in ext list {}'.format(img_p))
                os.remove(img_p)
            
        except Exception as e:
            #print('Issue with image {}'.format(img_p))
            print(e)
                


# # Preprocessing & Spliting into train and test

# In[5]:


# Load and preprocess data from subfolders
def load_data(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256,256))
            #image = plt.imread(image_path)
            #image = tf.image.resize(image,(256,256))
            images.append(image)
            labels.append(class_index)
    return images, labels

# Load and preprocess image data
images, labels = load_data('Data_language\data')


# Split data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize pixel values to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Print shapes for verification
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)


# # Visualizing the samples

# In[6]:


#plot sample train data
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for i,j in enumerate(train_images[0:4]):
    ax[i].imshow(j)
    ax[i].title.set_text(train_labels[i])


# # Building The Model

# In[7]:


#MODEL BULIDING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout


# In[22]:


model=Sequential()
#hidden layers,16 32 16 256 are filters
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#output layers always comes in 1d
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='sigmoid'))


# In[28]:


model.compile('adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#adam optiimizer

# # Training The Model

# In[29]:


#TRAINING
tensor_call=tf.keras.callbacks.TensorBoard(log_dir='logs')
tensor_call
#using logs & call backs we can manage the learning phase at any instance


# In[30]:


#fit the model---->training component
history=model.fit(train_images,train_labels,epochs=50,callbacks=[tensor_call])


# In[31]:


di=history.history #gives entire trained DL model info like loss,validation loss and accuracy
di


# # Vizualizing

# In[32]:


#<----PERFORMANCE------>

#LOSS VIZULATIZATION
plt.figure(figsize=(5,5))
plt.plot(di['loss'],color="red",label="loss")
plt.show()
#observe by considering Y axis for both--->graphs down means accuacry increasing


# In[33]:


#ACCURACY VIZULIZATION
plt.figure(figsize=(5,5))
plt.plot(di['accuracy'],color="green",label="loss")
plt.show()


# # Accuracy Measure

# In[34]:


import tensorflow as tf

#'model' is your trained CNN model
# test_images and test_labels contain your test data

# Use the evaluate method to calculate accuracy
results = model.evaluate(test_images, test_labels, verbose=0)


# The second element of the results list corresponds to accuracy
accuracy = results[1]

print(f"Accuracy: {accuracy * 100:.2f}%")


# # Unknow Prediction

# In[35]:


ref={0:"English",1:"Hindi",2:"Telugu"}


#unseen prediction
from numpy import *
img=plt.imread('hindi1.jpg')
plt.imshow(img)

#resize according to layers
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int)) #adjusted according to RGB b/w 0 to 1
plt.show()

#optimize the new image
resize=resize/255
#expand your image array
img=expand_dims(resize,0)


predictions = model.predict(img)

# Convert the predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)


print("The langauge detected is",ref[predicted_labels[0]])


# In[36]:


predictions


# # Saving The Model

# In[38]:


model.save('detector.keras')


# In[ ]:




