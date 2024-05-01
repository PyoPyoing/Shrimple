#!/usr/bin/env python
# coding: utf-8

# In[22]:


get_ipython().system('nvidia-smi')


# In[23]:


import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt


# In[24]:


IMAGE_SIZE = 640
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 250


# RGB Red Green blue

# In[25]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\Pyo\\OneDrive - DEPED REGION 4A-3\\Desktop\\Jupyter Shrimple\\Test-20240427T184520Z-001\\Test",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)


# In[26]:


class_names = dataset.class_names
class_names


# In[27]:


plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
  for i in range(12):
    ax = plt.subplot(3,4,i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[label_batch[i]])
    print(image_batch[i].shape)


# In[28]:


len(dataset)


# In[29]:


train_size = 0.8
len(dataset)*train_size


# In[30]:


train_ds = dataset.take(17)
len(train_ds)


# In[31]:


test_ds = dataset.skip(17)
len(test_ds)


# In[32]:


val_size=0.1
len(dataset)*val_size


# NOTE: validation size is zero ;-; our dataset is small ill figure it out later

# In[33]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1,test_split=0.1, shuffle=True, shuffle_size=10000):

  ds_size = len(ds)
  if shuffle:
    ds = ds.shuffle(shuffle_size, seed=12)
  train_size = int(train_split * ds_size)

  val_size = int(val_split * ds_size)

  train_ds = ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)
  return  train_ds, val_ds, test_ds


# In[34]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[35]:


len(train_ds)


# In[36]:


len(val_ds)


# In[37]:


len(test_ds)


# In[38]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[39]:


resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.Rescaling(1.0/255)
])


# In[40]:


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])


# In[41]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[42]:


from keras.layers import Input, Dropout, BatchNormalization

# Define the input shape
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 4

# Define the model
model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    Dropout(0.5),
    layers.Dense(128, activation='relu'),
    Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()


# In[43]:


model.summary()


# In[44]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# 

# In[ ]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=150,
)


# In[29]:


scores = model.evaluate(test_ds)


# In[30]:


scores


# In[31]:


history


# In[32]:


history.params


# In[33]:


history.history.keys()


# In[34]:


type(history.history['loss'])


# In[35]:


len(history.history['loss'])


# In[36]:


history.history['loss'][:5]


# In[18]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[38]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[19]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):

    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[40]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[4]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

        plt.axis("off")


# In[ ]:





# In[46]:


import os

# Define the directory path
directory_path = "C:\\Users\\Pyo\\OneDrive - DEPED REGION 4A-3\\Desktop\\Jupyter Shrimple\\ShrimpleModelProto"

# Get list of model versions
model_versions = [int(i) for i in os.listdir(directory_path) if i.isdigit()]
# Increment the maximum version number
model_version = max(model_versions) + 1 if model_versions else 1

# Save the model in the native Keras format
model.save(os.path.join(directory_path, f"Test_{model_version}.keras"))


# In[47]:


model.save("Shrimple_model_PROTOTYPE.keras")


# In[49]:


model.save("ShrimplePrototype.h5")


# In[51]:


model.save("Shrimple_Proto.keras")


# In[66]:


def predict_image(model, img_path, class_names):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis('off')
    plt.show()

    return img, predicted_class

sample_img_path = "D:\\Downloads\\Downloads 2024\\Jupyter Shrimple\\1-s2.0-S0932473920300031-gr1.jpg"
img, predicted_class = predict_image(model, sample_img_path, dataset.class_names)

print("Displayed image:", sample_img_path)
print("Predicted class:", predicted_class)


# In[67]:


print(model.summary())


# In[58]:


def load_saved_model(model_path):
    """
    Load a previously saved model from the specified path.

    Args:
    - model_path (str): The path to the saved model.

    Returns:
    - loaded_model: The loaded model.
    """
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

def predict_with_loaded_model(loaded_model, img_path, class_names):
    """
    Predict the class of an image using a loaded model.

    Args:
    - loaded_model: The loaded model.
    - img_path (str): The path to the image file.
    - class_names (list): List of class names.

    Returns:
    - img: The loaded image.
    - predicted_class (str): The predicted class.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize the image

    prediction = loaded_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis('off')
    plt.show()

    return img, predicted_class

# Example usage:
model_path = 'C:\\Users\\Pyo\\OneDrive - DEPED REGION 4A-3\\Desktop\\Jupyter Shrimple\\Test-20240427T184520Z-001\\Shrimple_Proto.keras'  # Path to the saved model
loaded_model = load_saved_model(model_path)

sample_img_path = 'D:\\Downloads\\Downloads 2024\\Jupyter Shrimple\\images(1).jpg'
img, predicted_class = predict_with_loaded_model(loaded_model, sample_img_path, dataset.class_names)

print("Displayed image:", sample_img_path)
print("Predicted class:", predicted_class)


# In[2]:


jupyter nbconvert SHRIMPLE PROTOTYPE 1.ipynb --to python


# In[10]:


import tensorflow as tf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

# Define the path to the saved model
model_path = 'C:\\Users\\Pyo\\OneDrive - DEPED REGION 4A-3\\Desktop\\Jupyter Shrimple\\Test-20240427T184520Z-001\\Shrimple_Proto.keras'

# Load the model
loaded_model = tf.keras.models.load_model(model_path)

# Optionally, you can check the architecture of the loaded model
loaded_model.summary()

# Optionally, you can also check the model's training configuration
print(loaded_model.optimizer)

# Now you can use the loaded_model for inference or fine-tuning
# For example:
# result = loaded_model.predict(input_data)


# In[5]:


import tensorflow as tf

# Save the Keras model as a .pbtxt file
tf.keras.models.save_model(model, 'model.pbtxt')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
model_tflite = converter.convert()

# Save the TensorFlow Lite model as a .tflite file
with open("model.tflite", "wb") as f:
    f.write(model_tflite)


# interpreter = tf.lite.Interpreter(model_path = tmpsgchtxmv)
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print("Input Shape:", input_details[0]['shape'])
# print("Input Type:", input_details[0]['dtype'])
# print("Output Shape:", output_details[0]['shape'])
# print("Output Type:", output_details[0]['dtype'])
