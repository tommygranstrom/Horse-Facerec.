import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

#Set variables
ant = "500" #Dummie variable
epochs = 5 
dataset_url = "/Users/tommygranstrom/Desktop/AI-Projekt Hästar/Antal/"+ant
data_dir = pathlib.Path(dataset_url)

batch_size = 32
img_height = 180
img_width = 180

#Create training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#Create validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Load and print class names (Folders in directory)
class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#Preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
num_classes = 3

#If using data augmentation

# data_augmentation = keras.Sequential(
#   [
#     layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                  input_shape=(img_height, 
#                                                               img_width,
#                                                               3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
#     layers.experimental.preprocessing.RandomZoom(0.1),
#   ]
# )

#Create the model
model = Sequential([
  #data_augmentation, #If augmentation
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  #layers.Dropout(0.2), #If using dropout
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.summary()

#Fit the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#Set variables and plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(7, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show() #Plot the figure
plt.savefig('plottar/'+str(epochs)+'epoker'+ant+'bilder.png') #Save plot

#If you want to save the model and dont need to create it again
# #save the model
# model.save('saved_model/my_model') 
import os
import csv

#Function that predict predicts a given image
def predict(path,answer,pathToFile):
  tobePredicted_path = path
  img = keras.preprocessing.image.load_img(
      tobePredicted_path, target_size=(img_height, img_width)
  )

  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  f = open(pathToFile,"a")
  f.write(answer+','+str(format(class_names[np.argmax(score)]))+','+str(100 * np.max(score))+'\n')
  f.close()
  #Print result
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )

  # if format(class_names[np.argmax(score)] == answer:
  #   return [1,100 * np.max(score)]

  # return score

#Paths to all test images  
pathTrolle="/Users/tommygranstrom/Desktop/AI-Projekt Hästar/Testbilder/Trolle test"
pathAmmie="/Users/tommygranstrom/Desktop/AI-Projekt Hästar/Testbilder/Ammie"
pathEffie="/Users/tommygranstrom/Desktop/AI-Projekt Hästar/Testbilder/Effie"


#Dummie string to create textfile containing predictions of all test images
pathfile = "Predictions/"+str(epochs)+'epoker'+ant+'bilder.txt'
f = open(pathfile,"w")
f.write('Horse,Prediction,Confidence\n')
f.close()

#List to auto predict all test images for all three horses
listTrolle = os.listdir(pathTrolle)    
listAmmie = os.listdir(pathAmmie)
listEffie = os.listdir(pathEffie)
corrCounter = 0

#Predict horses and print result
print('\nPredict trolle\n')
for file in listTrolle:
  if file!='.DS_Store':
    patH = pathTrolle + '/' + file
    predict(patH,"Trolle",pathfile)

print('\n----------------\n')


print('\nPredict Ammie\n')
for file in listAmmie:
  if file!='.DS_Store':
    patH = pathAmmie + '/' + str(file)
    predict(patH,"Ammie",pathfile)
print('\n----------------\n')
corrCounter = 0

print('\nPredict Effie\n')
for file in listEffie:
  if file!='.DS_Store':
    patH = pathEffie + '/' + str(file)
    predict(patH,"Effie",pathfile)
print('\n----------------\n')
