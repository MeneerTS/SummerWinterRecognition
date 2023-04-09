# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:31:13 2022

@author: thijs
"""

import tensorflow as tf #version 2.9.1
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os
import PIL
import random
import matplotlib.pyplot as plt


#PROGRAMMED WITH Tenserflow version 2.9.1

"""
We kunnnen tf.keras.utils.image_dataset_from_directory() gebruiken om van de map een dataset te maken.

We moeten daarvoor de labels hebben, alfabetisch gesorteerd op de naam van de afbeelding.

We kunnen hiervoor de naam van de afbeeldingen gelijk te maken aan de hoeveelste afbeelding het is.

We moeten dan wel opletten dat de volgorde van de labels die we opslaan daar ook rekening mee houdt (De eerste eerst).

NVM handmatig is veel chiller
"""


#We want our test set to be the same every time!
tf.random.set_seed(
    1337123
)

def parallel_datetimebuilder(years, months, days):
    output = []
    for i in range(len(years)):
        output.append(datetime.date(years[i], months[i], days[i]))
    return output


n_data = len(os.listdir("data_collection\\images\\"))



image_data_set = tf.data.Dataset.list_files("data_collection\\images\\*", shuffle=False)


getallen = np.arange(n_data)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [90, 120])

#List of datetime.date labels

prelabelslist = []
for labelfile in os.listdir("data_collection\\labels\\"):
    labels = open("data_collection\\labels\\" + labelfile, 'r')
    lines = labels.readlines()
    prelabelslist += lines
    print(len(lines))
    labels.close()


postlabelslist = []

#zomer 1; winter 0
for prelabel in prelabelslist:
    splitlabel = prelabel.split()
    if int(splitlabel[1]) > 3 and int(splitlabel[1]) < 10:
        postlabelslist.append(1)
    else:
        postlabelslist.append(0)
    
imagename_list = os.listdir("data_collection\\images\\")
image_list = []
for imagename in os.listdir("data_collection\\images\\"):
    file_path = "data_collection\\images\\" + imagename
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    image_list.append(img)
    


def process_path(index):
  label = postlabelslist[index]
  # Load the raw data from the file as a string
  file_path = "data_collection\\images\\" + imagename_list[index]
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label







#labeltensor = tf.Tensor(postlabelslist, , dtype =tf.int32)

def balance_dataset(images, labels):
    """
    Works only for datasets where the labels are ones or zeros
    
    Removes data points until there are just as many ones as zeros
    """
    assert len(images) == len(labels)
    
    images_ones = []
    images_zeros = []
    labels_ones = []
    labels_zeros = []
    
    for i in range(len(images)):
        if labels[i] == 0:
            images_zeros.append(images[i])
            labels_zeros.append(labels[i])
        else:
            images_ones.append(images[i])
            labels_ones.append(labels[i])
    
    
    while len(images_zeros) > len(images_ones):
        toremove = random.randint(0,len(images_zeros) - 1)
        images_zeros.pop(toremove)
        labels_zeros.pop(toremove)
    
    while len(images_zeros) < len(images_ones):
        toremove = random.randint(0,len(images_ones) -1)
        images_ones.pop(toremove)
        labels_ones.pop(toremove)
    
    return images_zeros + images_ones, labels_zeros + labels_ones


image_list, postlabelslist = balance_dataset(image_list, postlabelslist)
print("balance succes")
ourdata = tf.data.Dataset.from_tensor_slices((image_list, postlabelslist))
ourdata = ourdata.shuffle(buffer_size = 10000, reshuffle_each_iteration=False)

ourdata_size = len(ourdata)
train_size = int(0.7 * ourdata_size)
val_size = int(0.15 * ourdata_size)
test_size = int(0.15 * ourdata_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = ourdata.take(train_size).prefetch(buffer_size=AUTOTUNE).cache().batch(50)
test_and_val_ds = ourdata.skip(train_size)
test_ds = test_and_val_ds.take(test_size).prefetch(buffer_size=AUTOTUNE).cache().batch(50) 
val_ds = test_and_val_ds.skip(test_size).prefetch(buffer_size=AUTOTUNE).cache().batch(50) 




#60% acc on validation with 1/10 data after 50 epochs
model_augment_image = tf.keras.models.Sequential([

  layers.RandomFlip("horizontal",
                      input_shape=(90,
                                  120,
                                  3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.5),
  layers.Rescaling(1./255, input_shape=(90, 120, 3)),  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  #layers.AveragePooling2D(),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  
  layers.Dense(2)

    ])


#60% acc on validation with 1/10 data after 50 epochs, reaches 60% quickly

model_dropout = tf.keras.models.Sequential([

  
  layers.Rescaling(1./255, input_shape=(90, 120, 3)),  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  #layers.AveragePooling2D(),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  
  layers.Dense(2)

    ])

#a steady 63% acc on validation with 1/10 data after 50 epochs
#Lol, ran it again and got 70% acc and completely different graph lol
model_both = tf.keras.models.Sequential([

  layers.RandomFlip("horizontal",
                      input_shape=(90,
                                  120,
                                  3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.5),
  layers.Rescaling(1./255, input_shape=(90, 120, 3)),  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  #layers.AveragePooling2D(),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  
  layers.Dense(2)

    ])



#CHANGE FOR DIFFERENT MODELS
model = model_both



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )


  
if True:
    our_epochs = 100
    history = model.fit(
      x = train_ds,
      validation_data=val_ds,
      epochs=our_epochs,
      callbacks=[callback]
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(history.history["loss"]))
    
    plt.figure(figsize=(8, 8))
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
    plt.show()
    
if False: 
    #"C:\\Users\\thijs\\Documents\\Gesorteerde documenten\\1. Kunstmatige Intelligentie\\Eigen projecten\\Image recognition\\pf pic.jpg"
    #
    #
    #
    #
    img = tf.keras.utils.load_img(
        "C:\\Users\\thijs\\Documents\\Gesorteerde documenten\\1. Kunstmatige Intelligentie\\Eigen projecten\\Image recognition\\Hippe happen1.jpg", target_size=(90, 120)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(np.argmax(score), 100 * np.max(score))
    )




print(max(history.history["val_loss"]))


#tfdf.keras.core.datetime(2012,5,1)

#train_ds = x.map(process_path, num_parallel_calls=1)

#process_path(index)

