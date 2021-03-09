import tensorflow as tf
import random
import pathlib
import os
import PIL
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow import keras


def predction(model,class_names,path,img_height=128,img_width=128):  
  #验证模型正确性
  file_path = path

  img = keras.preprocessing.image.load_img(
      file_path, target_size=(img_height, img_width)
  )
  
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) 

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  res ="This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
  return res