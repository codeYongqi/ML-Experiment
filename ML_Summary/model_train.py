import tensorflow as tf
import random
import pathlib
import os
import PIL

import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras.models import Sequential

def train_model():
  AUTOTUNE = tf.data.experimental.AUTOTUNE

  #加载本地图片集
  data_path = pathlib.Path('/Users/zhuyongqi/Desktop/coil-20-proc')
  data_var = pathlib.Path('/Users/zhuyongqi/Desktop/Col_Val')
  all_image_paths = list(data_path.glob('*/*'))  
  all_image_paths = [str(path) for path in all_image_paths]

  #设置数据集大小
  batch_size = 32
  img_height = 128
  img_width = 128
  
  #文件夹位置
  data_dir = data_path

  #生成训练集
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  #生成验证集
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_var,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


  class_names = train_ds.class_names

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  
  num_classes = 20

  #初始化卷积网络模型
  model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

  #选择模型优化器和损失函数
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  #设置迭代次数     
  epochs=10

  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  model.save('my_model.h5') 
  return model,class_names

train_model()

  