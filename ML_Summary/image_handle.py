import tensorflow as tf
import random
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_path = pathlib.Path('/Users/zhuyongqi/Downloads/coil-20-proc')
all_image_paths = list(data_path.glob('*/*'))  
all_image_paths = [str(path) for path in all_image_paths]


#random.shuffle(all_image_paths) 
image_count = len(all_image_paths)
#print(image_count)

#分配标签
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
#分配标签索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
#为每个文件分配索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

#print("First 10 labels indices: ", all_image_labels[:10])

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [128, 128])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)  

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

