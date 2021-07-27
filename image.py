import tensorflow as tf
import numpy as np

file = tf.keras.utils.get_file(
    "door.jpg",
    "https://i.pinimg.com/564x/66/9f/52/669f52d7ee8a0dcf27d874a7134fa490.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])


labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
labels = np.array(open(labels_path).read().splitlines())


model = tf.keras.applications.MobileNetV2()
predictions = model(x)

top_5_classes_index = np.argsort(predictions)[0, ::-1][:5]+1

top_5_classes = labels[top_5_classes_index]
print(top_5_classes)