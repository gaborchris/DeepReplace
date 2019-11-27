import tarfile
import tensorflow as tf
import pathlib
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from char_dict import num_to_char
print(num_to_char)

data_dir = tf.keras.utils.get_file('English', origin='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz',
                                   untar=True, extract=True)
data_dir = pathlib.Path(data_dir)
image_path = os.path.join(data_dir, "Img/GoodImg/Bmp")

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(image_path, target_size=(64, 64))

for image_batch, label_batch in image_data:
    print("shape:", image_batch.shape)
    print("label shape:", label_batch.shape)
    break

for n in range(16):
    plt.subplot(4, 4, n+1)
    plt.imshow(image_batch[n])
    plt.xlabel(num_to_char[np.argmax(label_batch[n])])

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    pass