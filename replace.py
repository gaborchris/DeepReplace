import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt

BATCH_SIZE = 4

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from softmax import make_softmax_model, make_discriminator_model, make_generator_model, num_classes, NOISE_DIM, \
    generate_and_save_images, save_defaults, num_to_char

GAMMA = 5e-2
GAMMA_DECAY = 1.0
G_LR = 6e-4
D_LR = 1e-4
SM_LR = 1e-5

BLUR_WIDTH = 1
FILTER_SIZE = 3 #must be odd integer

num_to_char = {}
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for i in range(0, 10):
    num_to_char[i] = str(i)
for i in range(0, 26):
    num_to_char[i + 10] = alphabet[i].upper()
for i in range(0, 26):
    num_to_char[i + 10 + 26] = alphabet[i].lower()


image_path = os.path.join("/home/christian/Desktop/test_images/", "1")

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
	preprocessing_function=lambda img: (img - 127.5) / 127.5)
image_ground_truth = image_generator.flow_from_directory(image_path, target_size=(28, 28),
														 batch_size=BATCH_SIZE,
														 class_mode='sparse')


generator = make_generator_model(n_classes=num_classes)
generator_optimizer = tf.keras.optimizers.RMSprop(G_LR)
discriminator = make_discriminator_model(n_classes=num_classes)
discriminator_optimizer = tf.keras.optimizers.Adam(D_LR)
softmax_discriminator = make_softmax_model(n_classes=num_classes)
softmax_optimizer = tf.keras.optimizers.Adam(SM_LR)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
								 discriminator_optimizer=discriminator_optimizer,
								 generator=generator,
								 discriminator=discriminator)

checkpoint_dir = '/home/christian/checkpoints/softmax/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

num_examples_to_generate = 1
num_images_to_generate = 5
imgs = []
labels = np.array([12, 28, 2, 3, 0])
print(labels.shape)
for i in range(num_images_to_generate):
	seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
	print(labels[i])
	seed_labels = np.array([[labels[i]]])
	images, _ = next(image_ground_truth)
	print(images.shape)
	# predictions = generator([seed, seed_labels, images], training=False)
	predictions = images[0]
	imgs.append(predictions[0])

# for img in output:
# 	plt.imshow(img)
# 	plt.show()
# 	plt.close()

# s = input('Desired string: ')
s = 'cs230'
# img = mpimg.imread(input('Path to style image: '))
#
#
#
# imgs = list()
# for c in s:
# 	TODO: append image produced by generator(c, img)
	# label = num_to_char[c]
	# imgs.append(img)

h, w, _ = imgs[0].shape
new_img = np.zeros((h, w*len(imgs), 3))

for i in range(len(imgs)):
	new_img[0:h, i*w:(i+1)*w] = imgs[i]

for i in range(1, len(imgs)):
	new_img[0:h, i*w-BLUR_WIDTH:i*w+BLUR_WIDTH] = cv2.GaussianBlur(
		new_img[0:h, i*w-BLUR_WIDTH:i*w+BLUR_WIDTH], (FILTER_SIZE, FILTER_SIZE), cv2.BORDER_DEFAULT)

plt.imshow(new_img)
plt.show()

