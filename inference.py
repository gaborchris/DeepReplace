import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt

BATCH_SIZE = 8

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from normalized_embed import make_discriminator_model, make_generator_model, num_classes, NOISE_DIM, \
    generate_and_save_images, save_defaults, num_to_char


def show_before_after(source, target, target_label, subscript):
    num_comps = source.shape[0]
    for i in range(num_comps):
        plt.subplot(num_comps/2, 4, 2*i + 1)
        plt.imshow(source[i, :, :, :] / 2. + 0.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("source")

        plt.subplot(num_comps/2, 4, 2*i+2)
        plt.imshow(target[i, :, :, :] / 2. + 0.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(num_to_char[target_label[i][0]])
    plt.tight_layout()
    plt.savefig('./outputs/{:02d}.png'.format(subscript))
    plt.close()


if __name__ == "__main__":
    data_dir = tf.keras.utils.get_file('English',
                                       origin='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz',
                                       untar=True, extract=True)
    data_dir = pathlib.Path(data_dir)
    image_path = os.path.join(data_dir, "Img/GoodImg/Bmp")

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=lambda img: (img - 127.5) / 127.5)
    image_ground_truth = image_generator.flow_from_directory(image_path, target_size=(28, 28),
                                                             batch_size=BATCH_SIZE,
                                                             class_mode='sparse')

    generator = make_generator_model(n_classes=num_classes)
    generator_optimizer = tf.keras.optimizers.Adam(6e-4)
    discriminator = make_discriminator_model(n_classes=num_classes)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint_dir = '/home/christian/checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # generator.summary()

    num_examples_to_generate = BATCH_SIZE
    num_images_to_generate = 100
    for i in range(num_images_to_generate):
        seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
        seed_labels = np.random.randint(0, num_classes, num_examples_to_generate).reshape((-1, 1))
        images, labels = next(image_ground_truth)
        replace_images = images[:num_examples_to_generate, :, :, :]
        predictions = generator([seed, seed_labels, replace_images], training=False)
        show_before_after(replace_images, predictions, seed_labels, i)


