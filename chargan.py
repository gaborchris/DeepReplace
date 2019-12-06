import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib

BATCH_SIZE = 128
NOISE_DIM = 100

num_to_char = {}
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for i in range(0, 10):
    num_to_char[i] = str(i)
for i in range(0, 26):
    num_to_char[i+10] = alphabet[i].upper()
for i in range(0, 26):
    num_to_char[i+10+26] = alphabet[i].lower()


def make_generator_model(n_classes=10):
    # Create class embedding channel
    input_label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(n_classes, 50)(input_label)
    upscaling = tf.keras.layers.Dense(7*7*1)(label_embedding)
    upscaling = tf.keras.layers.Reshape((7, 7, 1))(upscaling)

    # create seed encoding network
    seed_input = tf.keras.layers.Input(shape=(NOISE_DIM,))
    seed_fc = tf.keras.layers.Dense(7*7*256, use_bias=False)(seed_input)
    seed_fc = tf.keras.layers.BatchNormalization()(seed_fc)
    seed_fc = tf.keras.layers.LeakyReLU()(seed_fc)
    seed_fc = tf.keras.layers.Reshape((7, 7, 256))(seed_fc)

    # merge embedding with seed encoder
    merge = tf.keras.layers.Concatenate()([seed_fc, upscaling])
    assert tuple(merge.shape) == (None, 7, 7, 256+1)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    assert tuple(x.shape) == (None, 7, 7, 128)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    assert tuple(x.shape) == (None, 14, 14, 64)

    # TODO tanh function with output between [-1, 1]
    output = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    model = tf.keras.Model([seed_input, input_label], output)
    assert model.output_shape == (None, 28, 28, 3)

    return model


def make_discriminator_model(n_classes=10):
    # Create a class embedding
    input_label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(n_classes, 50)(input_label)
    upscaling = tf.keras.layers.Dense(28*28)(label_embedding)
    upscaling = tf.keras.layers.Reshape((28, 28, 1))(upscaling)

    # merge image with class embedding
    input_image = tf.keras.layers.Input(shape=(28, 28, 3))
    merge = tf.keras.layers.Concatenate()([input_image, upscaling])

    # define classification architecture
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merge)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model([input_image, input_label], output)

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def discriminator_loss_real(real_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return real_loss


def discriminator_loss_fake(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss


def generator_loss(predictions):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(predictions), predictions)


def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] / 2. + 0.5)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(num_to_char[test_labels[i][0]])

    plt.tight_layout()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def image_norm(img):
  return (img - 127.5) / 127.5

# Get ground truth data
data_dir = tf.keras.utils.get_file('English',
                                    origin='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz',
                                    untar=True, extract=True)
data_dir = pathlib.Path(data_dir)
image_path = os.path.join(data_dir, "Img/GoodImg/Bmp")

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=lambda img: (img - 127.5) / 127.5)
image_ground_truth = image_generator.flow_from_directory(image_path, target_size=(28, 28),
                                                          batch_size=BATCH_SIZE,
                                                          class_mode='sparse')

# Define Model
num_classes = 62

generator = make_generator_model(n_classes=num_classes)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator = make_discriminator_model(n_classes=num_classes)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
seed_labels = np.random.randint(0, num_classes, num_examples_to_generate).reshape((-1, 1))


# Define training procedure
@tf.function
def train_step(images, ground_truth_labels):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    random_labels = np.random.randint(0, num_classes, BATCH_SIZE).reshape((-1, 1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fakes = generator([noise, random_labels], training=True)
        ground_truth_preds = discriminator([images, ground_truth_labels], training=True)
        fake_preds = discriminator([fakes, random_labels], training=True)

        gen_loss = generator_loss(fake_preds)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        disc_loss = discriminator_loss(real_output=ground_truth_preds, fake_output=fake_preds)
      
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def train_step_alt(images, ground_truth_labels):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    random_labels = np.random.randint(0, num_classes, BATCH_SIZE).reshape((-1, 1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fakes = generator([noise, random_labels], training=True)
        ground_truth_preds = discriminator([images, ground_truth_labels], training=True)
        fake_preds = discriminator([fakes, random_labels], training=True)

        '''
        gen_loss = generator_loss(fake_preds)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        '''

        disc_loss = discriminator_loss(real_output=ground_truth_preds, fake_output=fake_preds)
      
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, ckpt_prefix):
    print("Starting training")
    for epoch in range(epochs):
        start = time.time()
        i = 0
        for (image_batch, labels_batch) in dataset:
            train_step(image_batch[:int(BATCH_SIZE/2)], labels_batch[:int(BATCH_SIZE/2)])
            train_step_alt(image_batch[int(BATCH_SIZE/2):], labels_batch[int(BATCH_SIZE/2):])
            i += 1
            if i > len(dataset):
              break
              
        if (epoch + 1) % 20 == 0:
            generate_and_save_images(generator, epoch+1, seed, seed_labels)  
            checkpoint.save(file_prefix=ckpt_prefix)
            try:
                files.download('image_at_epoch_{:04d}.png'.format(epoch+1))
            except Exception as e:
                print(e)
                    
        print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Setup repeated predictions
fakes = generator([seed, seed_labels], training=False)
generate_and_save_images(generator, 0, seed, seed_labels)

train(image_ground_truth, epochs=2000, ckpt_prefix=checkpoint_prefix)
