import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

BATCH_SIZE = 256
NOISE_DIM = 100


def make_generator_model(n_classes=10):
    # Create class embedding channel
    input_label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(n_classes, 50)(input_label)
    upscaling = tf.keras.layers.Dense(7*7)(label_embedding)
    upscaling = tf.keras.layers.Reshape((7, 7, 1))(upscaling)

    # create seed encoding network
    seed_input = tf.keras.layers.Input(shape=(NOISE_DIM,))
    seed_fc = tf.keras.layers.Dense(7*7*256, use_bias=False)(seed_input)
    seed_fc = tf.keras.layers.BatchNormalization()(seed_fc)
    seed_fc = tf.keras.layers.LeakyReLU()(seed_fc)
    seed_fc = tf.keras.layers.Reshape((7, 7, 256))(seed_fc)

    # merge embedding with seed encoder
    merge = tf.keras.layers.Concatenate()([seed_fc, upscaling])
    assert tuple(merge.shape) == (None, 7, 7, 257)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    assert tuple(x.shape) == (None, 7, 7, 128)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    assert tuple(x.shape) == (None, 14, 14, 64)

    output = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    model = tf.keras.Model([seed_input, input_label], output)
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model(n_classes=10):
    # Create a class embedding
    input_label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(n_classes, 50)(input_label)
    upscaling = tf.keras.layers.Dense(28*28)(label_embedding)
    upscaling = tf.keras.layers.Reshape((28, 28, 1))(upscaling)

    # merge image with class embedding
    input_image = tf.keras.layers.Input(shape=(28, 28, 1))
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


def generator_loss(predictions):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(predictions), predictions)


def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(test_labels[i][0]))

    plt.tight_layout()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()



if __name__ == "__main__":

    # Get ground truth data
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    BUFFER_SIZE = 60000
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels.reshape((-1, 1)))).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    # Define Model
    generator = make_generator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator = make_discriminator_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Model Checkpoints
    # checkpoint_dir = "./training_checkpoints"
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                  discriminator_optimizer=discriminator_optimizer,
    #                                  generator=generator,
    #                                  discriminator=discriminator)

    # Create Training Pipeline
    EPOCHS = 500
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
    seed_labels = np.random.randint(0, 10, num_examples_to_generate).reshape((-1, 1))
    # fakes = generator([seed, labels], training=False)
    # plt.imshow(fakes[0, :, :, 0], cmap='gray')
    generate_and_save_images(generator, 1, seed, seed_labels)
    # plt.show()

    @tf.function
    def train_step(images, ground_truth_labels):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        random_labels = np.random.randint(0, 10, BATCH_SIZE).reshape((-1, 1))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fakes = generator([noise, random_labels], training=True)
            ground_truth_preds = discriminator([images, ground_truth_labels], training=True)
            fake_preds = discriminator([fakes, random_labels], training=True)

            gen_loss = generator_loss(fake_preds)
            disc_loss = discriminator_loss(real_output=ground_truth_preds, fake_output=fake_preds)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
            print("Starting epoch")
            start = time.time()
            for (image_batch, labels_batch) in dataset:
                train_step(image_batch, labels_batch)

            generate_and_save_images(generator, epoch+1, seed, seed_labels)
            # if (epoch + 1) % 10 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))

    train(train_dataset, EPOCHS)

