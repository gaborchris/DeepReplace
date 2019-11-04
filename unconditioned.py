import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as pyplot


def define_discriminator(input_shape=(28, 28, 1)):
   model = tf.keras.Sequential()
   model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
   model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
   model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
   model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dropout(0.4))
   model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
   opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
   model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
   return model


def define_generator(latent_dim):
   model = tf.keras.Sequential()
   n_nodes = 128*7*7
   model.add(tf.keras.layers.Dense(n_nodes, input_dim=latent_dim))
   model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
   model.add(tf.keras.layers.Reshape((7, 7, 128)))
   model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same'))
   model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
   model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same'))
   model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
   model.add(tf.keras.layers.Conv2D(1, (7, 7), activation='tanh', padding='same'))
   return model


def define_gan(generator, discriminator):
   discriminator.trainable = False
   model = tf.keras.models.Sequential()
   model.add(generator)
   model.add(discriminator)
   opt = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
   model.compile(loss='binary_crossentropy', optimizer=opt)
   return model


def load_real_samples():
   (train_x, train_y), (_, _) = load_data()
   X = np.expand_dims(train_x, axis=-1)
   X = X.astype('float32')
   X = (X - 127.5) / 127.5
   return [X, train_y]


def generate_real_samples(dataset, n_samples):
   images, labels = dataset
   ix = np.random.randint(0, images.shape[0], n_samples)
   X, labels = images[ix], labels[ix]
   y = np.ones((n_samples, 1))
   return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=10):
   x_input = np.random.randn(latent_dim * n_samples)
   z_input = x_input.reshape(n_samples, latent_dim)
   labels = np.random.randint(0, n_classes, n_samples)
   return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
   z_input, labels_input = generate_latent_points(latent_dim, n_samples)
   images = generator.predict([z_input, labels_input])
   y = np.zeros((n_samples, 1))
   return [images, labels_input], y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
   bat_per_epo = int(dataset[0].shape[0] / n_batch)
   half_batch = int(n_batch / 2)
   # manually enumerate epochs
   for i in range(n_epochs):
      # enumerate batches over the training set
      for j in range(bat_per_epo):
         # get randomly selected 'real' samples
         [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
         # update discriminator model weights
         d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
         # generate 'fake' examples
         [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
         # update discriminator model weights
         d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
         # prepare points in latent space as input for the generator
         [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
         # create inverted labels for the fake samples
         y_gan = np.ones((n_batch, 1))
         # update the generator via the discriminator's error
         g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
         # summarize loss on this batch
         print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
               (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
   # save the generator model
   g_model.save('cgan_generator.h5')


def show_data(images):
   for i in range(100):
      pyplot.subplot(10, 10, 1 +i)
      pyplot.axis('off')
      pyplot.imshow(images[i], cmap='gray_r')
   pyplot.show()


# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
   # plot images
   for i in range(n * n):
      # define subplot
      pyplot.subplot(n, n, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
   pyplot.show()


if __name__ == "__main__":
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   (train_x, train_y), (test_x, test_y) = load_data()
   print("Train", train_x.shape, train_y.shape)
   print("Train", test_x.shape, test_y.shape)
   latent_dim = 100
   d_model = define_discriminator()
   g_model = define_generator(latent_dim)
   gan_model = define_gan(g_model, d_model)
   dataset = load_real_samples()
   train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128)

   # load model
   model = tf.keras.models.load_model('cgan_generator.h5')
   # generate images
   latent_points = generate_latent_points(100, 100)
   # generate images
   X = model.predict(latent_points)
   # plot the result
   show_plot(X, 10)







