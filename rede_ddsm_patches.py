"""Rede construída com a arquitetura DiaGRAM modificada de modo a conter apenas as partes treinadas pelos recortes das imagens"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, ReLU, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from IPython import display
matplotlib.use("Agg")


# ------------------------------------------------------ MODELS ------------------------------------------------------ #
def make_generator_model():
    model = Sequential()
    model.add(Dense(16*16*64, use_bias=False, input_shape=(100,)))

    model.add(Reshape((16, 16, 64)))

    model.add(Conv2DTranspose(32, kernel_size=5, strides=1, padding='valid', use_bias=False))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(16, kernel_size=5, strides=1, padding='valid', use_bias=False))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(8, kernel_size=5, strides=1, padding='valid', use_bias=False))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(3, kernel_size=5, strides=1, padding='valid', use_bias=False))
    model.add(ReLU())
    model.add(BatchNormalization())

    return model


def make_feature_extraction_model():
    feat_ext_input_patches = keras.Input(shape=(None, None, 3))

    hidden_patches = Conv2D(8, kernel_size=5, strides=1, padding='valid')(feat_ext_input_patches)
    hidden_patches = LeakyReLU(alpha=0.15)(hidden_patches)
    hidden_patches = BatchNormalization()(hidden_patches)
    hidden_patches = Dropout(0.2)(hidden_patches)

    hidden_patches = Conv2D(16, kernel_size=5, strides=1, padding='valid')(hidden_patches)
    hidden_patches = LeakyReLU(alpha=0.15)(hidden_patches)
    hidden_patches = BatchNormalization()(hidden_patches)
    hidden_patches = Dropout(0.2)(hidden_patches)

    hidden_patches = Conv2D(32, kernel_size=5, strides=1, padding='valid')(hidden_patches)
    hidden_patches = LeakyReLU(alpha=0.15)(hidden_patches)
    hidden_patches = BatchNormalization()(hidden_patches)
    hidden_patches = Dropout(0.2)(hidden_patches)

    hidden_patches = Conv2D(64, kernel_size=5, strides=1, padding='valid')(hidden_patches)
    hidden_patches = LeakyReLU(alpha=0.15)(hidden_patches)
    hidden_patches = BatchNormalization()(hidden_patches)
    output_patches = Dropout(0.2)(hidden_patches)

    feature_extraction = keras.Model(inputs=feat_ext_input_patches, outputs=output_patches, name='feature_extraction')
    return feature_extraction


def make_classifier_model():
    inputs = keras.Input(shape=(16, 16, 64))

    flatter = Flatten()(inputs)
    dense_softmax = Dense(1, use_bias=False, activation='sigmoid')(flatter)  # binary labeling

    classifier = keras.Model(inputs=inputs, outputs=dense_softmax, name='classifier')
    return classifier


def make_discriminator_model():
    inputs = keras.Input(shape=(16, 16, 64))
    flatter = Flatten()(inputs)
    dense_sigmoid = Dense(1, use_bias=False, activation='sigmoid')(flatter)

    discriminator = keras.Model(inputs=inputs, outputs=dense_sigmoid, name='discriminator')
    return discriminator


# ------------------------------------------------------ LOSSES ------------------------------------------------------ #
def generator_loss(fake_output):
    cross_entropy = BinaryCrossentropy(from_logits=False)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def classifier_loss(real_output_class, labels):
    cross_entropy = BinaryCrossentropy(from_logits=False)
    entropy = cross_entropy(labels, real_output_class)
    return entropy


def discriminator_loss(real_output, fake_output):
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def feature_extractor_loss(class_loss, disc_loss):
    return class_loss + disc_loss


# ------------------------------------------------------- UTIL ------------------------------------------------------- #
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    pos_cal = std != 0
    X[:, pos_cal] = (X[:, pos_cal] - mean[pos_cal])/std[pos_cal]
    return X, mean, std


def convert_inputs(parent):
    X = []
    list_names = []
    for file_name in os.listdir(parent):
        img = load_img("".join((parent, "/", file_name)), color_mode='rgb')
        img_array = img_to_array(img)
        X.append(img_array)
        list_names.append(file_name)
    return np.asarray(X), list_names


def organize(array_images, list_names, labels_ds):
    indexes = np.zeros(len(list_names), dtype=np.int64)
    for i in range(len(list_names)):
        name = list_names[i]
        pos = labels_ds['NAME'] == name
        idx = labels_ds[pos].index
        for j in idx:
            indexes[i] = j
    assert max(indexes) < len(array_images)
    return array_images[indexes]


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig('generated_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def accuracy(X_set, y_set):
    correct = 0
    num_img = 0
    for image_batch, label_batch in zip(X_set, y_set):
        num_img += len(label_batch)
        features = feature_extractor.predict(image_batch)
        img_class = classifier.predict(features)
        diff = abs(np.array(label_batch - img_class))
        array_correct = np.less(diff, 0.5*np.ones_like(diff))
        correct += np.sum(array_correct)

    return correct/num_img * 100


def setup_labels_dataset(ds):
    pos_mal = np.where(ds == 'MALIGNANT')
    col = np.zeros_like(ds, dtype=np.float64)
    col[pos_mal] = 1.0  # malignant==1, benign==0

    return col


def plot_history(train_history, test_history, metric, nome_plot="graph"):
    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history)
    plt.plot(epochs, test_history)
    plt.title('Training and test ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'test_'+metric])
    plt.grid(True)
    if nome_plot != 'nada':
        plt.savefig(nome_plot, bbox_inches='tight')


def setup_final_dataset(dset):
    ds = tf.data.Dataset.from_tensor_slices(dset).shuffle(buffer_size=BUFFER_SIZE, seed=1234).batch(BATCH_SIZE)
    return ds


# ------------------------------------------------------- PREP ------------------------------------------------------- #
folder_patches = '/path/to/patches/folder'
X_patches, names = convert_inputs(folder_patches)
labels_ds_patches = pd.read_csv("ddsm_labels.csv", usecols=['NAME', 'PATHOLOGY', 'LESION'])

labels_ds_patches = labels_ds_patches.drop(labels_ds_patches[labels_ds_patches['LESION'] == 'NONE'].index)
labels_ds_patches = labels_ds_patches.reset_index(drop=True)
X_patches = organize(X_patches, names, labels_ds_patches)

pos_unproven = labels_ds_patches[labels_ds_patches['PATHOLOGY'] == 'UNPROVEN'].index
X_patches = np.delete(X_patches, pos_unproven, axis=0)
labels_ds_patches = labels_ds_patches.drop(pos_unproven)

labels_patches = setup_labels_dataset(labels_ds_patches.drop(['NAME', 'LESION'], axis=1))

datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)  # DATA AUGMENTATION

X_patches, mean, std = normalize(X_patches)
X_train_patches, X_test_patches, y_train_patches, y_test_patches = train_test_split(X_patches, labels_patches, test_size=0.2)
y_train_patches, y_test_patches = tf.convert_to_tensor(y_train_patches, dtype=tf.int32), tf.convert_to_tensor(y_test_patches, dtype=tf.int32)


BUFFER_SIZE = 60000
BATCH_SIZE = 64

dataset_X_train_patches = setup_final_dataset(X_train_patches)
dataset_y_train_patches = setup_final_dataset(y_train_patches)
dataset_X_test_patches = setup_final_dataset(X_test_patches)
dataset_y_test_patches = setup_final_dataset(y_test_patches)


generator = make_generator_model()
feature_extractor = make_feature_extraction_model()
classifier = make_classifier_model()
discriminator = make_discriminator_model()

generator_optimizer = Adam(0.0001)
classifier_optimizer = Adam(0.0001)
discriminator_optimizer = Adam(0.0001)
feature_extractor_optimizer = Adam(0.0001)
# ------------------------------------------------------ TRAIN ------------------------------------------------------ #
EPOCHS = 300
noise_dim = 100
num_examples_to_generate = 25
seed = tf.random.uniform(shape=[num_examples_to_generate, noise_dim], minval=-1, maxval=1)


@tf.function
def train_step(patches, labels_patches):
    noise = tf.random.uniform([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as disc_tape, tf.GradientTape() as class_tape, tf.GradientTape() as feat_ext_tape:
        generated_images = generator(noise, training=True)

        real_output = feature_extractor(tfa.image.blend(generated_images[:patches.shape[0],:,:,:], patches, 0.9), training=True)
        real_output_class = classifier(real_output, training=True)
        real_output = discriminator(real_output, training=True)
        fake_output = feature_extractor(generated_images, training=True)
        fake_output = discriminator(fake_output, training=True)

        class_loss = classifier_loss(real_output_class, labels_patches)
        disc_loss = discriminator_loss(real_output, fake_output)
        feat_ext_loss = feature_extractor_loss(class_loss, disc_loss)

    classifier_gradients = class_tape.gradient(class_loss, classifier.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    feature_extractor_gradients = feat_ext_tape.gradient(feat_ext_loss, feature_extractor.trainable_variables)

    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    feature_extractor_optimizer.apply_gradients(zip(feature_extractor_gradients, feature_extractor.trainable_variables))


def train_step_generator():
    noise = tf.random.uniform([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = feature_extractor(generated_images, training=True)
        fake_output = discriminator(fake_output, training=True)

        gen_loss = generator_loss(fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        count = 0
        start = time.time()

        for image_batch_p, label_batch_p in datagen.flow(X_train_patches, y_train_patches, batch_size=BATCH_SIZE):
            train_step(image_batch_p, label_batch_p)
            count += 1
            if count >= 6 * len(X_train_patches):
                break
        train_step_generator()

        display.clear_output(wait=True)
        train_acc_p = accuracy(dataset_X_train_patches, dataset_y_train_patches)
        test_acc_p = accuracy(dataset_X_test_patches, dataset_y_test_patches)
        train_history_p.append(train_acc_p), test_history_p.append(test_acc_p)
        print("Foi obtida uma acuracia de treino nos recortes de", train_acc_p, "% na epoca", epoch + 1)
        print("Foi obtida uma acuracia de teste nos recortes de", test_acc_p, "% na epoca", epoch + 1)
        generate_and_save_images(generator, epoch + 1, seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


train_history_p = []
test_history_p = []
train(EPOCHS)
# ------------------------------------------------------- PLOT ------------------------------------------------------- #
print("Foi obtida uma acuracia final nos recortes de", accuracy(dataset_X_train_patches, dataset_y_train_patches),
      "% no conjunto de treino")
print("Foi obtida uma acuracia final nos recortes de", accuracy(dataset_X_test_patches, dataset_y_test_patches),
      "% no conjunto de teste")


plot_history(train_history_p, test_history_p, 'accuracy', nome_plot="acuracias_treino_teste_patches.pdf")
