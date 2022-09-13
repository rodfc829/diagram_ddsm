"""Rede construída com a arquitetura DiaGRAM modificada de modo a retirar sua parte generativa"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, LeakyReLU, ReLU, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from IPython import display
matplotlib.use("Agg")


# ------------------------------------------------------ MODELS ------------------------------------------------------ #
def make_feature_extraction_model():
    feat_ext_input_patches = keras.Input(shape=(None, None, 3))  # (32,32,1) || (256,256,1)

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
    dense_softmax = Dense(1, use_bias=False, activation='sigmoid')(flatter)

    classifier = keras.Model(inputs=inputs, outputs=dense_softmax, name='classifier')
    return classifier


def make_discriminator_model():
    inputs = keras.Input(shape=(16, 16, 64))
    flatter = Flatten()(inputs)
    dense_sigmoid = Dense(1, use_bias=False, activation='sigmoid')(flatter)

    discriminator = keras.Model(inputs=inputs, outputs=dense_sigmoid, name='discriminator')
    return discriminator


def make_extended_classifier_model():
    ext_class_input = keras.Input(shape=(240, 240, 64))

    # RESNET BLOCK 1
    input_block1 = Conv2D(128, kernel_size=5, strides=2, padding='same')(ext_class_input)
    hidden = BatchNormalization()(input_block1)
    hidden = ReLU()(hidden)
    # shape==(120,120,128)

    hidden = Conv2D(128, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(120,120,128)

    hidden = Conv2D(128, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(120,120,128)

    hidden_short = Conv2D(128, kernel_size=5, strides=2, padding='same')(ext_class_input)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(120,120,128)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b1 = Dropout(0.2)(added)
    # -------------------------------------------------------------------

    # RESNET BLOCK 2
    input_block2 = Conv2D(256, kernel_size=5, strides=2, padding='same')(out_b1)
    hidden = BatchNormalization()(input_block2)
    hidden = ReLU()(hidden)
    # shape==(60,60,256)

    hidden = Conv2D(256, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(60,60,256)

    hidden = Conv2D(256, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(60,60,256)

    hidden_short = Conv2D(256, kernel_size=5, strides=2, padding='same')(out_b1)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(60,60,256)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b2 = Dropout(0.2)(added)
    # -------------------------------------------------------------------

    # RESNET BLOCK 3
    input_block3 = Conv2D(512, kernel_size=5, strides=2, padding='same')(out_b2)
    hidden = BatchNormalization()(input_block3)
    hidden = ReLU()(hidden)
    # shape==(30,30,512)

    hidden = Conv2D(512, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(30,30,512)

    hidden = Conv2D(512, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(30,30,512)

    hidden_short = Conv2D(512, kernel_size=5, strides=2, padding='same')(out_b2)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(30,30,512)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b3 = Dropout(0.1)(added)
    # -------------------------------------------------------------------

    # RESNET BLOCK 4
    input_block4 = Conv2D(1024, kernel_size=5, strides=2, padding='same')(out_b3)
    hidden = BatchNormalization()(input_block4)
    hidden = ReLU()(hidden)
    # shape==(15,15,1024)

    hidden = Conv2D(1024, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(15,15,1024)

    hidden = Conv2D(1024, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(15,15,1024)

    hidden_short = Conv2D(1024, kernel_size=5, strides=2, padding='same')(out_b3)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(15,15,1024)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b4 = Dropout(0.1)(added)
    # -------------------------------------------------------------------

    # RESNET BLOCK 5
    input_block5 = Conv2D(2048, kernel_size=5, strides=2, padding='same')(out_b4)
    hidden = BatchNormalization()(input_block5)
    hidden = ReLU()(hidden)
    # shape==(8,8,2048)

    hidden = Conv2D(2048, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(8,8,2048)

    hidden = Conv2D(2048, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(8,8,2048)

    hidden_short = Conv2D(2048, kernel_size=5, strides=2, padding='same')(out_b4)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(8,8,2048)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b5 = Dropout(0.1)(added)
    # -------------------------------------------------------------------

    # RESNET BLOCK 6
    input_block6 = Conv2D(4096, kernel_size=5, strides=2, padding='same')(out_b5)
    hidden = BatchNormalization()(input_block6)
    hidden = ReLU()(hidden)
    # shape==(4,4,4096)

    hidden = Conv2D(4096, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    # shape==(4,4,4096)

    hidden = Conv2D(4096, kernel_size=5, strides=1, padding='same')(hidden)
    hidden = BatchNormalization()(hidden)
    # shape==(4,4,4096)

    hidden_short = Conv2D(4096, kernel_size=5, strides=2, padding='same')(out_b5)
    hidden_short = BatchNormalization()(hidden_short)
    # shape==(4,4,4096)

    added = Add()([hidden, hidden_short])
    added = ReLU()(added)
    out_b6 = Dropout(0.1)(added)
    # -------------------------------------------------------------------

    pool = GlobalAveragePooling2D()(out_b6)
    flatter = Flatten()(pool)
    dense_softmax = Dense(1, use_bias=False, activation='sigmoid')(flatter)

    extended_classifier = keras.Model(inputs=ext_class_input, outputs=dense_softmax, name='extended_classifier')
    return extended_classifier


# ------------------------------------------------------ LOSSES ------------------------------------------------------ #
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


def extended_classifier_loss(whole_output_class, labels):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    entropy = cross_entropy(labels, whole_output_class)
    return entropy


# ------------------------------------------------------- UTIL ------------------------------------------------------- #
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    pos_cal = std != 0
    X[:, pos_cal] = (X[:, pos_cal] - mean[pos_cal])/std[pos_cal]
    return X, mean, std


def convert_inputs(parent):
    X = []
    for file_name in os.listdir(parent):
        img = load_img("".join((parent, "/", file_name)), color_mode='rgb')
        img_array = img_to_array(img)
        X.append(img_array)
    return np.asarray(X)


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


def accuracy(X_set, y_set):
    correct = 0
    num_img = 0
    for image_batch, label_batch in zip(X_set, y_set):
        num_img += len(label_batch)
        features = feature_extractor.predict(image_batch)
        img_class = classifier.predict(features)
        diff = np.array(label_batch - img_class)
        array_pos = np.where(np.greater(diff, np.zeros_like(diff)))
        array_correct = np.less(diff[array_pos], 0.5*np.ones_like(diff[array_pos]))
        correct += np.sum(array_correct)

    return correct/num_img * 100


def setup_labels_dataset(ds):  # malignant==1, benign==0
    pos_mal = np.where(ds == 'MALIGNANT')
    col = np.zeros_like(ds, dtype=np.float64)
    col[pos_mal] = 1.0

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


def load_dataset_whole(ds_name):
    ds = pd.read_csv(ds_name, usecols=['pathology', 'image file path'])
    ds = ds.drop_duplicates(subset=['image file path'])
    ds.drop('image file path', axis=1, inplace=True)
    return ds


def setup_final_dataset(set):
    ds = tf.data.Dataset.from_tensor_slices(set).shuffle(buffer_size=BUFFER_SIZE, seed=1234).batch(BATCH_SIZE)
    return ds


# ------------------------------------------------------- PREP ------------------------------------------------------- #
folder_train_patches = '/path/to/folder/with/train/patches'
folder_test_patches = '/path/to/folder/with/test/patches'
folder_train_whole = '/path/to/folder/with/train/whole/images'
folder_test_whole = '/path/to/folder/with/test/whole/images'

X_train_patches, X_test_patches = convert_inputs(folder_train_patches), convert_inputs(folder_test_patches)
X_train_whole, X_test_whole = convert_inputs(folder_train_whole), convert_inputs(folder_test_whole)

mass_train_labels_whole = load_dataset_whole("mass_case_description_train_set.csv")
calc_train_labels_whole = load_dataset_whole("calc_case_description_train_set.csv")
mass_test_labels_whole = load_dataset_whole("mass_case_description_test_set.csv")
calc_test_labels_whole = load_dataset_whole("calc_case_description_test_set.csv")

mass_train_labels_patches = pd.read_csv("mass_case_description_train_set.csv", usecols=['pathology'])
calc_train_labels_patches = pd.read_csv("calc_case_description_train_set.csv", usecols=['pathology'])
mass_test_labels_patches = pd.read_csv("mass_case_description_test_set.csv", usecols=['pathology'])
calc_test_labels_patches = pd.read_csv("calc_case_description_test_set.csv", usecols=['pathology'])

y_train_patches = setup_labels_dataset(mass_train_labels_patches, calc_train_labels_patches)
y_test_patches = setup_labels_dataset(mass_test_labels_patches, calc_test_labels_patches)
y_train_whole = setup_labels_dataset(mass_train_labels_whole, calc_train_labels_whole)
y_test_whole = setup_labels_dataset(mass_test_labels_whole, calc_test_labels_whole)

datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)  # DATA AUGMENTATION

X_train_patches, mean1, std1 = normalize(X_train_patches)
X_test_patches, mean2, std2 = normalize(X_test_patches)
X_train_whole, mean3, std3 = normalize(X_train_whole)
X_test_whole, mean4, std4 = normalize(X_test_whole)


BUFFER_SIZE = 60000  # ajustar
BATCH_SIZE = 128

dataset_X_train_patches = setup_final_dataset(X_train_patches)
dataset_y_train_patches = setup_final_dataset(y_train_patches)
dataset_X_test_patches = setup_final_dataset(X_test_patches)
dataset_y_test_patches = setup_final_dataset(y_test_patches)

dataset_X_train_whole = setup_final_dataset(X_train_whole)
dataset_y_train_whole = setup_final_dataset(y_train_whole)
dataset_X_test_whole = setup_final_dataset(X_test_whole)
dataset_y_test_whole = setup_final_dataset(y_test_whole)


feature_extractor = make_feature_extraction_model()
classifier = make_classifier_model()
discriminator = make_discriminator_model()
extended_classifier = make_extended_classifier_model()

feature_extractor_optimizer = Adam(1e-4)
classifier_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)
extended_classifier_optimizer = Adam(1e-4)


# ------------------------------------------------------ TRAIN ------------------------------------------------------ #
EPOCHS = 300
noise_dim = 100
num_examples_to_generate = 25
seed = tf.random.uniform(shape=[num_examples_to_generate, noise_dim], minval=-1, maxval=1)


@tf.function
def train_step(patches, whole, labels_patches, labels_whole):
    with tf.GradientTape() as disc_tape, tf.GradientTape() as class_tape, tf.GradientTape() as ext_class_tape:

        real_output = feature_extractor(patches, training=True)
        real_output_class = classifier(real_output, training=True)
        real_output = discriminator(real_output, training=True)

        whole_output = feature_extractor(whole, training=True)
        whole_output_class = extended_classifier(whole_output, training=True)

        class_loss = classifier_loss(real_output_class, labels_patches)
        disc_loss = discriminator_loss(real_output, fake_output)
        feat_ext_loss = feature_extractor_loss(class_loss, disc_loss)
        ext_class_loss = extended_classifier_loss(whole_output_class, labels_whole)

    classifier_gradients = class_tape.gradient(class_loss, classifier.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    feature_extractor_gradients =  feat_ext_tape.gradient(feat_ext_loss, feature_extractor.trainable_variables)
    extended_classifier_gradients = ext_class_tape.gradient(ext_class_loss, extended_classifier.trainable_variables)

    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    feature_extractor_optimizer.apply_gradients(zip(feature_extractor_gradients, feature_extractor.trainable_variables))
    extended_classifier_optimizer.apply_gradients(zip(extended_classifier_gradients, extended_classifier.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        count = 0
        start = time.time()

        for (image_batch_p, label_batch_p), (image_batch_w, labels_batch_w) in zip(datagen.flow(X_train_patches, y_train_patches, batch_size=BATCH_SIZE), datagen.flow(X_train_whole, y_train_whole, batch_size=BATCH_SIZE)):
            train_step(image_batch_p, image_batch_w, label_batch_p, labels_batch_w)
            count += 1
            if count >= 6 * len(X_train_patches) and count >= 6 * len(X_train_whole)
                break

        train_acc_p = accuracy(dataset_X_train_patches, dataset_y_train_patches)
        test_acc_p = accuracy(dataset_X_test_patches, dataset_y_test_patches)
        train_acc_w = accuracy(dataset_X_train_whole, dataset_y_train_whole)
        test_acc_w = accuracy(dataset_X_test_whole, dataset_y_test_whole)
        train_history_p.append(train_acc_p), test_history_p.append(test_acc_p)
        train_history_w.append(train_acc_w), test_history_w.append(test_acc_w)
        print("Foi obtida uma acurácia de treino  nas imagens inteiras de", train_acc_w, "% na época", epoch + 1)
        print("Foi obtida uma acurácia de teste nas imagens inteiras de", test_acc_w, "% na época", epoch + 1)
        print("Foi obtida uma acurácia de treino nos recortes de", train_acc_p, "% na época", epoch + 1)
        print("Foi obtida uma acurácia de teste nos recortes de", test_acc_p, "% na época", epoch + 1)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))



train_history_p = []
test_history_p = []
train_history_w = []
test_history_w = []
train(EPOCHS)
# ------------------------------------------------------- PLOT ------------------------------------------------------- #
print("Foi obtida uma acurácia final  nas imagens inteiras de", accuracy(dataset_X_train_whole, dataset_y_train_whole),
      "% no conjunto de treino")
print("Foi obtida uma acurácia final nas imagens inteiras de", accuracy(dataset_X_test_whole, dataset_y_test_whole),
      "% no conjunto de teste")
print("Foi obtida uma acurácia final nos recortes de", accuracy(dataset_X_train_patches, dataset_y_train_patches),
      "% no conjunto de treino")
print("Foi obtida uma acurácia final nos recortes de", accuracy(dataset_X_test_patches, dataset_y_test_patches),
      "% no conjunto de teste")

plot_history(train_history_p, test_history_p, 'accuracy', nome_plot="acuracias_treino_teste_patches.pdf")
plot_history(train_history_w, test_history_w, 'accuracy', nome_plot="acuracias_treino_teste_whole.pdf")
