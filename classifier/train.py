import time

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
import pandas as pd
import tensorflow as tf


##### Load Dataset CSV: image paths and labels #########################################################################

csv_root    = '../preprocessing/mimic'
mimic_root  = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'

# Set unsure values (-1) to 0 or 1
replacements = {float('nan'): 0, -1.0: 1}

train_csv_file = os.path.join(csv_root, 'MIMIC_AP_PA_train.csv')
train_reports = pd.read_csv(train_csv_file).replace(replacements).values
train_image_paths = [os.path.join(mimic_root, path) for path in train_reports[:, 0]]
train_labels = np.uint8(train_reports[:, 2:])

valid_csv_file = os.path.join(csv_root, 'MIMIC_AP_PA_validate.csv')
valid_reports = pd.read_csv(valid_csv_file).replace(replacements).values
valid_image_paths = [os.path.join(mimic_root, path) for path in valid_reports[:, 0]]
valid_labels = np.uint8(valid_reports[:, 2:])


##### Create Tensorflow Dataset ########################################################################################

def parse_function(filename, text):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image with padding to 244x244
    image = tf.image.resize_with_pad(image, 224, 224, method=tf.image.ResizeMethod.BILINEAR)

    return image, text

def augmentation_fn(image, text):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)

    # Random brightness, saturation and contrast shifting
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, text

def make_grayscale_fn(image, text):
    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)

    return image, text

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(augmentation_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(make_grayscale_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_labels))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.map(make_grayscale_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(16)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)


##### Setup DenseNet-121 Network and Training ##########################################################################

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 1)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(14, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5',
    monitor='val_loss', verbose=1)

class CkptDenseNet(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'End epoch {epoch} of training; saving model weights to: checkpoint/epoch_{epoch}.hdf5')
        print(f'Sanity check: {self.model.layers[0].name}')
        self.model.layers[0].save_weights(f'checkpoint/densenet-121_{epoch}.hdf5')

model.fit(train_dataset,
          validation_data=valid_dataset,
          # use_multiprocessing=True, workers=8,
          epochs=10,
          callbacks=[checkpoint, CkptDenseNet()])


