from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tokenizers import ByteLevelBPETokenizer


def parse_function(filename, texts_inputs, texts_labels):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image with padding to 244x244
    image = tf.image.resize_with_pad(image, 224, 224, method=tf.image.ResizeMethod.BILINEAR)

    return (image, texts_inputs), texts_labels


def augmentation_fn(inputs, texts_labels):
    # Random left-right flip the image
    image, texts_inputs = inputs
    image = tf.image.random_flip_left_right(image)

    # Random brightness, saturation and contrast shifting
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return (image, texts_inputs), texts_labels


def make_grayscale_fn(inputs, texts_labels):
    # Convert image to grayscale
    image, texts_inputs = inputs
    image = tf.image.rgb_to_grayscale(image)

    return (image, texts_inputs), texts_labels


def get_mimic_dataset(csv_root,
                      vocab_root,
                      mimic_root,
                      max_length=128,
                      batch_size=16,
                      n_threads=16,
                      buffer_size=10000,
                      mode='train',
                      unsure=1):

    assert mode in ['train', 'validate', 'test']
    assert unsure in [0, 1]

    # Load Byte-Level BPE Tokenizer with mimic vocabulary
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(vocab_root, 'mimic-vocab.json'),
        os.path.join(vocab_root, 'mimic-merges.txt'),
    )

    # Read MIMIC_AP_PA_{mode}.csv file and set unsure values (-1) to 0 or 1
    csv_file        = os.path.join(csv_root, f'MIMIC_AP_PA_{mode}.csv')
    replacements    = {float('nan'): 0, -1.0: unsure}
    reports         = pd.read_csv(csv_file).replace(replacements).values

    image_paths     = [os.path.join(mimic_root, path) for path in reports[:, 0]]
    texts           = reports[:, 1]
    labels          = np.uint8(reports[:, 2:])

    # Tokenize reports
    texts_tokenized = tokenizer.encode_batch(list(texts))
    texts_tokenized = [[tokenizer.token_to_id('<s>')] +
                             seq.ids +
                             [tokenizer.token_to_id('</s>')]
                             for seq in texts_tokenized]
    texts_tokenized = tf.keras.preprocessing.sequence.pad_sequences(
        texts_tokenized, maxlen=max_length, dtype='int32', padding='post', truncating='post')

    texts_tokenized = texts_tokenized[:, :(max_length + 1)]
    texts_inputs = texts_tokenized[:, :-1]  # Drop the [END] tokens
    texts_labels = texts_tokenized[:, 1:]  # Drop the [START] tokens

    # Create Tensorflow dataset (image, text) pair
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts_inputs, texts_labels))
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(parse_function, num_parallel_calls=n_threads)
    if mode == 'train':
        dataset = dataset.map(augmentation_fn, num_parallel_calls=n_threads)
    dataset = dataset.map(make_grayscale_fn, num_parallel_calls=n_threads)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size)

    return dataset, tokenizer


if __name__ == '__main__':

    csv_root = 'preprocessing/mimic/'
    vocab_root = 'preprocessing/mimic/'
    mimic_root = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'

    train_dataset, _ = get_mimic_dataset(csv_root, mimic_root, vocab_root)

    iterator = train_dataset.as_numpy_iterator()
    batch = iterator.next()
