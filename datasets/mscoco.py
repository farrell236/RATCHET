from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import tqdm

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
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return (image, texts_inputs), texts_labels


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img):
    blur = _gaussian_kernel(3, 2, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def get_mscoco_dataset(coco_root,
                       vocab_root,
                       max_length=64,
                       batch_size=16,
                       n_threads=tf.data.experimental.AUTOTUNE,
                       buffer_size=None,
                       mode='train'):
    assert mode in ['train', 'val']

    # Load Byte-Level BPE Tokenizer with mimic vocabulary
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(vocab_root, 'coco-vocab.json'),
        os.path.join(vocab_root, 'coco-merges.txt'),
    )

    # Read the json file
    with open(os.path.join(coco_root, f'annotations/captions_{mode}2017.json'), 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in tqdm.tqdm(annotations['annotations']):
        caption = annot['caption']
        image_id = annot['image_id']
        full_coco_image_path = os.path.join(coco_root, f'{mode}2017', f'{image_id:012d}.jpg')

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Tokenize reports
    texts_tokenized = tokenizer.encode_batch(all_captions)
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
    dataset = tf.data.Dataset.from_tensor_slices((all_img_name_vector, texts_inputs, texts_labels))
    if mode == 'train':
        dataset = dataset.shuffle(len(dataset) if buffer_size == None else buffer_size)
    dataset = dataset.map(parse_function, num_parallel_calls=n_threads)
    if mode == 'train':
        dataset = dataset.map(augmentation_fn, num_parallel_calls=n_threads)
    # dataset = dataset.map(lambda x, y: (apply_blur(x), y), num_parallel_calls=n_threads)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(n_threads)

    return dataset, tokenizer


if __name__ == '__main__':
    vocab_root = 'preprocessing/mscoco/'
    coco_root = '/data/datasets/MS-COCO/2017/'

    train_dataset, _ = get_mscoco_dataset(coco_root, vocab_root)

    iterator = train_dataset.as_numpy_iterator()
    batch = iterator.next()
