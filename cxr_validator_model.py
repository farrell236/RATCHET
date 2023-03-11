import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import tqdm

import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle


##### MIMIC Dataset ####################################################################################################

csv_root = 'preprocessing/mimic'
img_root = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org'

train_reports            = pd.read_csv(os.path.join(csv_root, 'MIMIC_AP_PA_train.csv')).values
cxr_train_image_paths    = [os.path.join(img_root, path) for path in train_reports[:, 0]]

validate_reports         = pd.read_csv(os.path.join(csv_root, 'MIMIC_AP_PA_validate.csv')).values
cxr_validate_image_paths = [os.path.join(img_root, path) for path in train_reports[:, 0]]


##### COCO Dataset #####################################################################################################

def get_coco_paths(coco_root, dataset):
    # Read the json file
    with open(os.path.join(coco_root, f'annotations/captions_{dataset}.json'), 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    coco_image_paths = []
    for annot in tqdm.tqdm(annotations['annotations']):
        image_id = annot['image_id']
        full_coco_image_path = os.path.join(coco_root, dataset, f'{image_id:012d}.jpg')
        coco_image_paths.append(full_coco_image_path)

    return coco_image_paths

MS_COCO_ROOT = '/data/datasets/MS-COCO/2017'
coco_train_image_paths   = get_coco_paths(MS_COCO_ROOT, dataset='train2017')
coco_validate_image_paths = get_coco_paths(MS_COCO_ROOT, dataset='val2017')


##### Train/Validate Image-Path/Ground-Truth pairs #####################################################################

all_train_image_paths = cxr_train_image_paths + coco_train_image_paths
all_train_image_labels = [1] * len(cxr_train_image_paths) + [0] * len(coco_train_image_paths)
all_train_image_paths, all_train_image_labels = \
    shuffle(all_train_image_paths, all_train_image_labels)

all_validate_image_paths = cxr_validate_image_paths + coco_validate_image_paths
all_validate_image_labels = [1] * len(cxr_validate_image_paths) + [0] * len(coco_validate_image_paths)
all_validate_image_paths, all_validate_image_labels = \
    shuffle(all_validate_image_paths, all_validate_image_labels)


##### Tensorflow Dataloader ############################################################################################

def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image with padding to 224x224
    image = tf.image.resize_with_pad(image, 224, 224, method=tf.image.ResizeMethod.BILINEAR)

    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((all_train_image_paths, all_train_image_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


validate_dataset = tf.data.Dataset.from_tensor_slices((all_validate_image_paths, all_validate_image_labels))
validate_dataset = validate_dataset.shuffle(len(validate_dataset))
validate_dataset = validate_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validate_dataset = validate_dataset.batch(8)
validate_dataset = validate_dataset.prefetch(tf.data.experimental.AUTOTUNE)


##### Network Definition ###############################################################################################

model = tf.keras.Sequential([
    tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=(224,224,1)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/cxr_validator_model.tf',
    monitor='val_accuracy', verbose=1, save_best_only=True)

model.fit(train_dataset, epochs=20, callbacks=[checkpoint])
model.fit(train_dataset,
          validation_data=validate_dataset,
          steps_per_epoch=len(train_dataset),
          validation_steps=len(validate_dataset),
          epochs=100,
          callbacks=[checkpoint])


########################################################################################################################
