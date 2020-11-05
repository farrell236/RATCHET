import time
import tqdm

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import tensorflow as tf


class_names = [
    'No Finding',
    'Enl. C. med.',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


##### Load Dataset CSV: image paths and labels #########################################################################

csv_root    = '../preprocessing/mimic'
mimic_root  = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/'

# Set unsure values (-1) to 0 or 1
replacements = {float('nan'): 0, -1.0: 1}

test_csv_file = os.path.join(csv_root, 'MIMIC_AP_PA_test.csv')
test_reports = pd.read_csv(test_csv_file).replace(replacements).values
test_image_paths = [os.path.join(mimic_root, path) for path in test_reports[:, 0]]
test_labels = np.uint8(test_reports[:, 2:])


##### Create Tensorflow Dataset ########################################################################################

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


test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(16)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


##### Setup DenseNet-121 Network and Training ##########################################################################

model = tf.keras.models.load_model('weights-improvement-09-0.32.hdf5')

y_true_all = []
y_pred_all = []
for i, (x, y_true) in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
    y_pred = model.predict(x)
    y_true_all.append(y_true[0])
    y_pred_all.append(y_pred[0])
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

label_baseline_probs = []

from sklearn.metrics import f1_score, roc_curve, \
    precision_recall_curve, \
    auc, accuracy_score, \
    roc_auc_score, \
    precision_recall_fscore_support, \
    average_precision_score
from matplotlib import pyplot

print('                     auc,        accuracy    precision   recall      f_score')
for i in range(14):
    fpr, tpr, thresholds = roc_curve(y_true_all[:, i], y_pred_all[:, i])
    auc_score = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    acc_score = accuracy_score(y_true_all[:, i], y_pred_all[:, i] > optimal_threshold)

    precision, recall, f_score, _ = precision_recall_fscore_support(y_true_all[:, i],
                                                                    y_pred_all[:, i] > optimal_threshold,
                                                                    average='binary')

    print(f'{class_names[i]:<20} {auc_score:>.5f}, \t {acc_score:>.5f}, \t {precision:>.5f}, \t {recall:>.5f}, \t {f_score:>.5f}')

    pyplot.title(class_names[i])
    pyplot.plot(fpr, tpr, marker='.', label=f'{class_names[i]}')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.savefig(f'roc_curves/roc_curve_{i}.png')
    pyplot.clf()


a=1


'''
CheXpert Classes (*)     AUC         ACC         Precision   Recall      F1  
    No Finding           0.70434, 	 0.78404, 	 0.44444, 	 0.48780, 	 0.46512
    Enl. C. med.         0.65104, 	 0.50704, 	 0.13675, 	 0.80000, 	 0.23358
  * Cardiomegaly         0.71192, 	 0.66197, 	 0.46739, 	 0.65152, 	 0.54430
    Lung Lesion          0.75976, 	 0.79812, 	 0.11111, 	 0.62500, 	 0.18868
    Lung Opacity         0.64735, 	 0.65728, 	 0.52564, 	 0.53247, 	 0.52903
  * Edema                0.78450, 	 0.70423, 	 0.50467, 	 0.84375, 	 0.63158
  * Consolidation        0.68629, 	 0.59155, 	 0.10753, 	 0.71429, 	 0.18692
    Pneumonia            0.62936, 	 0.64319, 	 0.36145, 	 0.56604, 	 0.44118
  * Atelectasis          0.73597, 	 0.65728, 	 0.46491, 	 0.81538, 	 0.59218
    Pneumothorax         0.81538, 	 0.63380, 	 0.04938, 	 0.80000, 	 0.09302
  * Pleural Effusion     0.86574, 	 0.82629, 	 0.75000, 	 0.69565, 	 0.72180
    Pleural Other        0.78230, 	 0.68545, 	 0.04348, 	 0.75000, 	 0.08219
    Fracture             0.41270, 	 0.30986, 	 0.01351, 	 0.66667, 	 0.02649
    Support Devices      0.82396, 	 0.76995, 	 0.66250, 	 0.70667, 	 0.68387
'''

