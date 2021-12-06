from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os
import torch

import numpy as np
import pandas as pd
import tensorflow as tf

from tokenizers import ByteLevelBPETokenizer


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, img_dir, mode='train', max_length=128):

        assert mode in ['train', 'validate', 'test']

        # Load Byte-Level BPE Tokenizer with mimic vocabulary
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(csv_dir, 'mimic-vocab.json'),
            os.path.join(csv_dir, 'mimic-merges.txt'),
        )

        # Read MIMIC_AP_PA_{mode}.csv file and set unsure values (-1) to 0 or 1
        csv_file = pd.read_csv(os.path.join(csv_dir, f'MIMIC_AP_PA_{mode}.csv'))

        self.image_paths = os.path.join(img_dir, '') + csv_file['DicomPath']
        texts = csv_file['Reports']
        labels = csv_file[csv_file.columns[2:]]

        # Tokenize reports
        texts_tokenized = tokenizer.encode_batch(list(texts))
        texts_tokenized = [[tokenizer.token_to_id('<s>')] +
                           seq.ids +
                           [tokenizer.token_to_id('</s>')]
                           for seq in texts_tokenized]
        self.texts_tokenized = tf.keras.preprocessing.sequence.pad_sequences(
            texts_tokenized, maxlen=max_length, dtype='int32', padding='post', truncating='post')

    def __len__(self):
        return len(self.image_paths)

    def get_padding(self, image):
        max_length = max(image.shape[0], image.shape[1])
        h_padding = (max_length - image.shape[0]) / 2
        v_padding = (max_length - image.shape[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
        padding = ((int(l_pad), int(r_pad)), (int(t_pad), int(b_pad)))
        return padding

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths.iloc[idx], 0)
        image = np.pad(image, self.get_padding(image))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = np.divide(image, 255.).astype('float32')
        label = self.texts_tokenized[idx, ...]

        return image[None, ...], label  # sample



if __name__ == '__main__':

    csv_root = '.'
    img_dir = '/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org'

    # Parameters
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6}
    max_epochs = 100

    # Generators
    training_set = CustomImageDataset(csv_root, img_dir, mode='train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = CustomImageDataset(csv_root, img_dir, mode='validate')
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    device = torch.device('cuda')

    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            a=1

            # Model computations
            # [...]

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                # [...]

