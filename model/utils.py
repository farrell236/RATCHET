from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...].astype('float32')

    return torch.from_numpy(pos_encoding)


def create_padding_mask(seq):
    seq = torch.eq(seq, 0).type(torch.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size, device=torch.device('cpu')):
    mask = 1 - torch.ones((size, size), device=device).tril()
    return mask  # (seq_len, seq_len)


def create_target_masks(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1], tar.device)
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask
