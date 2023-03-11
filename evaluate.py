import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['XLA_FLAGS'] = '--xla_compile=False'
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from datasets.mimic import get_mimic_dataset
from model.transformer import Transformer, default_hparams


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def evaluate(inp_img, transformer, tokenizer, max_length=128):

    # The first token to the transformer should be the start token
    output = tf.convert_to_tensor([[tokenizer.token_to_id('<s>')]])

    for _ in tqdm(range(max_length)):

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer([inp_img, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1, :]  # (batch_size, vocab_size)
        predictions = top_k_logits(predictions, k=6)
        # predictions = top_p_logits(predictions, p=0.5)

        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)[:, tf.newaxis]
        predicted_id = tf.random.categorical(predictions, num_samples=1, dtype=tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 2:  # stop token #tokenizer_en.vocab_size + 1:
            break

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    # transformer([inp_img, output[:, :-1]], training=False)
    return tf.squeeze(output, axis=0)[1:], transformer.decoder.last_attn_scores


def main(args, hparams):

    # Get test dataset
    test_dataset, tokenizer = get_mimic_dataset(args.csv_root, args.vocab_root, args.mimic_root,
                                                batch_size=args.batch_size, mode='test')

    # Load Model
    transformer = Transformer(
        num_layers=hparams['num_layers'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        dff=hparams['dff'],
        target_vocab_size=tokenizer.get_vocab_size(),
        dropout_rate=hparams['dropout_rate'])
    transformer.load_weights(args.model)

    #################### Run inference ####################
    pred_txt = dict()
    true_txt = dict()

    t = tqdm(enumerate(test_dataset), total=len(test_dataset))
    for (idx, ((image, texts_inputs), texts_labels)) in t:
        true_txt[idx] = tokenizer.decode(np.trim_zeros(texts_labels[0].numpy(), 'b')[:-1])
        result, attention_weights = evaluate(image, transformer=transformer, tokenizer=tokenizer)
        pred_txt[idx] = tokenizer.decode(result)

    pred_txt_df = pd.DataFrame.from_dict(pred_txt, orient='index')
    true_txt_df = pd.DataFrame.from_dict(true_txt, orient='index')

    pred_txt_df.to_csv('/tmp/all_pred.csv', index=False, header=False)
    true_txt_df.to_csv('/tmp/all_true.csv', index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', default='preprocessing/mimic')
    parser.add_argument('--vocab_root', default='preprocessing/mimic')
    parser.add_argument('--mimic_root', default='/mnt/nas_houbb/datasets/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--model', default='checkpoints/RATCHET.tf')
    parser.add_argument('--model_params', default='model/hparams.json')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    # Load mode default hyperparameters and update from file if exist
    hparams = default_hparams()
    if args.model_params:
        with open(args.model_params) as json_file:
            hparams_from_file = json.load(json_file)
            hparams.update((k, hparams_from_file[k])
                           for k in set(hparams_from_file).intersection(hparams))

    # Set tensorflow random seed
    tf.random.set_seed(args.seed)

    # Run main training sequence
    main(args=args, hparams=hparams)
