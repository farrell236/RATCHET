import argparse
import datetime
import json
import os
import tqdm

import numpy as np
import pandas as pd

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


def evaluate(image, encoder, decoder, tokenizer, max_length=128):
    attention_plot = np.zeros((max_length, 7*7))

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(image)

    dec_input = tf.expand_dims([tokenizer.token_to_id('<s>')], 0)
    result = []

    at = []
    H = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)[0].numpy()
        # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(predicted_id)

        if predicted_id == 2:
            break

        H.append(hidden)
        at.append(attention_weights[..., 0])

        dec_input = tf.expand_dims([predicted_id], 0)

    H = tf.stack(H, axis=-1)
    at = tf.stack(at, axis=-1)

    G = tf.nn.softmax(tf.matmul(decoder.W2, tf.tanh(tf.matmul(decoder.W1, H))))
    M = tf.matmul(G, H, transpose_b=True)

    X_AETE = tf.reduce_max(M, axis=1)

    g = tf.reduce_max(G, axis=1, keepdims=True)
    aw_s = tf.reduce_sum(at * g, axis=-1, keepdims=True)
    X_SW_GAP = tf.reduce_sum(features * aw_s, axis=1)

    joint_learning = tf.concat((X_AETE, X_SW_GAP), axis=-1)
    class_pred = decoder.fc2_label(decoder.fc1_label(joint_learning))

    attention_plot = attention_plot[:len(result), :]
    return result[:-1], class_pred, attention_plot


def main(args, hparams):

    # Get test dataset
    test_dataset, tokenizer = get_mimic_dataset(args.csv_root, args.vocab_root, args.mimic_root,
                                                batch_size=args.batch_size, mode='test')

    # Define model
    target_vocab_size = tokenizer.get_vocab_size()
    encoder = CNN_Encoder(hparams['d_model'], input_shape=(hparams['img_x'], hparams['img_y'], hparams['img_ch']))
    decoder = TieNet_Decoder(hparams['d_model'], hparams['dff'], target_vocab_size)

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder)
    checkpoint_path = os.path.join('checkpoints', args.model_name)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    if latest_checkpoint:
        print(f'{datetime.datetime.now()}: [*] Restoring Checkpoint: {latest_checkpoint}')
        ckpt.restore(latest_checkpoint)
    else:
        print(f'{datetime.datetime.now()}: [*] No checkpoints found. Exiting.')
        exit(0)


    #################### Run inference ####################
    pred_txt = dict()
    true_txt = dict()
    pred_lbl = []
    true_lbl = []

    t = tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset))
    for (idx, (inp, tar, lbl)) in t:
        true_txt[idx] = tokenizer.decode(np.trim_zeros(tar[0].numpy(), 'b')[1:-1])
        true_lbl.append(lbl[0])
        result, class_pred, attention_plot = evaluate(inp, encoder, decoder, tokenizer)
        pred_txt[idx] = tokenizer.decode(result)
        pred_lbl.append(tf.nn.sigmoid(class_pred)[0])
        if idx==10: break

    pred_txt_df = pd.DataFrame.from_dict(pred_txt, orient='index')
    true_txt_df = pd.DataFrame.from_dict(true_txt, orient='index')
    pred_lbl_df = pd.DataFrame(np.array(pred_lbl))
    true_lbl_df = pd.DataFrame(np.array(true_lbl))

    pred_txt_df.to_csv('all_pred_txt.csv', index=False, header=False)
    true_txt_df.to_csv('all_true_txt.csv', index=False, header=False)
    pred_lbl_df.to_csv('all_pred_lbl.csv', index=False, header=False)
    true_lbl_df.to_csv('all_true_lbl.csv', index=False, header=False)

    print('Evaluation Complete!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', default='preprocessing/mimic')
    parser.add_argument('--vocab_root', default='preprocessing/mimic')
    parser.add_argument('--mimic_root', default='/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org')
    parser.add_argument('--model_name', default='train31')
    parser.add_argument('--model_params', default='model/hparams.json')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--debug_level', default='3')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='')
    args = parser.parse_args()

    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.debug_level

    # Set available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.nGPU = 0 if len(args.gpu) == 0 else len(args.gpu.split(','))
    print(f'{datetime.datetime.now()}: [*] Using GPU(s): {args.gpu}')

    # Import Tensorflow AFTER setting environment variables
    # ISSUE: https://github.com/tensorflow/tensorflow/issues/31870
    import tensorflow as tf
    from datasets.mimic import get_mimic_dataset
    from model.baseline import CNN_Encoder, TieNet_Decoder, default_hparams

    # Set Tensorflow 2.0 logging level
    error_level = {'0': 'DEBUG', '1': 'INFO', '2': 'WARN', '3': 'ERROR'}
    tf.get_logger().setLevel(error_level[args.debug_level])
    print(f'{datetime.datetime.now()}: [*] Setting Tensorflow Global Logging Level: {error_level[args.debug_level]}')

    # Load mode default hyperparameters and update from file if exist
    hparams = default_hparams()
    if args.model_params:
        with open(args.model_params) as json_file:
            hparams_from_file = json.load(json_file)
            hparams.update((k, hparams_from_file[k])
                           for k in set(hparams_from_file).intersection(hparams))
    print(f'{datetime.datetime.now()}: [*] Model Parameters: {hparams}')

    # Set tensorflow random seed
    tf.random.set_seed(args.seed)

    # Run main training sequence
    main(args=args, hparams=hparams)
