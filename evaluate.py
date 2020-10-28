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


def evaluate(inp_img, transformer, tokenizer, max_length=128):

    # The first token to the transformer should be the start token
    output = tf.convert_to_tensor([[tokenizer.token_to_id('<s>')]])

    # txt_pred, lbl_pred, attention_weights = transformer(inp_img,
    #                                                     output,
    #                                                     False,
    #                                                     create_target_masks(output),
    #                                                     None)


    for _ in tqdm.tqdm(range(max_length)):
        combined_mask = create_target_masks(output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, lbl_pred, attention_weights = transformer(inp_img,
                                                            output,
                                                            False,
                                                            combined_mask,
                                                            None)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1, :]  # (batch_size, vocab_size)
        predictions = top_k_logits(predictions, k=6)
        predictions = top_p_logits(predictions, p=0.5)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)[:, tf.newaxis]
        # predicted_id = tf.random.categorical(predictions, num_samples=1, dtype=tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 2:  # stop token #tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0)[1:], lbl_pred, attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)


    return tf.squeeze(output, axis=0)[1:], lbl_pred, attention_weights


def main(args, hparams):

    # Get test dataset
    test_dataset, tokenizer = get_mimic_dataset(args.csv_root, args.vocab_root, args.mimic_root,
                                                batch_size=args.batch_size, mode='test')

    # Define model
    target_vocab_size = tokenizer.get_vocab_size()
    transformer = Transformer(hparams['n_layer'], hparams['d_model'],
                              hparams['n_head'], hparams['dff'],
                              target_vocab_size=target_vocab_size,
                              rate=hparams['dropout_rate'],
                              input_shape=(hparams['img_x'], hparams['img_y'], hparams['img_ch']))

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer)
    checkpoint_path = os.path.join('checkpoints', args.model_name)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    if latest_checkpoint:
        print(f'{datetime.datetime.now()}: [*] Restoring Checkpoint: {latest_checkpoint}')
        ckpt.restore(latest_checkpoint)
    else:
        print(f'{datetime.datetime.now()}: [*] No checkpoints found. Exiting.')
        exit(0)


    #################### Run inference ####################

    inp_all = []
    lbl_true_all = []
    lbl_pred_all = []
    tar_true_all = []
    tar_pred_all = []

    t = tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset))
    for (batch, (inp, lbl, tar)) in t:

        true_lbl = lbl
        true_txt = tokenizer.decode(np.trim_zeros(tar[0].numpy(), 'b'))

        result, pred_lbl, attention_weights = evaluate(
            inp, transformer=transformer, tokenizer=tokenizer)

        inp_all.append(inp)
        lbl_true_all.append(true_lbl)
        lbl_pred_all.append(pred_lbl)
        tar_true_all.append(true_txt)
        tar_pred_all.append(result)

    lbl_true_all = np.concatenate(lbl_true_all, axis=0)
    lbl_pred_all = np.concatenate(lbl_pred_all, axis=0)

    from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc, accuracy_score, roc_auc_score
    from matplotlib import pyplot

    for i in range(14):
        # precision, recall, thresholds = precision_recall_curve(y_true_all[:, i], y_pred_all[:, i])
        fpr, tpr, thresholds = roc_curve(lbl_true_all[:, i], lbl_pred_all[:, i])
        auc_score = auc(fpr, tpr)
        print(i, auc_score)

    tar_pred_all = tokenizer.decode_batch(tar_pred_all)
    lbl_true_all_df = pd.DataFrame(lbl_true_all)
    lbl_pred_all_df = pd.DataFrame(lbl_pred_all)
    tar_true_all_df = pd.DataFrame(tar_true_all)
    tar_pred_all_df = pd.DataFrame(tar_pred_all)

    all_true = pd.concat((lbl_true_all_df, tar_true_all_df), axis=1)
    all_pred = pd.concat((lbl_pred_all_df, tar_pred_all_df), axis=1)

    all_true.to_csv('all_true.csv', index=False, header=False)
    all_pred.to_csv('all_pred.csv', index=False, header=False)

    print('----- EVAL COMPLETE -----')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', default='preprocessing/mimic')
    parser.add_argument('--vocab_root', default='preprocessing/mimic')
    parser.add_argument('--mimic_root', default='/data/datasets/chest_xray/MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org')
    parser.add_argument('--model_name', default='train1')
    parser.add_argument('--model_params', default='model/hparams.json')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--debug_level', default='3')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
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
    from model.transformer import Transformer, default_hparams
    from model.utils import create_target_masks

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
