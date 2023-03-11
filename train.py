import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['XLA_FLAGS'] = '--xla_compile=False'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import argparse
import json

import tensorflow as tf

from model.transformer import Transformer, default_hparams
from datasets.mimic import get_mimic_dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': str(int(self.d_model)),
            'warmup_steps': str(int(self.warmup_steps)),
        }


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def main(args, hparams):

    train_batches, tokenizer = get_mimic_dataset(args.csv_root, args.vocab_root, args.mimic_root,
                                                 batch_size=args.batch_size)

    val_batches, _ = get_mimic_dataset(args.csv_root, args.vocab_root, args.mimic_root,
                                       mode='validate',
                                       batch_size=args.batch_size)

    csv_logger_callback = tf.keras.callbacks.CSVLogger('checkpoints/training.log')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/{args.model_name}.tf',
        save_weights_only=True,
        monitor='val_masked_accuracy',
        mode='max',
        save_best_only=True)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        learning_rate = args.init_lr if args.init_lr is not None else \
             CustomSchedule(hparams['d_model'])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        transformer = Transformer(
            num_layers=hparams['num_layers'],
            d_model=hparams['d_model'],
            num_heads=hparams['num_heads'],
            dff=hparams['dff'],
            target_vocab_size=tokenizer.get_vocab_size(),
            dropout_rate=hparams['dropout_rate'],
            input_shape=(224, 224, 1),
            classifier_weights=args.classifier_weights)

        transformer.compile(
            loss=masked_loss,
            optimizer=optimizer,
            metrics=[masked_accuracy],
        )

    transformer.fit(
        train_batches,
        epochs=args.n_epochs,
        validation_data=val_batches,
        callbacks=[model_checkpoint_callback, csv_logger_callback]
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', default='preprocessing/mimic')
    parser.add_argument('--vocab_root', default='preprocessing/mimic')
    parser.add_argument('--mimic_root', default='/vol/biodata/data/MIMIC-CXR/mimic-cxr-jpg')
    parser.add_argument('--model_name', default='RATCHET')
    parser.add_argument('--model_params', default='model/hparams.json')
    parser.add_argument('--classifier_weights', default=None)
    parser.add_argument('--n_epochs', default=10)
    parser.add_argument('--init_lr', default=None)
    parser.add_argument('--batch_size', default=32)
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
