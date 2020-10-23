import argparse
import datetime
import json
import os
import time
import tqdm


def main(args, hparams):

    # Create strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    print(f'{datetime.datetime.now()}: [*] Number of devices: {strategy.num_replicas_in_sync}')

    # Load dataset
    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset, tokenizer = get_mscoco_dataset(args.data_root, args.vocab_root, batch_size=GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    # Define computational graph in a Strategy wrapper
    with strategy.scope():

        # Create Adam Optimiser
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.init_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Create TF Sparse Categorical Crossentropy Loss Object
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # Loss Function
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        # Define Loss and Accuracy metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        # Define Model
        target_vocab_size = tokenizer.get_vocab_size()
        transformer = Transformer(hparams['n_layer'], hparams['d_model'],
                                  hparams['n_head'], hparams['dff'],
                                  target_vocab_size=target_vocab_size,
                                  rate=hparams['dropout_rate'],
                                  input_shape=(hparams['img_x'], hparams['img_y'], hparams['img_ch']))

        # Model Checkpointing
        ckpt = tf.train.Checkpoint(transformer=transformer,
                                   optimizer=optimizer)
        checkpoint_path = os.path.join('checkpoints', args.model_name)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=args.max_ckpt)
        print(f'{datetime.datetime.now()}: [*] Saving {args.max_ckpt} checkpoints')

        # If resume and checkpoint exists, restore the latest checkpoint.
        init_epoch = 0
        if args.resume:
            if ckpt_manager.latest_checkpoint:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                init_epoch = ckpt.save_counter.numpy()
                print(f'{datetime.datetime.now()}: [*] Restoring Checkpoint: {ckpt_manager.latest_checkpoint}')
            else:
                print(f'{datetime.datetime.now()}: [*] Checkpoint not found. Skipping.')


    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        combined_mask = create_target_masks(tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp,
                                         tar_inp,
                                         True,
                                         combined_mask,
                                         None)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    @tf.function()
    def distributed_train_step(inp, tar):
        strategy.run(train_step, args=(inp, tar))

    print(f'{datetime.datetime.now()}:', '='*20, 'BEGIN TRAINING', '='*20)
    for epoch in range(init_epoch, init_epoch + args.n_epochs):

        print(f'{datetime.datetime.now()}: [*] Training: Epoch {epoch} of {init_epoch + args.n_epochs}...')
        start = time.time()

        # Reset Loss and Accuracy Metrics
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Main Train Step
        t = tqdm.tqdm(enumerate(train_dist_dataset), total=len(train_dataset))
        t_start = datetime.datetime.now()
        for (batch, (inp, tar)) in t:
            distributed_train_step(inp, tar)
            t.set_description(f'{t_start}: Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # Save Checkpoint
        ckpt_save_path = ckpt_manager.save()

        # Print Epoch Summary
        print(f'{datetime.datetime.now()}: '
              f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')
        print(f'{datetime.datetime.now()}: '
              f'Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'{datetime.datetime.now()}: '
              f'Time taken for epoch: {time.time() - start} secs\n')

    print(f'{datetime.datetime.now()}:', '='*20, 'TRAINING COMPLETE', '='*20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_root', default='preprocessing/mscoco')
    parser.add_argument('--data_root', default='/data/datasets/MS-COCO/2017/')
    parser.add_argument('--model_name', default='coco_train0')
    parser.add_argument('--model_params', default='model/hparams.json')
    parser.add_argument('--n_epochs', default=20)
    parser.add_argument('--init_lr', default=1e-4)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--resume', default=True)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--max_ckpt', default=5)
    parser.add_argument('--debug_level', default='3')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--n_threads_intra_op', type=int, default=None)
    parser.add_argument('--n_threads_inter_op', type=int, default=None)
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
    from datasets.mscoco import get_mscoco_dataset
    from model.transformer import Transformer, default_hparams
    from model.utils import create_target_masks
    from model.lr_scheduler import CustomSchedule

    # Set Tensorflow 2.0 logging level
    error_level = {'0': 'DEBUG', '1': 'INFO', '2': 'WARN', '3': 'ERROR'}
    tf.get_logger().setLevel(error_level[args.debug_level])
    print(f'{datetime.datetime.now()}: [*] Setting Tensorflow Global Logging Level: {error_level[args.debug_level]}')

    # Set available CPU Threads
    tf.config.threading.set_intra_op_parallelism_threads(args.n_threads_intra_op)
    tf.config.threading.set_inter_op_parallelism_threads(args.n_threads_inter_op)
    print(f'{datetime.datetime.now()}: [*] Intra op parallelism threads: {args.n_threads_intra_op}')
    print(f'{datetime.datetime.now()}: [*] Inter op parallelism threads: {args.n_threads_intra_op}')

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
