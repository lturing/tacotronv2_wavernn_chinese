import os
import threading
import time
import traceback

import numpy as np
import tensorflow as tf
from tacotron.utils.infolog import log
from sklearn.model_selection import train_test_split
from tacotron.utils.text import text_to_sequence

_batches_per_group = 20

class Feeder:
    """
        Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._train_offset = 0
        self._test_offset = 0

        # Load metadata
        self._mel_dir = os.path.dirname(metadata_filename) 
        self._linear_dir = os.path.dirname(metadata_filename) #, 'linear')
        dura = 0 
        self._metadata = []
        with open(metadata_filename, encoding='utf-8') as f:
            for line in f:
                #audio-000001.npy|mel-000001.npy|46200|168|卡尔普陪外孙玩滑梯。|k a3 er3 p u3 p ei2 w ai4 s un1 w an2 h ua2 t i1 。
                line = line.strip().split('|')
                mel = line[1].strip()
                dura += int(line[3])
                pyin = line[-1].strip()
                self._metadata.append([mel, pyin])

            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = dura * frame_shift_ms / (3600)
            log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))


        self._train_meta = self._metadata
        print(len(self._train_meta), '*' * 100)

        #pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        #explicitely setting the padding to a value that doesn't originally exist in the spectogram
        #to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.
        #Mark finished sequences with 1s
        self._token_pad = 1.

        # Create placeholders for inputs and targets. Don't specify batch size because we want
        # to be able to feed different batch sizes at eval time.
        self._placeholders = [
        tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
        tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
        tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
        tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
        tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
        ]

        # Create queue for buffering data
        queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32], name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.targets_lengths = queue.dequeue()

        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.mel_targets.set_shape(self._placeholders[2].shape)
        self.token_targets.set_shape(self._placeholders[3].shape)
        self.targets_lengths.set_shape(self._placeholders[4].shape)


    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
        thread.daemon = True #Thread will close when parent quits
        thread.start()


    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples
            n = self._hparams.tacotron_batch_size
            r = self._hparams.outputs_per_step
            examples = [self._get_next_example() for i in range(n * _batches_per_group)]

            # Bucket examples based on similar output sequence length for efficiency
            examples.sort(key=lambda x: x[-1])
            batches = [examples[i: i+n] for i in range(0, len(examples), n)]
            np.random.shuffle(batches)

            log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)

        meta = self._train_meta[self._train_offset]
        self._train_offset += 1

        text = meta[-1].strip().split(' ')

        input_data = np.asarray(text_to_sequence(text), dtype=np.int32)
        mel_target = np.load(os.path.join(self._mel_dir, meta[0]))
        #Create parallel sequences containing zeros to represent a non finished sequence
        token_target = np.asarray([0.] * (len(mel_target) - 1))
        return (input_data, mel_target, token_target, len(mel_target))

    def _prepare_batch(self, batch, outputs_per_step):
        np.random.shuffle(batch)

        targets_lengths = np.asarray([x[-1] for x in batch], dtype=np.int32) #Used to mask loss
        input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)

        inputs = self._prepare_inputs([x[0] for x in batch])
        mel_targets = self._prepare_targets([x[1] for x in batch], outputs_per_step)

        #Pad sequences with 1 to infer that the sequence is done
        token_targets = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)

        return (inputs, input_lengths, mel_targets, token_targets, targets_lengths)

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs])

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets])

    def _prepare_token_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets]) + 1
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_token_target(t, data_len) for t in targets])

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
