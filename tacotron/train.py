import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime

from tacotron.utils import infolog
import numpy as np
import tensorflow as tf
from tacotron.datasets import audio
from tacotron_hparams import hparams_debug_string
from tacotron.feeder import Feeder
from tacotron.models import create_model
from tacotron.utils import ValueWindow, plot
from tacotron.utils.text import sequence_to_text
from tacotron.utils.symbols import symbols
from tqdm import tqdm

log = infolog.log


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    #Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path

    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        #Initialize config
        embedding = config.embeddings.add()
        #Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta
    
    #Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

def add_train_stats(model, hparams):
    with tf.variable_scope('stats') as scope:
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)

        if hparams.predict_linear:
            tf.summary.scalar('linear_loss', model.linear_loss)
            tf.summary.histogram('linear_outputs', model.linear_outputs)
            tf.summary.histogram('linear_targets', model.linear_targets)
        
        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
        if hparams.tacotron_teacher_forcing_mode == 'scheduled':
            tf.summary.scalar('teacher_forcing_ratio', model.ratio) #Control teacher forcing ratio decay when mode = 'scheduled'
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
        return tf.summary.merge_all()


def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
        model_name = None
        if args.model == 'Tacotron-2':
            model_name = 'Tacotron'
        model = create_model(model_name or args.model, hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
            targets_lengths=feeder.targets_lengths, global_step=global_step,
            is_training=True)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_train_stats(model, hparams)
        return model, stats


def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, 'taco_pretrained')
    wav_plot = os.path.join(log_dir, 'wav_plot')

    tensorboard_dir = os.path.join(log_dir, 'tacotron_events')
    meta_folder = os.path.join(log_dir, 'metas')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    os.makedirs(wav_plot, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
    input_path = os.path.join(args.data_dir, args.tacotron_input)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    #Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tacotron_random_seed)

    #Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = Feeder(coord, input_path, hparams)

    #Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model, stats = model_train_mode(args, feeder, hparams, global_step)

    #Embeddings metadata
    char_embedding_meta = os.path.join(meta_folder, 'CharacterEmbeddings.tsv')
    if not os.path.isfile(char_embedding_meta):
        with open(char_embedding_meta, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                if symbol == ' ':
                    symbol = '\\s' #For visual purposes, swap space with \s

                f.write('{}\n'.format(symbol))

    char_embedding_meta = char_embedding_meta.replace(log_dir, '..')

    #Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=20)

    log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))

    #Memory allocation on the GPU as needed
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    '''

    #Train
    with tf.Session() as sess: #config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            #saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                        #initial_global_step = global_step.assign(0)
                        #sess.run(initial_global_step)

                    else:
                        log('No model to load at {}'.format(save_dir), slack=True)
                        saver.save(sess, checkpoint_path, global_step=global_step)

                except tf.errors.OutOfRangeError as e:
                    log('Cannot restore checkpoint: {}'.format(e), slack=True)
            else:
                log('Starting new training!', slack=True)
                saver.save(sess, checkpoint_path, global_step=global_step)

            #initializing feeder
            feeder.start_threads(sess)

            #Training loop
            while not coord.should_stop() and step < args.tacotron_train_steps:
                start_time = time.time()
                step, loss, opt, before_loss, after_loss, token_loss, reg_loss = sess.run([global_step, model.loss, model.optimize, 
                            model.before_loss, model.after_loss, model.stop_token_loss, model.regularization_loss])
                
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step{:6d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}, mel_before={:.5f}, mel_after={:.5f}, token_loss={:.5f}, reg_loss={:.5f}]'.format(step, time_window.average, loss, loss_window.average, before_loss, after_loss, token_loss, reg_loss)
            
                log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

                if np.isnan(loss) or loss > 100.:
                    log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                    raise Exception('Loss exploded')

                if step % args.summary_interval == 0:
                    log('\nWriting summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(stats), step)

                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or step == 300:
                    #Save model and current global step
                    saver.save(sess, checkpoint_path, global_step=global_step)

                    log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')

                    input_seq, mel_prediction, alignment, target, target_length = sess.run([
                        model.inputs[0],
                        model.mel_outputs[0],
                        model.alignments[0],
                        model.mel_targets[0],
                        model.targets_lengths[0],
                        ])
                    
                    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)

                    audio.save_wav(wav, os.path.join(wav_plot, 'step-{}-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)

                    #save alignment plot to disk (control purposes)
                    plot.plot_alignment(alignment, os.path.join(wav_plot, 'step-{}-align.png'.format(step)),
                        title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss),
                        max_len=target_length // hparams.outputs_per_step)
                    #save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot.plot_spectrogram(mel_prediction, os.path.join(wav_plot, 'step-{}-mel-spectrogram.png'.format(step)),
                        title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss), target_spectrogram=target,
                        max_len=target_length)
                    
                    print(', '.join(map(str, input_seq.tolist())))

                    log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

                if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
                    #Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    #Update Projector
                    log('\nSaving Model Character Embeddings visualization..')
                    add_embedding_stats(summary_writer, [model.embedding_table.name], [char_embedding_meta], checkpoint_state.model_checkpoint_path)
                    log('Tacotron Character embeddings have been updated on tensorboard!')

            log('Tacotron training complete after {} global steps!'.format(args.tacotron_train_steps), slack=True)
            return save_dir

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)

def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)
