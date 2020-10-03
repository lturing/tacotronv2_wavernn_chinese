import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tacotron.datasets import audio
import sys
from tacotron.pinyin.parse_text_to_pyin import get_pyin
import glob 
import re 

def build_from_path_v1(hparams, input_dirs, out_dir, n_jobs=12, tqdm=lambda x: x):

    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    wav_path = os.path.join(input_dirs, 'wav', '*.wav')
    for wav in glob.glob(wav_path):
        trn = wav + '.txt'
        if os.path.exists(trn):
            text = open(trn, 'r', encoding='utf-8').readline().strip()
            text = re.sub(r'\s+', '', text)
            pyin, txt = get_pyin(text)
            basename = os.path.basename(wav).split('.')[0]
            futures.append(executor.submit(partial(_process_utterance_v1, out_dir, basename, wav, txt, pyin, hparams)))

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance_v1(out_dir, index, wav_path, text, pyin, hparams):

    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    #Trim lead/trail silences
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    #Pre-emphasize
    preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

    #rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        #Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    if hparams.use_lws:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        #Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

        #Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = os.path.join(out_dir, 'audio-{}.npy'.format(index))
    mel_filename = os.path.join(out_dir, 'mel-{}.npy'.format(index))
    
    np.save(audio_filename, out.astype(out_dtype), allow_pickle=False)
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, time_steps, mel_frames, text, pyin)


