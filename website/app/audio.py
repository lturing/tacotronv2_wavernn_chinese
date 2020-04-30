import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

def dc_notch_filter(wav):
    # code from speex
    notch_radius = 0.982
    den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
    b = np.array([1, -2, 1]) * notch_radius
    a = np.array([1, -2 * notch_radius, den])
    return signal.lfilter(b, a, wav)


def save_wav(wav):
    wav = dc_notch_filter(wav)
    wav = wav / np.abs(wav).max() * 0.999
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2

    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wav = wav.astype(np.int16)
    
    wavfile.write('./test.wav', 22050, wav)
    #proposed by @dsmiller
    return wav 


def save_wav_old(wav):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #wav = np.clip(wav, -2**15, 2**15-1)
    #proposed by @dsmiller
    return wav 


def inv_preemphasis(wav):
    k = 0.97
    return signal.lfilter([1], [1, -k], wav)

def _denormalize(D):
    max_abs_value = 4
    min_level_db = -100
    return (((np.clip(D, -max_abs_value, max_abs_value) + max_abs_value) * -min_level_db / (2 * max_abs_value))
                + min_level_db)


def _build_mel_basis():
    sample_rate = 22050
    n_fft = 2048
    num_mels = 80
    fmin = 95
    fmax = 7600

    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)


def _mel_to_linear(mel_spectrogram):
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def inv_mel_spectrogram(mel_spectrogram):
    '''Converts mel spectrogram to waveform using librosa'''
    #print("mel_spectrogram", np.isnan(mel_spectrogram).any(), mel_spectrogram.shape)
    D = _denormalize(mel_spectrogram)
    ref_level_db = 20
    magnitude_power = 2.
    power = 1.5

    #print("D", np.isnan(D).any())

    S = _mel_to_linear(_db_to_amp(D + ref_level_db)**(1/magnitude_power))  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** power))


def _stft(y):
    hop_size = 275
    win_size = 1100
    n_fft = 2048
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, pad_mode='constant')

def _istft(y):
    hop_size = 275
    win_size = 1100
    return librosa.istft(y, hop_length=hop_size, win_length=win_size)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(60):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

