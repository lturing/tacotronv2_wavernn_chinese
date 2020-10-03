import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


def split_title_line(title_text, max_words=30):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png', bbox_inches='tight')
    #plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if split_title:
        title = split_title_line(title)

    #target spectrogram subplot
    if target_spectrogram is not None:
        fig = plt.figure(figsize=(20, 16))
        fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram)[::-1, :], aspect='auto', interpolation='none', origin='lower')
        else:
            im = ax1.imshow(np.rot90(target_spectrogram)[::-1, :], interpolation='none', origin='lower')
        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        fig = plt.figure(figsize=(20, 6))
        fig.text(0.5, 0.95, title, horizontalalignment='center', fontsize=16)
        ax2 = fig.add_subplot()

    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram)[::-1, :], aspect='auto', interpolation='none', origin='lower')
    else:
        im = ax2.imshow(np.rot90(pred_spectrogram)[::-1, :], interpolation='none', origin='lower')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()
