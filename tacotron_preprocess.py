import argparse
import os
from multiprocessing import cpu_count

from tacotron.datasets import preprocessor
from tacotron_hparams import hparams, hparams_debug_string
from tqdm import tqdm
import shutil 

def preprocess(input_folders, out_dir, hparams):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    os.makedirs(out_dir, exist_ok=True)

    metadata = preprocessor.build_from_path_v1(hparams, input_folders, out_dir, cpu_count() * 2, tqdm=tqdm)
    write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
    with open(hparams.tacotron_input, 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    
    mel_frames = sum([int(m[3]) for m in metadata])
    timesteps = sum([int(m[2]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    #audio_filename, mel_filename, time_steps, mel_frames, text, pyin
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[4]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[3]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[2] for m in metadata)))



if __name__ == '__main__':
    print(hparams_debug_string())

    input_folders = hparams.dataset
    output_folder = os.path.join(hparams.dataset, hparams.feat_out_dir)

    preprocess(input_folders, output_folder, hparams)
