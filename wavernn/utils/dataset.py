import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from wavernn.utils.dsp import *
from wavernn.utils import hparams as hp
from wavernn.utils.paths import Paths
from pathlib import Path
from wavernn.models.fatchord_version import WaveRNN



###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################

def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path: Path):

    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples: break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else:
            x = label_2_float(x, bits)

        save_wav(x, save_path/f'{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = str(save_path/f'{k}k_steps_{i}_{batch_str}.wav')

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)



class VocoderDataset(Dataset):
    def __init__(self, dataset, dataset_ids):
        self.metadata = dataset_ids
        self.dataset = dataset 

    def __getitem__(self, index):
        item_id = self.metadata[index]
        wav, mel = self.dataset[item_id]
        m = np.load(mel).T 
        x = np.load(wav) 
        return m, x

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(feature_path, batch_size, train_gta=None):

    with open(feature_path, 'r', encoding='utf-8') as f:
        dataset = []
        for line in f:
            line = line.strip().split('|')
            wav_path = line[0].strip() 
            wav_len = np.load(wav_path).shape[0]
            pred_mel = line[2].strip()
            mel_len = np.load(pred_mel).shape[0]

            mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
            if mel_len - (mel_win + 2 * hp.voc_pad + 2) < 0:
                continue 

            dataset.append((wav_path, pred_mel))

    dataset_ids = list(range(len(dataset)))

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(dataset, train_ids)
    test_dataset = VocoderDataset(dataset, test_ids)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set



def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels

