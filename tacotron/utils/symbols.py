'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
#import sys
#sys.path.append('./../../')
import os 
from tacotron_hparams import hparams 

chars = set()
train_input = hparams.tacotron_input if os.path.exists(hparams.tacotron_input) else './train.txt'
with open(train_input, 'r', encoding='utf-8') as f:
    for line in f:
        words = line.strip().split('|')[-1].strip().split(' ')
        for w in words:
            chars.add(w)

chars = list(chars)
chars.sort()

_pad = '_'
_eos = '~'
#_sos = '#'

#symbols = [_pad, _sos, _eos] + chars 
symbols = [_pad, _eos] + chars

'''
from . import cmudict

_pad        = '_'
_eos        = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet
'''
