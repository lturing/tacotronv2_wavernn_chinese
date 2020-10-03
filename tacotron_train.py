import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from time import sleep

from tacotron.utils import infolog
import tensorflow as tf
from tacotron_hparams import hparams
from tacotron_hparams import hparams as hp 
from tacotron.utils.infolog import log
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train
import sys 

log = infolog.log


def train(args, log_dir, hparams):
    log('\n#############################################################\n')
    log('Tacotron Train\n')
    log('###########################################################\n')
    tacotron_train(args, log_dir, hparams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_interval', type=int, default=100,
        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
        help='Steps between writing checkpoints')
    parser.add_argument('--tacotron_train_steps', type=int, default=3000, help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--model', default='Tacotron-2')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    log_dir = os.path.join('logs-{}'.format(args.model), hparams.dataset)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), args.model)

    train(args, log_dir, hparams)


if __name__ == '__main__':
    main()
