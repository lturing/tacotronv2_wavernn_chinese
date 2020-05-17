import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name)
    return log_dir, modified_hp


def train(args, log_dir, hparams):
    log('\n#############################################################\n')
    log('Tacotron Train\n')
    log('###########################################################\n')
    tacotron_train(args, log_dir, hparams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=hp.dataset) 
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--tacotron_input', default=hp.tacotron_input)
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--input_dir', default=hp.feat_out_dir, help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=1000,
        help='Steps between running summary ops')
    parser.add_argument('--embedding_interval', type=int, default=10000,
        help='Steps between updating embeddings projection visualization')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
        help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_steps', type=int, default=300000, help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    assert args.model == 'Tacotron-2', 'args.model != Tacotron-2'
    log_dir, hparams = prepare_run(args)
    
    train(args, log_dir, hparams)

if __name__ == '__main__':
    main()
