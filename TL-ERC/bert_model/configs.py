import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
from layer.rnncells import StackedLSTMCell, StackedGRUCell

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {'iemocap': data_dir.joinpath('iemocap') , 'dailydialog': data_dir.joinpath('dailydialog')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}
username = Path.home().name

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                if key == 'rnncell':
                    value = rnncell_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')
        self.word_emb_path = self.dataset_dir.joinpath('word_emb.pkl')

        # Pickled Dataframes
        self.sentences_path = self.data_dir.joinpath('sentences.pkl')
        self.label_path = self.data_dir.joinpath('labels.pkl')
        self.sentence_length_path = self.data_dir.joinpath('sentence_length.pkl')
        self.conversation_length_path = self.data_dir.joinpath('conversation_length.pkl')
        self.audio_path = self.data_dir.joinpath('audio.pkl')
        self.bertText_path = self.data_dir.joinpath('bertText.pkl')
        self.visual_path = self.data_dir.joinpath('visuals.pkl')

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):

    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--modalities', default=['text'], type=list, nargs='+')

    # Train
    parser.add_argument('--num_classes', type=int, default=0) 
    parser.add_argument('--batch_size', type=int, default=2)        #really small?
    parser.add_argument('--eval_batch_size', type=int, default=2)   #really small?
    parser.add_argument('--n_epoch', type=int, default=40)          #Usually stops early @ ~15-20
    parser.add_argument('--patience', type=int, default=5)          #lowered from 10 for efficiency during testing
    parser.add_argument('--minimum_improvement', type=int, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    #toggle whether to load pre-trained weights
    parser.add_argument('--text_checkpoint', type=str, default="../generative_weights/cornell_weights.pkl")
    # parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--num_bert_layers', type=int, default=4)
    parser.add_argument('--training_percentage', type=float, default=1.0)

    # Currently does not support lstm
    # TEXT MODEL PARAMETERS
    parser.add_argument('--text_model', type=str, default='bc_RNN')
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--encoder_hidden_size', type=int, default=768)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--train_emb', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--context_size', type=int, default=256)
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='Tanh')
    parser.add_argument('--text_input_dim', type=int, default=768)

    #AUDIO MODEL PARAMETERS
    parser.add_argument('--audio_checkpoint', type=str, default=None)
    parser.add_argument('--audio_model', type=str, default='ContextClassifier')
    # parser.add_argument('--audio_base_model', type=str, default="facebook/wav2vec2-base-960h")
    # parser.add_argument('--audio_pooling', type=str, default='max')
    parser.add_argument('--audio_activation', type=str, default='relu')
    parser.add_argument('--audio_input_dim', type=int, default=100)
    parser.add_argument('--audio_dropout', type=float, default=0.2)
    parser.add_argument('--audio_hidden_size', type=int, default=256)
    parser.add_argument('--audio_learning_rate', type=float, default=1e-3)
    # parser.add_argument('--audio_freeze_base', type=float, default=True)
    parser.add_argument('--audio_rnn', type=str, default='gru')
    parser.add_argument('--audio_bidirectional', type=str2bool, default=True)
    parser.add_argument('--audio_num_layers', type=int, default=1)


    #VISUAL MODEL PARAMETERS
    parser.add_argument('--visual_checkpoint', type=str, default=None)
    parser.add_argument('--visual_model', type=str, default='ContextClassifier')
    parser.add_argument('--visual_activation', type=str, default='relu')
    parser.add_argument('--visual_input_dim', type=int, default=512)
    parser.add_argument('--visual_dropout', type=float, default=0.2)
    parser.add_argument('--visual_hidden_size', type=int, default=256)
    parser.add_argument('--visual_learning_rate', type=float, default=1e-3)
    parser.add_argument('--visual_rnn', type=str, default='gru')
    parser.add_argument('--visual_bidirectional', type=str2bool, default=True)
    parser.add_argument('--visual_num_layers', type=int, default=1)

    # CONCATENATED MODEL PARAMETERS
    parser.add_argument('--concat_checkpoint', type=str, default=None)
    parser.add_argument('--concat_model', type=str, default='ConcatenatedClassifier')
    parser.add_argument('--concat_activation', type=str, default='relu')
    parser.add_argument('--concat_dropout', type=float, default=0.1)
    parser.add_argument('--concat_hidden_size', type=int, default=256)
    parser.add_argument('--concat_learning_rate', type=float, default=1e-3)
    parser.add_argument('--concat_rnn', type=str, default='gru')
    parser.add_argument('--concat_bidirectional', type=str2bool, default=True)
    parser.add_argument('--concat_num_layers', type=int, default=1)
    parser.add_argument('--concat_modalities', type=list, default=['text','audio','visual'], nargs='+')


    # HYBRID MODEL PARAMETERS
    parser.add_argument('--hybrid_checkpoint', type=str, default=None)
    parser.add_argument('--hybrid_model', type=str, default='MLP')
    parser.add_argument('--hybrid_activation', type=str, default='relu')
    parser.add_argument('--hybrid_dropout', type=float, default=0.1)
    parser.add_argument('--hybrid_hidden_dim', type=int, default=128)
    parser.add_argument('--hybrid_learning_rate', type=float, default=1e-3)


    #COMBINATION MODEL PARAMETERS
    #--> push raw features through single model together?
    #--> train independently and combine predictions?
    parser.add_argument('--combined_checkpoint', type=str, default=None)
    parser.add_argument('--combined_model', type=str, default='MLP')
    parser.add_argument('--combined_modalities', type=list, default=['text','audio','visual'], nargs='+')
    parser.add_argument('--combined_activation', type=str, default='relu')
    parser.add_argument('--combined_dropout', type=float, default=0.1)
    parser.add_argument('--combined_hidden_dim', type=int, default=128)
    parser.add_argument('--combined_learning_rate', type=float, default=1e-3)

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)
    parser.add_argument('--run_name', type=str, default=None)

    # Data
    parser.add_argument('--data', type=str, default='iemocap')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)
    if kwargs.data == "iemocap":
        kwargs.num_classes = 6
    elif kwargs.data == "dailydialog":
        kwargs.num_classes = 7
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
