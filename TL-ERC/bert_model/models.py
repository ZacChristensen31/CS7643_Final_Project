import torch
import torch.nn as nn
from util import to_var, pad, BIDIRECTIONAL_DIM
import layer
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from pytorch_pretrained_bert.modeling import BertModel
# from transformers import Wav2Vec2Model,Wav2Vec2Processor,Wav2Vec2FeatureExtractor

#from transformers import DistilBertModel, DistilBertConfig

class bc_RNN(nn.Module):
    def __init__(self, config):
        super(bc_RNN, self).__init__()

        self.config = config
        self.encoder = BertModel.from_pretrained("bert-base-uncased")   #pretrained bert transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size)

        self.context_encoder = layer.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.context2decoder = layer.FeedForward(config.context_size,
                                                  config.num_layers * config.context_size,
                                                  num_layers=1,
                                                  activation=config.activation,
                                                  isActivation=True)
        
        self.decoder2output = layer.FeedForward(config.num_layers * config.context_size,
                                                 config.num_classes,
                                                 num_layers=1,
                                                 isActivation=False)
        self.dropoutLayer = nn.Dropout(p=config.dropout)

    def forward(self, input_sentences, input_sentence_length, input_conversation_length, input_masks):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_len = input_conversation_length.max().item()


        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        # encoder_outputs, encoder_hidden = self.encoder(input_sentences,
        #                                                input_sentence_length)
        all_encoder_layers, _ = self.encoder(input_sentences, token_type_ids=None, attention_mask=input_masks)


        bert_output = []
        for idx in range(self.config.num_bert_layers):
          layer = all_encoder_layers[idx]
          bert_output.append(layer[:,0,:])
        bert_output = torch.stack(bert_output, dim=1)
        bert_output = torch.mean(bert_output, dim=1, keepdim=False)

        

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = bert_output

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), max_len, self.device)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # context_outputs: [batch_size, max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden,
                                                                    input_conversation_length)


        # flatten images
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        context_outputs = self.dropoutLayer(context_outputs)

        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        output = self.decoder2output(decoder_init)

        return output

# class Wav2Vec(nn.Module):
#     """Simple initial model that attaches classifier to wav2vec
#     """
#
#     def __init__(self, config):
#         super(Wav2Vec, self).__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.config = config
#
#         #load pretrained model & freeze feature extractor, as these CNN layers have been sufficiently pretrained
#         try:
#             self.processor = Wav2Vec2Processor.from_pretrained(config.audio_base_model)
#             self.base_model = Wav2Vec2Model.from_pretrained(config.audio_base_model)
#         except:
#             self.processor = Wav2Vec2Processor.from_pretrained(r"C:\Users\jglicksm\git\CS7643_Final_Project\TL-ERC\WAV2VEC")
#             self.base_model = Wav2Vec2Model.from_pretrained(r"C:\Users\jglicksm\git\CS7643_Final_Project\TL-ERC\WAV2VEC")
#
#         self.base_model.feature_extractor._freeze_parameters()
#         self.hidden_size = self.base_model.config.hidden_size
#
#         #add optional RNN context layer across sentences
#         if config.audio_rnn is not None:
#             self.rnn = getattr(nn, config.audio_rnn.upper())(input_size=self.hidden_size,
#                                                              hidden_size=self.hidden_size//2,
#                                                              num_layers=config.audio_num_layers,
#                                                              bidirectional=config.audio_bidirectional,
#                                                              batch_first=True)
#
#         #add dropout and FC layer for classification
#         self.fc1 = nn.Linear(self.hidden_size, 256)
#         self.dropout = nn.Dropout(config.audio_dropout)
#         self.act = nn.Tanh()
#         self.fc2 = nn.Linear(256, config.num_classes)
#
#     def pool(self, hidden):
#         if self.config.audio_pooling == 'mean':
#             return torch.mean(hidden, dim=1)
#         elif self.config.audio_pooling == 'sum':
#             return torch.sum(hidden, dim=1)
#         elif self.config.audio_pooling == 'max':
#             return torch.max(hidden, dim=1)[0]
#
#     def get_init_h(self, batch):
#         bidir = BIDIRECTIONAL_DIM[self.config.audio_bidirectional]
#         return to_var(torch.zeros(self.config.audio_num_layers*BIDIRECTIONAL_DIM[self.config.audio_bidirectional],
#                                   batch, self.hidden_size//bidir))
#
#     def pack_input_seq(self, seq_input, lengths):
#         """
#         Align sequences into batch form such that each batch
#         represents a conversation. Pack and pad data to match longest
#         conv length while minimizing compute on padded items
#         """
#         sequences, start = [], -0
#         for length in lengths:
#             sequences.append(seq_input[start:start + length])
#             start += length
#         padded_seq = pad_sequence(sequences, batch_first=True)
#         packed_input = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)
#         return packed_input
#
#     def forward(self, input, seq_lengths):
#         hidden = self.pool(self.base_model(input)['last_hidden_state'])
#
#         if self.config.audio_rnn is not None:
#
#             #pack features into aligned/padded convo batches, then
#             #flatten back to sentence level after context rnn
#             packed_input = self.pack_input_seq(hidden, seq_lengths)
#             hidden,_ = self.rnn(packed_input, self.get_init_h(len(seq_lengths)))
#             hidden,_ = pad_packed_sequence(hidden, batch_first=True)
#             hidden = torch.cat([hidden[i, :length] for i, length in enumerate(seq_lengths)], dim=0)
#
#         output = self.fc1(self.dropout(hidden))
#         output = self.act(output)
#         return self.fc2(output)


class ContextClassifier(nn.Module):
    """
    Simple model that attaches classification head to processed
    features, with optional context RNN across conversations
    """

    def __init__(self, config, modality):
        super(ContextClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.modality = modality
        self.input_dim = self.get_attr('input_dim')
        self.dir_dim = BIDIRECTIONAL_DIM[self.get_attr('bidirectional')]

        # add optional RNN context layer across sentences
        if self.get_attr('rnn') is not None:
            self.hidden_size = self.get_attr('hidden_size')
            self.rnn = getattr(nn, self.get_attr('rnn').upper())(input_size=self.input_dim,
                                                             hidden_size=self.hidden_size // self.dir_dim,
                                                             num_layers=self.get_attr('num_layers'),
                                                             bidirectional=self.get_attr('bidirectional'),
                                                             batch_first=True)
        else:
            self.hidden_size = self.input_dim

        # dropout and FC layer for classification
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.dropout = nn.Dropout(self.get_attr('dropout'))
        self.act = getattr(nn.functional, self.get_attr('activation'))
        self.fc2 = nn.Linear(256, config.num_classes)

    def get_attr(self,param):
        return getattr(self.config, f'{self.modality}_{param}')

    def get_init_h(self, batch):
        h = to_var(torch.zeros(self.get_attr('num_layers') * self.dir_dim,
                               batch, self.hidden_size // self.dir_dim))
        if self.get_attr('rnn').upper()=='GRU':
            return h
        elif self.get_attr('rnn').upper()=='LSTM':
            return (h, h)  #init hidden and cell state

    def pack_input_seq(self, seq_input, lengths):
        """
        Align sequences into batch form such that each batch
        represents a conversation. Pack and pad data to match longest
        conv length while minimizing compute on padded items
        """
        sequences, start = [], -0
        for length in lengths:
            sequences.append(seq_input[start:start + length])
            start += length
        padded_seq = pad_sequence(sequences, batch_first=True)
        packed_input = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)
        return packed_input

    def forward(self, input, seq_lengths):

        if self.get_attr('rnn') is not None:
            # pack features into aligned/padded convo batches, then
            packed_input = self.pack_input_seq(input, seq_lengths)
            hidden, _ = self.rnn(packed_input, self.get_init_h(len(seq_lengths)))
            hidden, _ = pad_packed_sequence(hidden, batch_first=True)

            #flatten to sentence level, rename as input for compatibility w no-rnn scenario
            input = torch.cat([hidden[i, :length] for i, length in enumerate(seq_lengths)], dim=0)

        output = self.fc1(self.dropout(input))
        output = self.act(output)
        return self.fc2(output)

class ConcatenatedClassifier(nn.Module):
    """
        Since ContextClassifier can handle different modalities, this is going to mimic the same functionality
        but with the ability to concatenate multiple modalities together before classification
        --> early fusion model
    """
    def __init__(self, config):

        super(ConcatenatedClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_dim = np.sum([getattr(config, f'{i}_input_dim')
                                 for i in self.config.early_fusion_modalities])
        self.dir_dim = BIDIRECTIONAL_DIM[getattr(config, 'early_fusion_bidirectional')]

        if config.early_fusion_rnn is not None:
            self.early_fusion_hidden_size = getattr(config, 'early_fusion_hidden_size')
            self.rnn = getattr(nn, config.early_fusion_rnn.upper())(input_size=self.input_dim,
                                                             hidden_size=self.early_fusion_hidden_size // self.dir_dim,
                                                             num_layers=getattr(config, 'early_fusion_num_layers'),
                                                             bidirectional=getattr(config, 'early_fusion_bidirectional'),
                                                             batch_first=True)
        else:
            self.early_fusion_hidden_size = self.input_dim

        # Deal with Audio RNN
        self.fc1 = nn.Linear(self.early_fusion_hidden_size, self.early_fusion_hidden_size)
        self.dropout1 = nn.Dropout(getattr(config, 'early_fusion_dropout'))
        self.act1 = getattr(nn.functional, getattr(config, 'early_fusion_activation'))
        self.fc2 = nn.Linear(self.early_fusion_hidden_size, self.early_fusion_hidden_size)
        self.dropout2 = nn.Dropout(getattr(config, 'early_fusion_dropout'))
        self.act2 = getattr(nn.functional, getattr(config, 'early_fusion_activation'))
        self.fc3 = nn.Linear(self.early_fusion_hidden_size, config.num_classes)

    def get_init_h(self, batch):
        h = to_var(torch.zeros(self.config.early_fusion_num_layers * self.dir_dim,
                               batch, self.early_fusion_hidden_size // self.dir_dim))
        if self.config.early_fusion_rnn.upper()=='GRU':
            return h
        elif self.config.early_fusion_rnn.upper()=='LSTM':
            return (h, h)  #init hidden and cell state

    def pack_input_seq(self, seq_input, lengths):
        """
        Align sequences into batch form such that each batch
        represents a conversation. Pack and pad data to match longest
        conv length while minimizing compute on padded items
        """
        sequences, start = [], -0
        for length in lengths:
            sequences.append(seq_input[start:start + length])
            start += length
        padded_seq = pad_sequence(sequences, batch_first=True)
        packed_input = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)
        return packed_input

    def forward(self, input_data, seq_lengths):

        if self.config.early_fusion_rnn is not None:
            packed_input = self.pack_input_seq(input_data, seq_lengths)
            hidden, _ = self.rnn(packed_input, self.get_init_h(len(seq_lengths)))
            hidden, _ = pad_packed_sequence(hidden, batch_first=True)
            input_data = torch.cat([hidden[i, :length] for i, length in enumerate(seq_lengths)], dim=0)

        input_data = self.fc1(input_data)
        input_data = self.dropout1(input_data)
        input_data = self.act1(input_data)
        input_data = self.fc2(input_data)
        input_data = self.dropout2(input_data)
        input_data = self.act2(input_data)
        return self.fc3(input_data)

class MLP(nn.Module):
    """
    Classification head on concatenated multi-modal predictors
    ie, late fusion model
    """
    def __init__(self, config, input_dim):
        super(MLP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.fc1 = nn.Linear(input_dim, config.late_fusion_hidden_size)
        self.dropout = nn.Dropout(config.late_fusion_dropout)
        self.act = getattr(nn.functional, config.late_fusion_activation)
        self.fc2 = nn.Linear(config.late_fusion_hidden_size, config.num_classes)

    def forward(self, input):
        input = self.act(self.fc1(input))
        input = self.dropout(input)
        return self.fc2(input)