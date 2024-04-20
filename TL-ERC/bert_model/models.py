import torch
import torch.nn as nn
from util import to_var, pad, normal_kl_div, normal_logpdf, bag_of_words_loss, to_bow, EOS_ID
import layer
import numpy as np
import random

from pytorch_pretrained_bert.modeling import BertModel
from transformers import Wav2Vec2Model,Wav2Vec2Processor,Wav2Vec2FeatureExtractor

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

class Wav2Vec(nn.Module):
    """Simple initial model that attaches classifier to wav2vec"""

    def __init__(self, config):
        super(Wav2Vec, self).__init__()


        #load pretrained model & freeze feature extractor, as these CNN layers have been sufficiently pretrained
        # self.feat_ext = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
        #                                          do_normalize=True, return_attention_mask=False)
        self.config = config
        self.processor = Wav2Vec2Processor.from_pretrained(config.audio_base_model)
        self.base_model = Wav2Vec2Model.from_pretrained(config.audio_base_model)
        self.base_model.feature_extractor._freeze_parameters()

        #add dropout and FC layer for classification
        self.dropout = nn.Dropout(config.audio_dropout)
        self.fc = nn.Linear(self.base_model.config.hidden_size, config.num_classes)

    def pool(self, hidden):
        if self.config.audio_pooling == 'mean':
            return torch.mean(hidden, dim=1)
        elif self.config.audio_pooling == 'sum':
            return torch.sum(hidden, dim=1)
        elif self.config.audio_pooling == 'max':
            return torch.max(hidden, dim=1)[0]

    def forward(self, input):
        processed = self.processor(input,sampling_rate=16000, return_tensors='pt', padding=True)
        hidden = self.pool(self.base_model(processed.input_values)['last_hidden_state'])
        hidden = self.dropout(hidden)
        return self.fc(hidden)
