from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from util import to_var, time_desc_decorator, flat_to_var, check_done
import os
from tqdm import tqdm
from math import isnan
import re
import math
import pickle
import gensim
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
from datetime import datetime


class Model(ABC):

    def __init__(self, config):
        self.config = config
        self.model = None
        self.done = False
        self.result_reset()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    @check_done
    def train(self, data):
        pass

    @abstractmethod
    @check_done
    def evaluate(self, data):
        pass

    @abstractmethod
    def checkpoint(self, data):
        pass

    def load_model(self):
        print(f'Load parameters from {self.checkpoint}')
        if torch.cuda.is_available():
            pretrained_dict = torch.load(self.checkpoint)
        else:
            pretrained_dict = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        model_dict = self.model.state_dict()

        # filter out unnecessary keys
        filtered_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if (k in model_dict) and ("embedding" not in k) and ("context" in k) and ("ih" not in k):
                filtered_pretrained_dict[k] = v

        # overwrite entries in the existing state dict + load new state dict
        model_dict.update(filtered_pretrained_dict)
        self.model.load_state_dict(model_dict)

    def result_reset(self):
        self.min_val_loss = np.inf
        self.patience = 0
        self.epoch_i = 0
        self.best_epoch = -1
        self.epoch_loss = []
        self.val_epoch_loss = []
        self.w_train_f1 = []
        self.w_valid_f1 = []
        self.batch_loss_history = []
        self.val_batch_loss_history = []
        self.predictions = []
        self.ground_truth = []
        self.val_predictions = []
        self.val_ground_truth = []

    def epoch_reset(self, epoch_i):
        """performance tracking -- could move this outside model """

        # update end of epoch stats
        if len(self.batch_loss_history) > 0:
            curr_val_loss = np.mean(self.val_batch_loss_history)
            self.epoch_loss.append(np.mean(self.batch_loss_history))
            self.w_train_f1.append(self.print_metric(self.ground_truth, self.predictions, "train"))
            self.val_epoch_loss.append(curr_val_loss)
            self.w_valid_f1.append(self.print_metric(self.val_ground_truth, self.val_predictions, "train"))

            if curr_val_loss < self.min_val_loss:
                self.min_val_loss = curr_val_loss
                self.best_epoch = self.epoch_i + 1  # from zero start
                self.patience = 0
            else:
                self.patience += 1

            # trigger early stopping
            if self.patience > self.config.patience:
                self.done = True

        # reset epoch trackers
        self.epoch_i = epoch_i
        self.batch_loss_history = []
        self.predictions = []
        self.ground_truth = []
        self.val_batch_loss_history = []
        self.val_predictions = []
        self.val_ground_truth = []

    def plot_results(self, typ='Loss', image_directory=None, epoch_num=None):
        """save plots?"""
        if typ == 'Loss':
            tr, val = self.epoch_loss, self.val_epoch_loss
        elif typ == 'F1':
            tr, val = self.w_train_f1, self.w_valid_f1
        else:
            print('Invalid Type')
            return

        plt.plot(tr, label=f'Train{typ}')
        plt.plot(val, label=f'Valid{typ}')
        plt.title(f'{self.__name__} {typ} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(typ)
        plt.legend()

        if image_directory is not None:
            image_name = f'{image_directory}/{typ}'

            if epoch_num:
                image_name += f'_{epoch_num}'

            plt.savefig(f'{image_name}.png', format='png')

        plt.clf()

    def save_epoch_results(self, epoch_num, run_directory):
        """save epoch results to file"""

        file_path = f'{run_directory}/results.csv'

        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('epoch,train_loss,valid_loss,train_f1,valid_f1\n')

        with open(f'{run_directory}/results.csv', 'a+') as f:
            f.write(f'{epoch_num},{self.epoch_loss[-1]},{self.val_epoch_loss[-1]},{self.w_train_f1[-1]},{self.w_valid_f1[-1]}\n')

    @staticmethod
    def print_metric(y_true, y_pred, mode):
        if mode in ["train", "test"]:
            print(mode)
            print(classification_report(y_true, y_pred, digits=4, zero_division=0.0))
        weighted_fscore = \
        classification_report(y_true, y_pred, output_dict=True, digits=4, zero_division=0.0)["weighted avg"]["f1-score"]
        return weighted_fscore


class TextModel(Model):
    __name__ = 'TextModel'

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """from TL-ERC"""
        if self.model is None:
            self.model = getattr(models, self.config.text_model)(self.config)

            # orthogonal initialiation for hidden weights, input gate bias for GRUs
            if self.config.mode == 'train' and self.config.text_checkpoint is None:
                # Make later layers require_grad = False
                for name, param in self.model.named_parameters():
                    if "encoder.encoder.layer" in name:
                        layer_num = int(name.split("encoder.encoder.layer.")[-1].split(".")[0])
                        if layer_num >= (self.config.num_bert_layers):
                            param.requires_grad = False

                print('Parameter initialization')
                for name, param in self.model.named_parameters():
                    if ('weight_hh' in name) and ("encoder.encoder" not in name):
                        nn.init.orthogonal_(param)

        if torch.cuda.is_available():
            self.model.cuda()

        # Overview Parameters
        print('Text Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.checkpoint is not None:
            self.load_model()

        # I removed "is_train" check here because seemed unused?
        self.optimizer = self.config.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate)

    @property
    def checkpoint(self):
        return self.config.text_checkpoint

    def train(self, data):
        """ from TL-ERC """
        self.model.train()

        # unpack, flatten, to cuda
        (conversations, labels, conversation_length, sentence_length, _, _, type_ids, masks) = data
        input_conversations = conversations
        orig_input_labels = [i for item in labels for i in item]

        input_sentences = flat_to_var(input_conversations)
        input_labels = flat_to_var(labels)
        input_sentence_length = flat_to_var(sentence_length)
        input_conversation_length = to_var(torch.LongTensor([l for l in conversation_length]))
        input_masks = flat_to_var(masks)

        # reset gradient
        self.optimizer.zero_grad()
        sentence_logits = self.model(input_sentences,
                                     input_sentence_length,
                                     input_conversation_length,
                                     input_masks)

        present_predictions = list(np.argmax(sentence_logits.detach().cpu().numpy(), axis=1))
        loss_function = nn.CrossEntropyLoss()
        batch_loss = loss_function(sentence_logits, input_labels)
        self.predictions += present_predictions
        self.ground_truth += orig_input_labels
        assert not isnan(batch_loss.item())
        self.batch_loss_history.append(batch_loss.item())

        # Back-prop, clip, step
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
        self.optimizer.step()

    def evaluate(self, data):
        self.model.eval()
        # unpack and flatten inputs
        (conversations, labels, conversation_length, sentence_length, _, _, type_ids, masks) = data
        input_conversations = conversations
        orig_input_labels = [i for item in labels for i in item]

        with torch.no_grad():
            input_sentences = flat_to_var(input_conversations)
            input_labels = flat_to_var(labels)
            input_sentence_length = flat_to_var(sentence_length)
            input_conversation_length = to_var(torch.LongTensor([l for l in conversation_length]))
            input_masks = flat_to_var(masks)

        sentence_logits = self.model(input_sentences,
                                     input_sentence_length,
                                     input_conversation_length,
                                     input_masks)

        present_predictions = list(np.argmax(sentence_logits.detach().cpu().numpy(), axis=1))
        loss_function = nn.CrossEntropyLoss()
        batch_loss = loss_function(sentence_logits, input_labels)

        self.val_predictions += present_predictions
        self.val_ground_truth += orig_input_labels
        assert not isnan(batch_loss.item())
        self.val_batch_loss_history.append(batch_loss.item())


class AudioModel(Model):
    __name__ = 'AudioModel'

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        pass

    def train(self, data):
        if self.done:
            pass
        self.model.train()
        pass

    def evaluate(self, data):
        self.model.eval()
        pass

    @property
    def checkpoint(self):
        return self.config.audio_checkpoint


class VisualModel(Model):
    __name__ = 'VisualModel'

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        pass

    def train(self, data):
        self.model.train()
        pass

    def evaluate(self, data):
        self.model.eval()
        pass

    @property
    def checkpoint(self):
        return self.config.visual_checkpoint


class CombinedModel(Model):
    """ runs concatenated features through same model rather than train separately"""
    __name__ = 'CombinedModel'

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        pass

    def train(self, data):
        self.model.train()
        pass

    def evaluate(self, data):
        self.model.eval()
        pass

    @property
    def checkpoint(self):
        return self.config.combined_checkpoint


class Solver(object):
    def __init__(self, config, train_data_loader, valid_data_loader, test_data_loader, is_train=True, models=[]):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.models = models

    def build(self):
        """initiate models for all modalities in config"""
        if len(self.models) == 0:
            for mod in self.config.modalities:
                if mod == 'text':
                    self.models.append(TextModel(self.config))
                elif mod == 'audio':
                    print("Audio model not implemented yet -- setting empty")
                    self.models.append(AudioModel(self.config))
                elif mod == 'visual':
                    print("Visual model not implemented yet -- setting empty")
                    self.models.append(VisualModel(self.config))
                elif mod == 'combined':
                    print("Combined model not implemented yet -- setting empty")
                    self.models.append(CombinedModel(self.config))

            for model in self.models:
                model.build()

    # @time_desc_decorator('Training Start!')
    def train(self, run_directory=None):
        """
        Don't love all this looping -- could theoretically trigger models in parallel
        but I don't have the compute for that.

        Alternatively, init each model with a loader and execute training separately,
        but thought managing them together like this might give us flexibility to
        combine more freely?
        """

        # Set up image output directories
        images = f'{run_directory}/images'

        if not os.path.exists(images):
            os.mkdir(images)

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i

            # run training for each model
            for batch_i, data in enumerate(tqdm(self.train_data_loader, ncols=80)):
                for model in self.models: model.train(data)

            # run validation for each model
            for batch_i, data in enumerate(tqdm(self.valid_data_loader, ncols=80)):
                for model in self.models: model.evaluate(data)

            # not really supposed to be running your test set during training?
            # for batch_i, data in enumerate(tqdm(self.test_data_loader, ncols=80)):
            #     for model in self.models: model.evaluate(data)

            # track results, plot update, reset trackers for new epoch
            for model in self.models:
                model.epoch_reset(epoch_i)
                model.plot_results("Loss", image_directory=images, epoch_num=epoch_i)
                model.plot_results('F1', image_directory=images, epoch_num=epoch_i)
                model.save_epoch_results(epoch_i, run_directory=run_directory)
