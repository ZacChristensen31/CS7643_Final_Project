from solver import *
from data_loader import get_loader
from configs import get_config
from util import Vocab
import os
import pickle
from datetime import datetime


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='train',data='iemocap',parse=False,run_name='TextOnly_Pretrained')
    val_config = get_config(mode='valid',data='iemocap',parse=False,text_checkpoint=None)
    test_config = get_config(mode='test',data='iemocap',parse=False,text_checkpoint=None,)

    _RUNS = 10

    _best_test_loss, _best_test_f1_w, _best_test_f1_m, _best_epoch = [], [], [], []

    for run in range(_RUNS):

        print(config)
        # No. of videos to consider
        training_data_len = int(config.training_percentage * \
            len(load_pickle(config.sentences_path)))

        train_data_loader = get_loader(
            sentences=load_pickle(config.sentences_path)[:training_data_len],
            labels=load_pickle(config.label_path)[:training_data_len],
            conversation_length=load_pickle(config.conversation_length_path)[:training_data_len],
            sentence_length=load_pickle(config.sentence_length_path)[:training_data_len],
            audio=load_pickle(config.audio_path)[:training_data_len],
            audioWav2Vec=load_pickle(config.audio_wav2vec_path)[:training_data_len],
            visual=load_pickle(config.visual_path)[:training_data_len],
            batch_size=config.batch_size)

        eval_data_loader = get_loader(
            sentences=load_pickle(val_config.sentences_path),
            labels=load_pickle(val_config.label_path),
            conversation_length=load_pickle(val_config.conversation_length_path),
            sentence_length=load_pickle(val_config.sentence_length_path),
            audio=load_pickle(val_config.audio_path),
            audioWav2Vec=load_pickle(val_config.audio_wav2vec_path),
            visual=load_pickle(val_config.visual_path),
            batch_size=val_config.eval_batch_size,
            shuffle=False)
        
        test_data_loader = get_loader(
            sentences=load_pickle(test_config.sentences_path),
            labels=load_pickle(test_config.label_path),
            conversation_length=load_pickle(test_config.conversation_length_path),
            sentence_length=load_pickle(test_config.sentence_length_path),
            audio=load_pickle(test_config.audio_path),
            audioWav2Vec=load_pickle(test_config.audio_wav2vec_path),
            visual=load_pickle(test_config.visual_path),
            batch_size=test_config.eval_batch_size,
            shuffle=False)

        # for testing
        solver = Solver(config,
                        train_data_loader,
                        eval_data_loader,
                        test_data_loader,
                        is_train=True,
                        models=[])

        outputs = '../run_outputs'
        if config.run_name is not None:
            outputs += f"/{config.run_name}/"

        if not os.path.exists(outputs):
            os.makedirs(outputs)

        #append seed/run number
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = f'{outputs}/{run_timestamp}_Run{run}'

        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        # Save Configs for train, test, valid
        with open(f'{run_directory}/train_config.txt', 'w') as f:
            f.write(str(config))

        with open(f'{run_directory}/valid_config.txt', 'w') as f:
            f.write(str(val_config))

        with open(f'{run_directory}/test_config.txt', 'w') as f:
            f.write(str(test_config))

        solver.build()
        solver.train(
            run_directory=run_directory
        )
