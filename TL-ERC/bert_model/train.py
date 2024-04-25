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


def summarize_run(solver):
    out = []
    for m in solver.models:
        out.append(pd.Series([m.best_epoch,
                              np.min(m.epoch_loss), np.min(m.val_epoch_loss),
                              np.max(m.w_train_f1), np.max(m.w_valid_f1)]).rename(m.__name__)
                   )
    return out


if __name__ == '__main__':

    # multi = get_config(mode='train',parse=False, modalities=['con'], run_name='TEST_MULTI',
    #                    visual_checkpoint = "../generative_weights/best_visual_model.pth",
    #                    audio_checkpoint = "../generative_weights/best_audio_model.pth",
    #                    text_checkpoint = "../generative_weights/best_text_model.pth",
    #                    )

    # visual = get_config(mode='train',parse=False, modalities=['visual'], run_name='VISUAL_TEST')
    # audio_lstm = get_config(mode='train',parse=False, modalities=['audio'], run_name='AUDIO_LSTM', audio_rnn='LSTM')

    concatenated = get_config(
        mode='train',
        parse=False,
        modalities=['concat'],
        run_name='Concat_with_RNN',
        batch_size=4,
        n_epoch=250,
        concat_dropout=0.1,
        concat_learning_rate=1e-4,
        concat_rnn='LSTM',
        visual_checkpoint="../generative_weights/best_visual_model.pth",
        audio_checkpoint="../generative_weights/best_audio_model.pth",
    )

    # hybrid = get_config(
    #     mode='train',
    #     parse=False,
    #     modalities=['hybrid'],
    #     run_name='HYBRID_TEST',
    #     text_checkpoint="../generative_weights/cornell_weights.pkl",
    #     concat_checkpoint="../generative_weights/best_concat_model.pth",
    # )

    configs = [concatenated]
    val_config = get_config(mode='valid', parse=False)
    test_config = get_config(mode='test', parse=False)

    _RUNS = 10
    for config in configs:
        summary = {}
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
                audioRaw=load_pickle(config.audioRaw_path)[:training_data_len],
                visual=load_pickle(config.visual_path)[:training_data_len],
                batch_size=config.batch_size)

            eval_data_loader = get_loader(
                sentences=load_pickle(val_config.sentences_path),
                labels=load_pickle(val_config.label_path),
                conversation_length=load_pickle(val_config.conversation_length_path),
                sentence_length=load_pickle(val_config.sentence_length_path),
                audio=load_pickle(val_config.audio_path),
                audioRaw=load_pickle(val_config.audioRaw_path),
                visual=load_pickle(val_config.visual_path),
                batch_size=val_config.eval_batch_size,
                shuffle=False)

            #commenting out for now to save memory since we arent using
            # test_data_loader = get_loader(
            #     sentences=load_pickle(test_config.sentences_path),
            #     labels=load_pickle(test_config.label_path),
            #     conversation_length=load_pickle(test_config.conversation_length_path),
            #     sentence_length=load_pickle(test_config.sentence_length_path),
            #     audio=load_pickle(test_config.audio_path),
            #     audioRaw=load_pickle(test_config.audioRaw_path),
            #     visual=load_pickle(test_config.visual_path),
            #     batch_size=test_config.eval_batch_size,
            #     shuffle=False)

            # for testing
            solver = Solver(config,
                            train_data_loader,
                            eval_data_loader,
                            test_data_loader=None,
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
            solver.train(run_directory=run_directory)
            summary[run] = pd.concat(summarize_run(solver))

        #save out summary across runs
        summary = pd.concat(summary, axis=1).T
        summary.columns = ['BestEpoch', 'BestTrainLoss', 'BestValLoss', 'BestTrainF1', 'BestValF1']
        summary.loc['AVG'] = summary.mean(axis=0)
        summary.loc['MED'] = summary.median(axis=0)
        summary.loc['MAX'] = summary.max(axis=0)
        summary.loc['MIN'] = summary.min(axis=0)
        summary.to_csv(outputs + "/summary.csv")
