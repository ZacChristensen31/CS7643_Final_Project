import pickle
import sys
sys.path.append(r"C:\Users\jrgli\git\DL\CS7643_Final_Project\TL-ERC")

from iemocap_preprocess import iemocap_pickle, IEMOCAP, project_dir
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torch

raw_iemocap_dir = r"C:\Users\jrgli\Downloads\IEMOCAP_full_release\IEMOCAP_full_release"

session_map = {'Ses01': 'Session1',
               'Ses02': 'Session2',
               'Ses03': 'Session3',
               'Ses04': 'Session4',
               'Ses05': 'Session5'}

#base instead of large model
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def process_wav(file_path):
    return sf.read(file_path)[0]

def extract_wav2vec_features(file_path):
    raw_audio = process_wav(file_path)
    inputs = processor(raw_audio, sampling_rate=16000, return_tensors='pt', padding=True)
    with torch.no_grad():
        features = model(inputs.input_values).last_hidden_state
    return features

def pool(hidden, method):
    if method == 'mean':
        return torch.mean(hidden, dim=1).squeeze()
    elif method == 'sum':
        return torch.sum(hidden, dim=1).squeeze()
    elif method == 'max':
        return torch.max(hidden, dim=1)[0].squeeze()

if __name__=='__main__':

    #Load existing iemocap data and add new audio variables
    iemocap = IEMOCAP()
    iemocap.load_iemocap_data()
    iemocap.audioRaw, iemocap.audioWav2vec = {}, {}
    max_audio_len = 50_000
    pool_method = 'mean'

    #iterate through raw iemocap data and store waveforms + frozen wav3vec features
    for vid,ids in iemocap.sent_ids.items():
        print(vid)
        dir = f"{raw_iemocap_dir}\\{session_map[vid[:5]]}\\sentences\\wav\\{vid}\\"
        iemocap.audioRaw[vid] = [process_wav(f"{dir}\\{id}.wav")[:max_audio_len] for id in ids]
        #iemocap.audioWav2vec[vid] = [pool(extract_wav2vec_features(f"{dir}\\{id}.wav"),pool_method)
                                     # for id in ids]


    #resave raw feature pickle with new audio features
    with open(r"C:\Users\jrgli\git\DL\CS7643_Final_Project\TL-ERC\datasets\iemocap\audio_data_2.pkl", "wb") as handle:
        data = (iemocap.sent_ids,
             iemocap.videoSpeakers,
             iemocap.videoLabels,
             iemocap.text_data,
             iemocap.videoAudio,
             iemocap.videoVisual,
             iemocap.videoSentence,
             iemocap.trainVid_raw,
             iemocap.testVid)
        pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)

    keys = list(iemocap.audioRaw.keys())
    path = r"C:\Users\jrgli\git\DL\CS7643_Final_Project\TL-ERC\datasets\iemocap\audio"
    for j,i in enumerate(range(0, len(keys)-4, 4)):
        cur_k = keys[i:i+4]
        if i == 144:
            cur_k = keys[i:]
        data = {k: iemocap.audioRaw[k] for k in cur_k}
        with open(path+f"\\audio_{j}.pkl","wb") as handle:
            pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)



