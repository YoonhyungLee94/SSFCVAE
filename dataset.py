import os
from tqdm import tqdm
import librosa
import torch
import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler

DV = "./Dataset/VoiceBank-DEMAND"
def load_numpy(x, kind):
    x_split = x.split('/')
    file_path = f"{DV}/preprocessed/{x_split[-2].replace('wav', kind)}/{x_split[-1][:8] +'.npy'}"
    return torch.from_numpy(np.load(file_path))


class AudioSet(torch.utils.data.Dataset):
    def __init__(self, data_path, w2v_clean_scaler=None, w2v_noisy_scaler=None):
        data_list = [os.path.join(sub_path, wav) for sub_path in data_path for wav in os.listdir(sub_path)]
        data_list = sorted(data_list, key=lambda x : x.split('/')[-1])
        
        train_scaler = (w2v_clean_scaler is None)
        if train_scaler:
            self.w2v_clean_scaler = StandardScaler()
            self.w2v_noisy_scaler = StandardScaler()
        else:
            self.w2v_clean_scaler = w2v_clean_scaler
            self.w2v_noisy_scaler = w2v_noisy_scaler

        self.ground_truth_list = []
        self.mel_clean_list = []
        self.mel_noisy_list = []
        self.spec_noisy_list = []
        self.w2v_feats_clean_list = []
        self.w2v_feats_noisy_list = []
        for x in tqdm(data_list):
            wav_clean, _ = librosa.load(f"{DV}/{x.split('/')[-2]}/{x.split('/')[-1]}", sr=22050)
            if len(wav_clean)%256!=0:
                wav_clean = wav_clean[:-(len(wav_clean)%256)]
                
            if (train_scaler==True) and (len(wav_clean) > 220500): # drop longer than 10sec
                continue
            
            wav_clean = librosa.resample(wav_clean, orig_sr=22050, target_sr=16000)

            wav_noisy, _ = librosa.load(f"{DV}/{x.split('/')[-2].replace('clean', 'noisy')}/{x.split('/')[-1]}", sr=22050)
            if len(wav_noisy)%256!=0:
                wav_noisy = wav_noisy[:-(len(wav_noisy)%256)]
            wav_noisy = librosa.resample(wav_noisy, orig_sr=22050, target_sr=16000)
            
            self.ground_truth_list.append( (wav_clean, wav_noisy) )
            self.mel_clean_list.append(load_numpy(x, 'melspectrogram'))
            self.mel_noisy_list.append(load_numpy(x.replace('clean', 'noisy'), 'melspectrogram'))
            self.spec_noisy_list.append(load_numpy(x.replace('clean', 'noisy'), 'spectrogram'))
        
            w2v_feats_clean = load_numpy(x, 'wav2vec')
            w2v_feats_noisy = load_numpy(x.replace('clean', 'noisy'), 'wav2vec')
            self.w2v_feats_clean_list.append(w2v_feats_clean)
            self.w2v_feats_noisy_list.append(w2v_feats_noisy)
            
            if train_scaler==True:
                self.w2v_clean_scaler.partial_fit(w2v_feats_clean.numpy().T)
                self.w2v_noisy_scaler.partial_fit(w2v_feats_noisy.numpy().T)
    
    def __getitem__(self, idx):
        ground_truth = self.ground_truth_list[idx]
        
        mel_clean = self.mel_clean_list[idx]
        mel_noisy = self.mel_noisy_list[idx]
        spec_noisy = self.spec_noisy_list[idx]
        
        w2v_feats_clean = self.w2v_feats_clean_list[idx]
        w2v_feats_clean = torch.from_numpy(self.w2v_clean_scaler.transform(w2v_feats_clean.numpy().T).T)
        
        w2v_feats_noisy = self.w2v_feats_noisy_list[idx]
        w2v_feats_noisy = torch.from_numpy(self.w2v_noisy_scaler.transform(w2v_feats_noisy.numpy().T).T)
        
        assert mel_clean.size(1)==mel_noisy.size(1)==spec_noisy.size(1)
        assert w2v_feats_clean.size(1)==w2v_feats_noisy.size(1)
            
        return (ground_truth, mel_clean, mel_noisy, spec_noisy, w2v_feats_clean, w2v_feats_noisy)
    
    def __len__(self):
        return len(self.mel_clean_list)
    

def collate_fn(batch):
    num_mels = batch[0][1].size(0)
    num_specs = batch[0][3].size(0)
    num_w2vs = batch[0][4].size(0)
    
    spec_lengths = torch.LongTensor([x[3].size(1) for x in batch])
    max_spec_len = spec_lengths.max().item()
    
    w2v_lengths = torch.LongTensor([x[4].size(1) for x in batch])
    max_w2v_len = w2v_lengths.max().item()
    
    ground_truth_list = []
    mel_clean_padded = torch.zeros(len(batch), num_mels, max_spec_len)
    mel_noisy_padded = torch.zeros(len(batch), num_mels, max_spec_len)
    spec_noisy_padded = torch.zeros(len(batch), num_specs, max_spec_len)
    w2v_feats_clean_padded = torch.zeros(len(batch), num_w2vs, max_w2v_len)
    w2v_feats_noisy_padded = torch.zeros(len(batch), num_w2vs, max_w2v_len)
    for i in range(len(batch)):
        ground_truth_list.append(batch[i][0])
                                 
        mel_clean = batch[i][1]
        mel_clean_padded[i, :, :mel_clean.size(1)] = mel_clean
        
        mel_noisy = batch[i][2]
        mel_noisy_padded[i, :, :mel_noisy.size(1)] = mel_noisy
        
        spec_noisy = batch[i][3]
        spec_noisy_padded[i, :, :spec_noisy.size(1)] = spec_noisy
        
        w2v_feats_clean = batch[i][4]
        w2v_feats_clean_padded[i, :, :w2v_feats_clean.size(1)] = w2v_feats_clean
        
        w2v_feats_noisy = batch[i][5]
        w2v_feats_noisy_padded[i, :, :w2v_feats_noisy.size(1)] = w2v_feats_noisy
        
    return (ground_truth_list, mel_clean_padded, mel_noisy_padded, spec_noisy_padded, spec_lengths,
            w2v_feats_clean_padded, w2v_feats_noisy_padded, w2v_lengths)

