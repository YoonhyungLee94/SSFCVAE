import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
from utils import *
from transformers import AutoModel, AutoFeatureExtractor
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy import signal

DATA_PATH = "./Dataset/VoiceBank-DEMAND"
    
if __name__ == "__main__":
    start = datetime.now()
    print(f"Start!!! ({start})")
    model = AutoModel.from_pretrained("facebook/wav2vec2-large-lv60")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
    wav_dir_list = [x for x in os.listdir(DATA_PATH) if (("wav" in x) and ("clean" in x))]
    for wav_dir in wav_dir_list:
        wav_nosiy_dir = wav_dir.replace('clean', 'noisy')
        if not os.path.exists(f"{DATA_PATH}/preprocessed/{wav_dir.replace('wav', 'melspectrogram')}"):
            os.mkdir(f"{DATA_PATH}/preprocessed/{wav_dir.replace('wav', 'melspectrogram')}")
            os.mkdir(f"{DATA_PATH}/preprocessed/{wav_dir.replace('wav', 'wav2vec')}")
            os.mkdir(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'melspectrogram')}")
            os.mkdir(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'spectrogram')}")
            os.mkdir(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'wav2vec')}")

        for i, wav_path in enumerate(tqdm(os.listdir(f"./Dataset/VoiceBank-DEMAND/{wav_dir}"))):
            ############# Clean wav #############
            y_clean_22k, sr = librosa.load(f"./Dataset/VoiceBank-DEMAND/{wav_dir}/{wav_path}", sr=22050)
            if len(y_clean_22k)%256 != 0:
                y_clean_22k = y_clean_22k[:-(len(y_clean_22k)%256)]
            assert len(y_clean_22k)%256 == 0
            
            melspec_clean = mel_spectrogram_torch(torch.FloatTensor(y_clean_22k)[None, :],1024,80,22050,256,1024,0,8000)[0].numpy()
            np.save(f"{DATA_PATH}/preprocessed/{wav_dir.replace('wav', 'melspectrogram')}/{wav_path[:-4]}.npy", melspec_clean)
            print(f"Clean melspec shape: {melspec_clean.shape}") if i==0 else None
            
            y_clean_16k = librosa.resample(y_clean_22k, orig_sr=sr, target_sr=16000)
            y_clean_16k = processor(y_clean_16k, sampling_rate=16000, return_tensors="pt")['input_values'].to(device)
            with torch.no_grad():
                wav2vec_clean = model(y_clean_16k, output_hidden_states=True).hidden_states[15].transpose(1,2) # [1, D, T]
            np.save(f"{DATA_PATH}/preprocessed/{wav_dir.replace('wav', 'wav2vec')}/{wav_path[:-4]}.npy",
                    wav2vec_clean[0].detach().cpu().numpy()) # [D, T]
            print(f"Clean wav2vec shape: {wav2vec_clean[0].detach().cpu().numpy().shape}") if i==0 else None
            
            ############# Noisy wav #############
            y_noisy_22k, sr = librosa.load(f"./Dataset/VoiceBank-DEMAND/{wav_nosiy_dir}/{wav_path}", sr=22050)
            if len(y_noisy_22k)%256 != 0:
                y_noisy_22k = y_noisy_22k[:-(len(y_noisy_22k)%256)]
            assert len(y_noisy_22k)%256 == 0
                
            spec_noisy = spectrogram_torch(torch.FloatTensor(y_noisy_22k)[None, :],1024,22050,256,1024)
            np.save(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'spectrogram')}/{wav_path[:-4]}.npy",
                    spec_noisy[0].numpy())
            print(f"Noisy spec shape: {spec_noisy[0].numpy().shape}") if i==0 else None
            
            melspec_noisy = spec_to_mel_torch(spec_noisy,1024,80,22050,0,8000)[0].numpy()
            np.save(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'melspectrogram')}/{wav_path[:-4]}.npy", melspec_noisy)
            print(f"Noisy melspec shape: {melspec_noisy.shape}") if i==0 else None
            
            y_noisy_16k = librosa.resample(y_noisy_22k, orig_sr=sr, target_sr=16000)
            y_noisy_16k = processor(y_noisy_16k, sampling_rate=16000, return_tensors="pt")['input_values'].to(device)
            with torch.no_grad():
                wav2vec_noisy = model(y_noisy_16k, output_hidden_states=True).hidden_states[15].transpose(1,2) # [1, D, T]
            np.save(f"{DATA_PATH}/preprocessed/{wav_nosiy_dir.replace('wav', 'wav2vec')}/{wav_path[:-4]}.npy",
                    wav2vec_noisy[0].detach().cpu().numpy()) # [D, T]
            print(f"Noisy wav2vec shape: {wav2vec_noisy[0].detach().cpu().numpy().shape}") if i==0 else None

    print()
    print(f"Finish!!! ({datetime.now()})")
    print(f"It took '{datetime.now()-start}'")
