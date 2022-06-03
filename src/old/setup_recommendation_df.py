import os
import numpy as np
import pandas as pd
import librosa
from librosa.feature import *
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


music_dir='../music_data/recomm_db/'
catalog_dir='../features/'

recomm_df=pd.read_csv(catalog_dir+'recommendations_new.csv' , sep=';', usecols=list(range(8)))
recomm_df=recomm_df.fillna('Unknown')

recomm_df.head()

dirpath=music_dir

tempo_lst=[]
rmse_lst=[]
zcr_lst=[]
spectral_cent_lst=[]
spectral_band_lst=[]
spectral_roll_lst=[]
chroma_lst=[]

for num, file in enumerate(tqdm(recomm_df['Filename'])):
    file_path=dirpath + file
    try:
        y, sr = librosa.load(file_path, offset=30, duration=60, sr=None)
    except:
        print('Cannot open '+file_path)
        recomm_df=recomm_df[recomm_df.index != num].copy()
        continue

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    rmse = np.mean(rms(y=y))
    zcr = np.mean(zero_crossing_rate(y=y))
    chroma = np.mean(chroma_stft(y=y))
    sc = np.mean(spectral_centroid(y=y))
    sb = np.mean(spectral_bandwidth(y=y))
    sr = np.mean(spectral_rolloff(y=y))

    tempo_lst.append(tempo)
    rmse_lst.append(rmse)
    zcr_lst.append(zcr)
    spectral_cent_lst.append(sc)
    spectral_band_lst.append(sb)
    spectral_roll_lst.append(sr)
    chroma_lst.append(chroma)
    

recomm_df['zero_crossing_rate_mean'] = zcr_lst
recomm_df['rms_mean'] = rmse_lst
recomm_df['chroma_stft_mean'] = chroma_lst
recomm_df['spectral_centroid_mean'] = spectral_cent_lst
recomm_df['spectral_bandwidth_mean'] = spectral_band_lst
recomm_df['spectral_rolloff_mean'] = spectral_roll_lst
recomm_df['Tempo'] = tempo_lst

recomm_df.to_csv(catalog_dir+'recomm_processed_new_copy.csv',index=False)
