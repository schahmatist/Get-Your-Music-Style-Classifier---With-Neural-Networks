import os
import numpy as np
import pandas as pd
import librosa
from librosa.feature import *
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


music_dir='../data_music/recomm_db/'
catalog_dir='../features/'

recomm_df=pd.read_csv(catalog_dir+'recommendations_new.csv' , sep=';', usecols=list(range(8)))
#recomm_df=recomm_df[0:1].fillna('Unknown')
recomm_df=recomm_df.fillna('Unknown')

recomm_df.head()

dirpath=music_dir

tempo_lst=[]
rms_harm_lst=[]
rms_perc_lst=[]
mfcc2_lst=[]
mfcc3_lst=[]
mfcc4_lst=[]
spectr0_lst=[]
spectr5_lst=[]

for num, file in enumerate(tqdm(recomm_df['Filename'])):
    file_path=dirpath + file
    try:
        y, sr = librosa.load(file_path, offset=30, duration=45, sr=None)
    except:
        print('Cannot open '+file_path)
        recomm_df=recomm_df[recomm_df.index != num].copy()
        continue

    y_harm, y_perc = librosa.effects.hpss(y)
    s=librosa.stft(y)
    s_harm=librosa.stft(y_harm)
    s_perc=librosa.stft(y_perc)

    # tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y_harm, sr=sr)

    # Spectral Contrast 0 and 5
    sc_lst = np.mean(spectral_contrast(S=np.abs(s), sr=sr), axis=1)
    sc0=sc_lst[0]
    sc5=sc_lst[5]

    # mfcc 2,3,4
    mfcc_lst = np.mean(mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    mfcc2=mfcc_lst[2]
    mfcc3=mfcc_lst[3]
    mfcc4=mfcc_lst[4]

    # perc and harm rmse
    S_harm, phase = librosa.magphase(s_harm)
    rms_harm = np.mean(rms(S=S_harm))

    S_perc, phase = librosa.magphase(s_perc)
    rms_perc = np.mean(rms(S=S_perc))


    spectr0_lst.append(sc0)
    spectr5_lst.append(sc5)
    mfcc2_lst.append(mfcc2)
    mfcc3_lst.append(mfcc3)
    mfcc4_lst.append(mfcc4)
    rms_harm_lst.append(rms_harm)
    rms_perc_lst.append(rms_perc)
    tempo_lst.append(tempo)

recomm_df['rms_perc_mean'] = rms_perc_lst
recomm_df['rms_harm_mean'] = rms_harm_lst
recomm_df['spectral_contrast_0'] = spectr0_lst
recomm_df['spectral_contrast_5'] = spectr5_lst
recomm_df['mfcc_2'] = mfcc2
recomm_df['mfcc_3'] = mfcc3
recomm_df['mfcc_4'] = mfcc4
recomm_df['Tempo'] = tempo_lst

recomm_df.to_csv(catalog_dir+'recomm_processed_feat_new.csv',index=False)
