#%run ../src/import_libraries.py
import librosa
from librosa.feature import rms, chroma_stft, spectral_bandwidth, zero_crossing_rate, spectral_contrast, mfcc
import sox
from pydub import AudioSegment

import pandas as pd
import os
import numpy as np
from nltk import FreqDist

from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, LabelEncoder, StandardScaler


import matplotlib.pyplot as plt
plt.style.use('ggplot')

from wordcloud import WordCloud
import seaborn as sns
from PIL import Image
import IPython.display as ipd
from IPython.display import Markdown as md

import joblib
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

##############################################

base_dir = '../data_music/'
processed_path = '../data_music/classify/'
dirpath = base_dir + 'recomm_db/'
base_dir = '../data_music/'
skip=35
recomm_df = pd.read_csv('../features/recomm_processed_feat_new_.csv')

################################################

with open('../models/transformers_6sec_14genres_new.joblib', 'rb') as f:
    encoder,scaler = joblib.load(f)

genres=encoder.classes_

def extract_wave_features (x, sr):
    feat_dic={}
    x_harm, x_perc = librosa.effects.hpss(x)

# Power Based total Means
    for num, series in enumerate([ x_harm, x_perc]):
        label={0:'harm',1:'perc'}
        for func in [rms, chroma_stft, spectral_bandwidth, zero_crossing_rate]:
            feat_name=str(func).split()[1]
            if func == rms:
                s=librosa.stft(series)
                S, phase = librosa.magphase(s)
                feature = np.mean(func(S=S))
            else:
                feature = np.mean(func(y=series))
            feat_dic[f'{feat_name}_{label[num]}_mean']=feature

# Power Based Multiple Means
    for func in [ mfcc]:
        feat_lst = np.mean(func(y=x, sr=sr, n_mfcc=13), axis=1)

        feat_name=str(func).split()[1]

        for num, feature in enumerate(feat_lst):
            feat_dic[f'{feat_name}_{num}']=feature
# Energy Based:
    for func in [spectral_contrast ]:
        s=librosa.stft(x)
        feat_lst = np.mean(func(S=np.abs(s), sr=sr), axis=1)
        feat_name=str(func).split()[1]
        for num, feature in enumerate(feat_lst):
            feat_dic[f'{feat_name}_{num}']=feature
# Tempo    
    tempo, beat_frames = librosa.beat.beat_track(y=x_harm, sr=sr)
    feat_dic['Tempo']=tempo
    
    return feat_dic



def extract_file_features(mp3_path, mini_chunk_length, mini_chunk_n):
    feat_data_lst=[]
    song_file=mp3_path.split('.mp3')[0] +'.wav'

    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(song_file, format="wav")
    song_duration=sox.file_info.duration(song_file)-skip

    if mini_chunk_length > song_duration: 
        mini_chunk_length = int(song_duration)

    sample_duration=mini_chunk_length*mini_chunk_n

    if sample_duration > song_duration:
        sample_duration = song_duration
        mini_chunk_n=int(song_duration//mini_chunk_length)

    y, sr = librosa.load(song_file, offset=skip, duration=sample_duration)
    os.remove(song_file)
    mini_chunk_samples = mini_chunk_length * sr

    for mini_chunk in range(mini_chunk_n):
        start = mini_chunk * mini_chunk_samples
        end = start + mini_chunk_samples
        ch = y[start:end]
        feat_dic=extract_wave_features(ch, sr)
        feat_data_lst.append(feat_dic)
    return feat_data_lst


def extract_featues(processed_path, mini_chunk_length, mini_chunk_n):
    feat_all_lst=[]
#    if chosen_file == 'None':
    song_files=[x for x in os.listdir(processed_path) if x.endswith('.mp3')]
#    else:
#        song_files=[chosen_file]
    
    for mp3 in song_files:
        print(mp3)
  #      mp3_path=mp3
        mp3_path=processed_path + mp3
        ipd.display(ipd.Audio(mp3_path))
        feat_lst = extract_file_features(mp3_path, mini_chunk_length, mini_chunk_n) 
        feat_all_lst.extend(feat_lst)

    classify_df=pd.DataFrame.from_dict(feat_all_lst)
    classify_df.columns=feat_all_lst[0].keys()
    return mini_chunk_length, classify_df

def visualize (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label):
    fig, ax = plt.subplots(figsize=(10,24), nrows=4)
    fontsize=30
    
    sns.scatterplot(x=(np.arange(mini_chunk_n)*mini_chunk_length)+skip, y=NN_pred-0.1, 
                    alpha=0.5, s=340, color='blue', label='NN', ax=ax[2])
    sns.scatterplot(x=(np.arange(mini_chunk_n)*mini_chunk_length)+skip, y=XGB_pred-0.1,
                    alpha=0.5,  s=250, color='green', label='XGB', ax=ax[2])

    sns.barplot(x=genres, y=sum_prob, color='green', ax=ax[3]);
    
    wc = WordCloud(background_color="black",  collocations=False,  max_font_size=2000, min_font_size=1, 
           width=1600, height=800, relative_scaling=1 ).generate_from_frequencies(genre_dic)
    ax[0].set_title('Influences:', fontsize=fontsize, loc='left')
    ax[0].imshow(wc, interpolation='bilinear')
    ax[0].axis("off");
    
    ax[1].set_title('Primary Genre:',fontsize=fontsize, loc='left')
    wc = WordCloud(background_color="white",  collocations=False,  max_font_size=2000, min_font_size=1, 
       width=1600, height=800, relative_scaling=1 ).generate_from_text(prediction_label)
    ax[1].imshow(wc, interpolation='bilinear')
    ax[1].axis("off");
    
    ax[3].set_title('Genre Probabilities:', fontsize=fontsize, loc='left')
    ax[3].set_xticklabels(ax[3].get_xticklabels(),rotation = 75, size=20)
    ax[3].set_yticklabels([round(tick,2) for tick in ax[3].get_yticks()], size=15)

    ax[2].set_title('Genre Influences Analysis:',fontsize=fontsize, loc='left')
    ax[2].set_yticklabels(genres, size=15)
    ax[2].set_yticks(range(len(genres)))
    ax[2].set_xlabel('Seconds');
    ax[2].legend()
    ax[2].set_ylim(-1, len(genres)+1);

    fig.tight_layout(pad=0)
    fig.savefig('test.png')
    
def print_details (XGB_prob, NN_prob, XGB_pred, NN_pred, sum_prob, prediction, prediction_label):
    print('======================================================')
    print('XGB: ', XGB_pred,': ', np.argmax(sum(XGB_prob)))
    print('NN: ', NN_pred, ': ', np.argmax(sum(NN_prob) ))
    print('--------------------------------------------')
    print('XGB prob: ', sum(XGB_prob),': ', np.argmax(sum(XGB_prob)))
    print('NN prob: ',sum(NN_prob),': ', np.argmax(sum(NN_prob) )) 
    print('--------------------------------------------')
    print(sum_prob)
    print(prediction_label, ': ', prediction)
    print('======================================================')



def visualize0 (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label):
    fig, ax = plt.subplots(figsize=(8,5))
    fontsize=30

    ax.set_title('Primary Genre:',fontsize=fontsize, loc='left')
    wc = WordCloud(background_color="white",  collocations=False,  max_font_size=2000, min_font_size=1,
       width=1600, height=800, relative_scaling=1 ).generate_from_text(prediction_label)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off");
    fig.tight_layout(pad=0)


def visualize1 (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label):
    fig, ax = plt.subplots(figsize=(8,5) )
    fontsize=30

    wc = WordCloud(background_color="black",  collocations=False,  max_font_size=2000, min_font_size=1,
           width=1600, height=800, relative_scaling=1 ).generate_from_frequencies(genre_dic)
    ax.set_title('Influences:', fontsize=fontsize, loc='left')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off");
    fig.tight_layout(pad=0)


def visualize2 (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label):
    fig, ax = plt.subplots(figsize=(8,5) )
    fontsize=30

    sns.scatterplot(x=(np.arange(mini_chunk_n)*mini_chunk_length)+skip, y=NN_pred-0.1,
                    alpha=0.5, s=340, color='blue', label='NN', ax=ax)
    sns.scatterplot(x=(np.arange(mini_chunk_n)*mini_chunk_length)+skip, y=XGB_pred-0.1,
                    alpha=0.5,  s=250, color='green', label='XGB', ax=ax)

    ax.set_title('Genre Influences Analysis:',fontsize=fontsize, loc='left')
    ax.set_yticklabels(genres, size=15)
    ax.set_yticks(range(len(genres)))
    ax.set_xlabel('Seconds');
    ax.legend()
    ax.set_ylim(-1, len(genres)+1);
    fig.tight_layout(pad=0)


def visualize3 (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label):
    fig, ax = plt.subplots(figsize=(8,6))
    fontsize=30

    sns.barplot(x=genres, y=sum_prob, color='green', ax=ax);

    ax.set_title('Genre Probabilities:', fontsize=fontsize, loc='left')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 75, size=20)
    ax.set_yticklabels([round(tick,2) for tick in ax.get_yticks()], size=15)

    fig.tight_layout(pad=0)


def classify (model, model2, processed_path, mini_chunk_length, mini_chunk_n ):
    np.set_printoptions(suppress=True)
    mini_chunk_length, classify_df=extract_featues(processed_path, mini_chunk_length, mini_chunk_n)
    mini_chunk_n=np.shape(classify_df)[0]

    X=classify_df.drop('Tempo', axis=1)
    X_scaled = scaler.transform(np.array(X, dtype = float))

    NN_prob=model.predict(X_scaled)
    XGB_prob=model2.predict_proba(X)
    NN_pred=np.argmax(NN_prob, axis=1)
    XGB_pred=model2.predict(X)

    joined=np.concatenate((NN_pred,XGB_pred),axis=0)
    sum_prob=sum(XGB_prob)+sum(NN_prob)
    sum_prob=sum_prob/np.sum(sum_prob)

    prediction=np.argmax(sum(XGB_prob)+sum(NN_prob))
    prediction_label=encoder.inverse_transform([prediction])[0]

    genre_vote_freqdist=FreqDist([encoder.inverse_transform([x])[0] for x in joined])

    genre_dic={}
    for key,value in  genre_vote_freqdist.most_common(3):
        genre_dic[key]=value

 #   visualize1 (NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label)
    # print_details (XGB_prob, NN_prob, XGB_pred, NN_pred, sum_prob, prediction, prediction_label)

    return NN_pred, XGB_pred, mini_chunk_n, mini_chunk_length, sum_prob, genre_dic, prediction_label, classify_df



def display_recommendations(prediction, classify_df, recomm_df, num):
    features=[  'Tempo', 'spectral_contrast_0', 'rms_harm_mean' ]
  
    
    if sum(recomm_df['Genre']==prediction) == 0:
        print('Cannot Find a genre!')
        return None
    database_genre=recomm_df[recomm_df['Genre']==prediction].copy()
    #database_genre=recomm_df.copy()
    database_feat=database_genre[features].reset_index(drop=True)
    
    given_feat=list(classify_df[features].mean())
    database_feat.loc[len(database_feat)] = given_feat
    
    database_feat_scaled=(database_feat-np.mean(database_feat))/np.std(database_feat)
    given_feat_scaled=database_feat_scaled.iloc[len(database_feat)-1]
    
    score_list=[]
    for row in np.array(database_feat_scaled):
        score=mean_squared_error(row, given_feat_scaled, squared=False)
        if score != 0:
            score_list.append(score)
        
    database_genre['Score']=score_list
    recommendations=database_genre.sort_values(by='Score').head(6)
    recommendations=recommendations.sample(num).sort_values(by='Score')

    for file in recommendations['Filename']:
        filepath=dirpath+file
        print(file)
        ipd.display(ipd.Audio(filepath))
    
    return database_genre.sort_values(by='Score')[['Score','Filename']].head(20)

