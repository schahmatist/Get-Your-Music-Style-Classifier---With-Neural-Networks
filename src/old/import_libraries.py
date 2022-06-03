from tensorflow import keras
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.layers import BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
#
from keras.models import load_model
from keras import regularizers
from keras.models import Sequential
#
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay,  confusion_matrix
from sklearn.preprocessing import scale, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

import numpy as np
import pandas as pd
import librosa
from librosa.feature import *
from pydub import AudioSegment
from nltk import FreqDist

import sox
#import ffmpeg
import os
import IPython.display as ipd

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
#warnings.filterwarnings(action='ignore', category=FutureWarning)

