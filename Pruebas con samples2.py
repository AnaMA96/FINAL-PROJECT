import pandas as pd
import numpy as pd

import pickle
import joblib 

from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft
import pydub
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import speech_recognition as sr
import csv
from keras import layers
from keras import models
from sklearn import preprocessing 
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from keras.models import load_model

df_validation = pd.read_pickle("./validation-bueno.csv")

instrument = df_validation["type"]=="female singer"
df_total = df_validation[instrument].iloc[[12]]

from keras.models import load_model
new_model = load_model("models/0.9829-accuracy-200000-50epochs-loss0.0823.h5")

X_test=np.vstack(df_total['transformed'])    

X_test.shape


pred = new_model.predict(X_test)
inst = pred[0].tolist()
print(pred)

hola = inst.index(max(inst))
hola

def test(df_total):
    new_model = load_model("models/0.9829-accuracy-200000-50epochs-loss0.0823.h5")
    X_test=np.vstack(df_total['transformed'])  
    pred = new_model.predict(X_test)
    for index in range(0, len(pred)):
        inst = pred[index].tolist()
        hola = inst.index(max(inst))
        print(hola)


df_validation = pd.read_pickle("./validation-bueno.csv")

instrument = df_validation["type"]=="violin"
df_total = df_validation[instrument]

test(df_total)

