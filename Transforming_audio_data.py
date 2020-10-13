import pandas as pd
import numpy as np

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
from keras.models import load_model

def configuredDataFrame(audioPath):
    audio = AudioSegment.from_file(audioPath, format='wav')
    audio_samples = audio.get_array_of_samples()
    four = abs(fft(audio_samples))

    return pd.DataFrame({'transformed': [four]})

def predict(audioPath):
    transformedDF = configuredDataFrame(audioPath)
    print(transformedDF)
    new_model = load_model("models/0.9829-accuracy-200000-50epochs-loss0.0823.h5")
    X_test=np.vstack(transformedDF['transformed'])  
    pred = new_model.predict(X_test)

    inst_type = -1
    for index in range(0, len(pred)):
        inst = pred[index].tolist()
        inst_type = inst.index(max(inst))
    return instName(inst_type)

def instName(instType):
    if instType == 0:
        return "We are listening to a clarinet."
    if instType == 1:
        return "We are listening to a distorted electric guitar."
    if instType == 2:
        return "We are listening to a female singer."
    if instType == 3:
        return "We are listening to a flute."
    if instType == 4:
        return "We are listening to a piano."
    if instType == 5:
        return "We are listening to a tenor saxophone."
    if instType == 6:
        return "We are listening to a trumpet."
    if instType == 7:
        return "We are listening to a violin."
    return "Invalid type of instrument."
