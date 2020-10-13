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

data2 = pd.read_csv("./Medley-solos-DB/Medley-solos-DB_metadata.csv")
df2 = pd.DataFrame(data2)
df2.shape

validation = df2["subset"]=="validation"
df_validation= df2[validation]
df_validation.shape

#df_validation = df_validation.apply(lambda df: f"./Medley-solos-DB_validation-{df['instrument_id']}{df['uuid4']}.wav")

df_validation['uuid4'] = df_validation.apply(lambda row: f"./Medley-solos-DB/Medley-solos-DB_validation-{row.instrument_id}_{row.uuid4}.wav", axis = 1)

df_validation.head()

instrument_lst = set(df_validation.instrument.values)  
instrument_lst

instrument_lst = list(instrument_lst)

instruments_validation_paths = {}
for inst in instrument_lst:
    instruments_validation_paths.update({inst:list(df_validation['uuid4'][df_validation["instrument"]==inst])})

# Creamos una copia de instruments_test_paths por si modificamos algo por error,
# para no tener que volver a ejecutar la celda anterior.
instruments_validation_paths_copy = instruments_validation_paths

def removeEmptyPaths(inst_paths_dict):
    '''
    Esta función limpia los instrumentos cuya lista
    de paths esté vacía y te devuelve una copia del diccionario limpio.
    '''
    inst_paths_dict_cleaned = inst_paths_dict
    for inst, paths in inst_paths_dict.items():
        if not paths:
            inst_paths_dict_cleaned.pop(inst)
        else:
            print(inst)
    return inst_paths_dict_cleaned

instruments_validation_paths_copy = removeEmptyPaths(instruments_validation_paths_copy)

def normalizeInstPathsLen(inst_paths_dict):
    '''
    Esta función va a homogeneizar la longitud 
    de las listas de paths que tiene el diccionario
    que recibe, va a crear un nuevo diccionario y 
    lo va a devolver.
    '''
    insts_paths_normalized_len = {}
    min_len = len(list(inst_paths_dict.items())[0][1])
    for inst, paths in inst_paths_dict.items():
        if min_len > len(paths):
            min_len = len(paths)
    for inst, paths in inst_paths_dict.items():
        insts_paths_normalized_len[inst] = paths[:min_len]
        
    return insts_paths_normalized_len

insts_validation_paths_normalized = normalizeInstPathsLen(instruments_validation_paths_copy)

insts_validation_paths_normalized

def samplesList(path_lst):
    '''
    Esta función crea una lista de objetos de AudioSegment
    por cada fichero de la lista que recibe (path_lst),
    después por cada objeto AudioSegment, obtiene la lista
    de samples de cada audio, creando una nueva lista y devolviéndola.
    '''
    audio_lst = []
    audio_samples_lst = []
    for path in path_lst:
        audio = AudioSegment.from_file(path, format='wav')
        audio_lst.append(audio)
    for a in audio_lst:
        audio_samples_lst.append(a.get_array_of_samples())
    
    return audio_samples_lst

def fftList(samples_lst):
    '''
    Esta función recibe una lista de samples de audios,
    hace la transformada de Fourier y devuelve una lista, 
    con toda la lista transformada.
    '''
    fft_lst = []
    for sample in samples_lst:
        four = abs(fft(sample))
        fft_lst.append(four)
    
    return fft_lst

# Cogemos el diccionario y por cada instrumento generamos su lista de samples:

def instSamples(insts_paths):
    '''
    Esta función genera un dict de instrumentos
    y su lista samples a partir del dict de paths 
    que recibe y lo devuelve.
    '''
    inst_samples = {}
    for inst, paths in insts_paths.items():
        inst_samples[inst] = samplesList(paths)
        
    return inst_samples

insts_validation_samples = instSamples(insts_validation_paths_normalized)

def infoSamples(inst_samples):
    '''
    Esta función halla la longitud máxima, mínima y
    la media de los samples de todos los instrumentos.
    '''
    min_len = len(list(inst_samples.items())[0][1][0])
    max_len = 0
    joined_samples_lst = []
    for inst, samples_lst in inst_samples.items():
        joined_samples_lst = joined_samples_lst + samples_lst
        
    means_lst = []
    for samples in joined_samples_lst:
        samples_len = len(samples)
        means_lst.append(samples_len)
        if min_len > samples_len:
            min_len = samples_len
        if max_len < samples_len:
            max_len = samples_len
    
    print(f'Mean: {sum(means_lst) / len(means_lst)}')
    print(f'Max len: {max_len}')
    print(f'Min len: {min_len}')

infoSamples(insts_validation_samples)

def instFft(insts_samples):
    '''
    Esta función genera un dict de instrumentos
    y su lista samples a partir del dict de paths 
    que recibe y lo devuelve.
    '''
    inst_ffts = {}
    for inst, samples in insts_samples.items():
        inst_ffts[inst] = fftList(samples)
        
    return inst_ffts

insts_ffts = instFft(insts_validation_samples)

def dataFramer(insts_ffts):
    '''
    Esta función toma el diccionario que se le pase como argumento
    y, por cada entrada del diccionario, crea un dataframe y los concatena 
    todos.
    '''
    lst_df = []
    for inst, fft_lst in insts_ffts.items():
        inst_val = {"transformed": fft_lst, "type": inst}
        lst_df.append(pd.DataFrame(inst_val))
    return pd.concat(lst_df).reset_index(drop=True)

df_total_validation = dataFramer(insts_ffts)
df_total_validation.head()
df_total_validation.to_pickle(r'./validation-bueno.csv')