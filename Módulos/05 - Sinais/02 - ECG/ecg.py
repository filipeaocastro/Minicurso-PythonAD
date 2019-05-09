"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Classificação de sinal Eletrocardiográficos

"""

import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

def get_fft(y_values, N, fs):
    f_values = np.linspace(0.0, fs/2.0, N//2)
    fft_values_ = fft(y_values)
    fft_values = np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def plota_fft(y, N, fs, fig = 10):
    fft_x, fft_y = get_fft(y, N, fs)
    plt.figure(fig)
    plt.plot(fft_x, fft_y, color = 'blue', linewidth = 2)
    plt.xlabel('Hz')
    plt.ylabel('Amplitude')
    plt.show()
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Importando o dataset
dataset = pd.read_csv('samples.csv', header = [0, 1])
t = dataset.iloc[:, 0].values
sinal = dataset.iloc[:, 1].values


T = t[1] - t[0]
fs = 1 / T
N = len(t)

lowcut = 0.5
highcut = 10

sinal_filt = butter_bandpass_filter(sinal, lowcut, highcut, fs, order = 2)

plota_fft(sinal, N, fs, fig = 1)
plota_fft(sinal_filt, N, fs, fig = 2)

plt.figure(3)
plt.plot(t, sinal, color = 'grey', linewidth = 3)
plt.plot(t, sinal_filt, color = 'red', linewidth = 3)

plt.figure(4)
plt.subplot(211)
plt.plot(t, sinal, color = 'grey', linewidth = 2, label = 'Raw ECG')
plt.legend(loc='upper right')
plt.subplot(212)
plt.plot(t, sinal_filt, color = 'red', linewidth = 2, label = 'ECG Filtrado')
plt.legend(loc='upper right')





