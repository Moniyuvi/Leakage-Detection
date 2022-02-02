from os.path import join
from posixpath import curdir
import glob
from datetime import datetime
import numpy as np

import librosa, librosa.display
from librosa.core import audio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.lib.function_base import average
from openpyxl import Workbook


def load_audio(file_name):
    y, sr = librosa.load(file_name)
    return y, sr

def display_audio(audio_signal, axis_, sr=44100):
    librosa.display.waveshow(audio_signal, sr=sr, ax=axis_)
    axis_.set_xlabel('Time [s]')
    axis_.set_ylabel('Amplitude')

def display_spectrogram(audio_signal, axis_, sr=44100, type='linear'):
    X = librosa.stft(audio_signal, n_fft=1024, hop_length=512)
    X = np.abs(X)**2
    Xdb = librosa.power_to_db(abs(X))
    if type == 'linear':
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='linear', ax=axis_)
        #axis_.set_ylim([0, 4000])
        #axis_.colorbar(format="%+2.f")
        axis_.set_xlabel('Time [s]')
        axis_.set_ylabel('Frequency')
    elif type == 'log':
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log', ax=axis_)
        #axis_.colorbar(format="%+2.f")
        axis_.set_xlabel('Time [s]')
        axis_.set_ylabel('Frequency')

def amplitude_envolope(audio_signal, frame_size, hop_length):
    amplitude_envolople = []
    for i in range(0, len(audio_signal), hop_length):
        curr_frame_amp_envolople = max(audio_signal[i: i+frame_size])
        amplitude_envolople.append(curr_frame_amp_envolople)
    return np.array(amplitude_envolople)

def frequency_spectrum(audio_signal, sr, f_ratio = 0.5):
    y_fft = np.fft.fft(audio_signal)
    magnitude_spectrum = np.abs(y_fft)
    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    num_freq_bins = int(len(audio_signal) * f_ratio)
    max_ = 0
    
    # max frequency component
    index_of_max_amplitude = np.argmax(magnitude_spectrum[1:num_freq_bins])
    freq = frequency[index_of_max_amplitude]
            
    return (frequency[:num_freq_bins], magnitude_spectrum[:num_freq_bins], freq)

def signal_plotter(x_axis, y_axis, title):
    plt.figure(figsize=(14, 10))
    plt.title(title)
    plt.plot(x_axis, y_axis)
    

def main():
    #load files
    file_name = 'Gilfach Road2' + datetime.now().strftime("_%H%M%d%m%Y")
    audio_directory = 'Gilfach Road2'
    audio_files = glob.glob('./'+audio_directory+'/*.wav')
    file_names = [a.split('/')[-1].split('.')[-2] for a in audio_files]
    
    # excel
    wb = Workbook()
    ws1 = wb.active
    ws1.title = audio_directory + 'Gilfach Road2'
    ws1.append(['File Name', 'Date', 'Intensity', 'Frequency', 'Average Instensity', 'Avg Intensity dB'])    
    
    with PdfPages(join('./plots', file_name+'.pdf')) as pdf:
        for index, audio in enumerate(audio_files):
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))
            signal, sr = load_audio(audio)
            
            #Waveform
            axs[0].set(title=file_names[index])
            display_audio(signal, axs[0])
            display_spectrogram(signal, axs[1])
            display_spectrogram(signal, axs[2], type='log')
            
            # frequency
            frequency, magnitude_spectrum, max_freq = frequency_spectrum(signal, sr)
            
            # Max Intensity
            intensity = max(amplitude_envolope(signal, frame_size=1024, hop_length=512))

            # Average of Intensities
            intensity_set = amplitude_envolope(signal, frame_size=1024, hop_length=512)
            average_intensity = np.sum(intensity_set)/len(intensity_set)

            # Decibel Conversion
            amplitude_db = 10 * np.log10(average_intensity/1e-12)
            
            #excel
            ws1.append([file_names[index], datetime.now().strftime('%D'), intensity, max_freq, average_intensity, amplitude_db])
            wb.save("./excel/"+file_name+".xlsx")
            
            pdf.savefig()
        print('Process Complete')   


if __name__ == '__main__':
    main()
