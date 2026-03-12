import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import usv_library as ul
from PIL import Image
import numpy as np


## EDITING ##


def pad_to_square(img):
    h, w, _ = img.shape
    if h == w:
        return img
    if h > w:
        pad_size = (h - w) // 2
        return np.pad(img, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Width is greater than height, cannot pad to square.")


def get_spectrogram(wavfile, classification, time_window, name, freq_range=(15000, 35000), save=True, buffer=0.01, gain=1):
    """
    Takes in a WAV file path with time stamps and returns a spectrogram image of a USV.
    """
    y, fs = librosa.load(wavfile, offset= time_window[0]-buffer, duration= time_window[1]-time_window[0]+buffer, sr= None)

    # get appropriate figure size based on time window and frequency range
    spec_duration = (time_window[1] - time_window[0] + buffer * 2) / 0.05
    freq_range_size = (freq_range[1] - freq_range[0]) / 5000
        
    plt.figure(figsize= (spec_duration, freq_range_size))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=40) + 20 
    spectrogram = librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='hz')
    plt.ylim(freq_range)
    plt.axis('off')
    plt.savefig(f"spectrograms/{classification}/{name}.jpeg", bbox_inches='tight', pad_inches=0)


def pad_spectrograms(directory):
    for img_file in os.listdir(directory):
        if img_file.endswith(".jpeg"):
            img = Image.open(f"{directory}/{img_file}")
            img_array = np.array(img)
            padded_img_array = pad_to_square(img_array)
            padded_img = Image.fromarray(padded_img_array)
            padded_img.save(f"{directory}/padded/pad_{img_file}")


def get_from_rated(wavfile, csv, folder):
    # need to load csv to get list of usvs with start/stop and 25 rating
    usv_data = pd.read_csv(csv)[['start', 'stop', 'label', '25_kHz_call']]
    split_name = folder.split("_")
    session = split_name[0]+split_name[4]+split_name[5]

    # break up dataframe into 25s and non25s
    usv_data_25 = ...
    usv_data_noise = usv_data[usv_data['25_kHz_call'] == 0]

    i = 0
    while i <= 1:
        usv_data_filt = usv_data[usv_data['25_kHz_call'] == i]
        classification = '25' if i == 1 else 'noise'
        for index, row in usv_data_filt.iterrows():
            time_window = (row['start'], row['stop'])
            name = f"{session}_{row['label']}"
            get_spectrogram(wavfile, classification, time_window, name)
        i += 1
    
    print("Spectrograms generated from rated data.")
            


