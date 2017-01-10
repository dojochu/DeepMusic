import tensorflow as tf
from keras.layers import Input, Dense, Convolution2D, AveragePooling2D
from keras.models import Model
#from DeepMusicStyle.reverse_melspectrogram import read_audio, compute_melgram
from musicCNN.audio_processor import compute_melgram
import pandas as pd
import numpy as np

#Path to song files, METADATA and labels
data_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/mp3/'
label_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/annotations_final.csv'
info_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/clip_info_final.csv'

# Reading and processing METADATA and labels
clip_info = pd.read_csv(filepath_or_buffer=info_path, delimiter='\t')
label_info = pd.read_csv(filepath_or_buffer=label_path, delimiter='\t')
clip_info['batch_id'] = pd.Series([str(b)[2] for b in clip_info.mp3_path.str.split('/')])
label_info['batch_id'] = pd.Series([str(b)[2] for b in label_info.mp3_path.str.split('/')])

# CONSTANTS
SR = 12000 #Sample Rate
N_FFT = 512 #Parameter for Short Fourier Transforms
N_MELS = 96 # Number of Mel-Bins
HOP_LEN = 256 #Hop Length
DURA = 29.12  #Duration (to make it 1366 frame)
FRAME_SIZE = int(DURA * SR / HOP_LEN + 1) #song data length
NUM_LABELS = 188 #Number of categories
TEST_SIZE = 2000 #Test data_set size
iterations = 1000 #Number of iterations (epochs)

'''
Input Tensor - Using the Mel-Spectrogram of the .wav audio file to create a 96 x 1366 feature matrix.
input shape: (samples, channels, mel-bins, framesize) = (1, 1, 96, 1366).
'''
model_input = Input(shape=(1 ,1, N_MELS,FRAME_SIZE),batch_shape=(None,1, N_MELS,FRAME_SIZE), name='initial_input',dtype='float32')

'''
Model Architecture


1. 2D Convolutional Layer

    64 feature maps of size 3x3
    initialization: 'glorot_normal' Gaussian initialization scaled by fan_in + fan_out (Glorot 2010)
    activation: 'tanh'
    border_mode: 'same' Pad the input with zeros so that the input shape equals output shape
    input shape: (samples, channels, mel-bins, framesize) = (1, 1, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 64, 96, 1366)

2. 2D Convolutional Layer

    32 feature maps of size 3x3
    input shape: (samples, number of filters, rows, cols) = (1, 64, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 32, 96, 1366)

3. 2D Average Pooling Layer

    Pool size: 2x2
    strides: 2x2
    border_mode: 'same'
    input shape: (samples, number of filters, rows, cols) = (1, 32, 96, 1366)
    output shape: (samples, number of filters, rows, cols) = (1, 64, 48, 683)

4. Fully-Connected Layer

    input shape: output shape: (samples, number of filters, rows, cols) = (1, 64, 48, 683)
    output shape: (number of genres, ) = (188,)

'''
model = Convolution2D(nb_filter=64,nb_row=3,nb_col=3,
                      init='glorot_normal',activation='tanh',
                      border_mode='same')(model_input)

model = Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                       init='glorot_normal',activation='relu',
                       border_mode='same')(model)

model =  AveragePooling2D(pool_size=(2,2), border_mode='valid')(model)

model_output = Dense(output_dim = NUM_LABELS, activation = 'relu')(model)

model = Model(input=model_input, output=model_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

melgrams = np.zeros(shape=(FRAME_SIZE,))
labels = np.zeros(shape =(NUM_LABELS))
for bid in pd.unique(clip_info.batch_id)[:2]:
    print('Training on batch_id: ' + bid)
    melgrams = np.zeros(shape=(1, 1, N_MELS, FRAME_SIZE))
    labels = np.zeros(shape =(NUM_LABELS,))
    for song in clip_info[clip_info['batch_id'] == bid]['mp3_path']:
        #melgrams = np.concatenate(melgrams, compute_melgram(read_audio(data_path + song)).flatten())
        try:
            melgrams = np.concatenate((melgrams, compute_melgram(data_path + song)))
            add_label = label_info[label_info['mp3_path'] == song].filter(
                        items=label_info.columns.tolist()[1:-2]).transpose().iloc[:,0]
            labels = np.vstack((labels, add_label))
        except FileNotFoundError:
            continue
    model.train_on_batch(melgrams, labels)

#loss_and_metrics = model.evaluate(melgrams, labels)

