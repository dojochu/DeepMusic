import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
#from DeepMusicStyle.reverse_melspectrogram import read_audio, compute_melgram
from musicCNN.audio_processor import compute_melgram
import pandas as pd
import numpy as np

data_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/mp3/'
label_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/annotations_final.csv'
info_path = 'C:/Users/stephanie.kao/Documents/DeepMusic/DeepMusicStyle/music_data/clip_info_final.csv'


clip_info = pd.read_csv(filepath_or_buffer=info_path, delimiter='\t')
label_info = pd.read_csv(filepath_or_buffer=label_path, delimiter='\t')
clip_info['batch_id'] = pd.Series([str(b)[2] for b in clip_info.mp3_path.str.split('/')])
label_info['batch_id'] = pd.Series([str(b)[2] for b in label_info.mp3_path.str.split('/')])

SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12  # to make it 1366 frame..
FRAME_SIZE = int(N_MELS*(DURA * SR / HOP_LEN + 1))
NUM_LABELS = 188
TEST_SIZE = 2000
iterations = 1000

model_input = Input(shape=(FRAME_SIZE,),batch_shape=(None, FRAME_SIZE), name='initial_input',dtype='float32')

layer = Dense(output_dim=64, activation='softmax')(model_input)
layer2 = Dense(output_dim=25, activation='softmax')(layer)
model_output = Dense(output_dim = NUM_LABELS, activation = 'softmax')(layer2)

model = Model(input=model_input, output=model_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

melgrams = np.zeros(shape=(FRAME_SIZE,))
labels = np.zeros(shape =(NUM_LABELS))
for bid in pd.unique(clip_info.batch_id)[:2]:
    print('Training on batch_id: ' + bid)
    melgrams = np.zeros(shape=(FRAME_SIZE,))
    labels = np.zeros(shape =(NUM_LABELS,))
    for song in clip_info[clip_info['batch_id'] == bid]['mp3_path']:
        #melgrams = np.concatenate(melgrams, compute_melgram(read_audio(data_path + song)).flatten())
        try:
            melgrams = np.vstack((melgrams, compute_melgram(data_path + song).flatten()))
            add_label = label_info[label_info['mp3_path'] == song].filter(
                        items=label_info.columns.tolist()[1:-2]).transpose().iloc[:,0]
            labels = np.vstack((labels, add_label))
        except FileNotFoundError:
            continue
    model.train_on_batch(melgrams, labels)

#loss_and_metrics = model.evaluate(melgrams, labels)

