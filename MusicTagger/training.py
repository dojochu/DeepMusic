import tensorflow as tf
from DeepMusicStyle.reverse_melspectrogram import read_audio, compute_melgram
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


input = tf.placeholder(dtype='float32',shape = (FRAME_SIZE,1))
truth = tf.placeholder(dtype='float32', shape = (NUM_LABELS,1))

weights = tf.Variable(initial_value = tf.random_uniform(
                                    shape=(NUM_LABELS,FRAME_SIZE)))
biases = tf.Variable(initial_value = tf.random_uniform(shape=(NUM_LABELS,1)))

pred = tf.add(tf.matmul(weights, input), biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, truth))

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


for bid in pd.unique(clip_info.batch_id):
    melgrams = np.zeros(shape=(FRAME_SIZE,))
    labels = np.zeros(shape =(NUM_LABELS))
    for song in clip_info[clip_info['batch_id'] == bid]['mp3_path']:
        melgrams = np.concatenate(melgrams, compute_melgram(read_audio(data_path + song)).flatten())
        labels = np.concatenate(labels, label_info[label_info['mp3_path'] == song].filter(
                                items=label_info.columns.tolist()[1:-1]).transpose())
        optimize.run(feed_dict={input: melgrams, truth: labels})

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,1), tf.argmax(pred, 1)),tf.float32))

test_x = compute_melgram(read_audio(data_path + clip_info['mp3_path'][:TEST_SIZE]))
test_y = label_info[label_info['mp3_path'].isin(clip_info['mp3_path'][:TEST_SIZE])].filter(
                                                    items=label_info.columns.tolist()[1:-1]).transpose()

print(accuracy.eval(feed_dict={input:test_x, truth:test_y}))

