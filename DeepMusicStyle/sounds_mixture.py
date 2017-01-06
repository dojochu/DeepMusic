import tensorflow as tf
import musicCNN.music_tagger_cnn as mtc
import musicCNN.music_tagger_crnn as mtcr
import musicCNN.audio_processor as ap
import numpy as np
#from keras import models
import keras
import os
import librosa
from DeepMusicStyle.reverse_melspectrogram import compute_melgram, read_audio
import scipy.optimize as sopt


def mix_music(network_path, initial_song, content_song_path, style_song_path, iterations,
        content_weight, style_weight, style_blend_weights, tv_weight,
        learning_rate, SR):

    # Constants
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

#Create Melgram Spectrogram feature input
    #content = read_audio(content_song_path)
    #style = read_audio(style_song_path)
    content = tf.constant(read_audio(content_song_path).astype('float32'))
    style = tf.constant(read_audio(style_song_path).astype('float32'))
    #content = tf.constant(ap.compute_melgram(content_song_path))
    #style = tf.constant(ap.compute_melgram(style_song_path))

    if initial_song is None:
        #result = read_audio(os.path.abspath('DeepMusicStyle/music_data/white_noise.wav'))
        result= tf.Variable(read_audio(os.path.abspath('DeepMusicStyle/music_data/white_noise.wav')).astype('float32'))
        #result = tf.Variable(ap.compute_melgram(os.path.abspath('DeepMusicStyle/music_data/white_noise.wav')))
    else:
        #result = read_audio(initial_song)
        result= tf.Variable(read_audio(initial_song).astype('float32'))
        #result = tf.Variable(ap.compute_melgram(initial_song).astype('float32'))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
#Load music tagger weights from h5 file
    model = mtcr.MusicTaggerCRNN('msd', include_top=False, input_tensor = tf.placeholder(dtype='float32',shape=(1,1,96,1366),name='input0'))

# Create melgrams
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #content_melgram = compute_melgram(content, SR=SR, HOP_LEN=HOP_LEN, N_FFT=N_FFT,N_MELS=N_MELS)
    #style_melgram = compute_melgram(style, SR=SR, HOP_LEN=HOP_LEN, N_FFT=N_FFT, N_MELS=N_MELS)
    #result_melgram = compute_melgram(result, SR=SR, HOP_LEN=HOP_LEN, N_FFT=N_FFT, N_MELS=N_MELS)
    content_melgram = compute_melgram(content)
    style_melgram = compute_melgram(style)
    result_melgram = compute_melgram(result)


#Create feature matrices
    #content_feature = tf.Variable(model.predict(x = content_melgram))
    #style_feature = tf.Variable(model.predict(x=style_melgram))
    #result_feature = tf.Variable(model.predict(x=result_melgram))
    content_feature = tf.constant(content_melgram)
    style_feature = tf.constant(style_melgram)
    result_feature = tf.Variable(result_melgram)

#Create Loss functions
    #content_loss = content_weight * (2 * tf.nn.l2_loss(content_feature - result_feature)/content_feature.size)
    #style_loss = style_weight * (2 * tf.nn.l2_loss(style_feature - result_feature)/style_feature.size)
    content_loss = content_weight * (2 * tf.nn.l2_loss(content_feature - result_feature))
    style_loss = style_weight * (2 * tf.nn.l2_loss(style_feature - result_feature))

    loss = content_loss + style_loss

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def print_progress(i, total_iteration):
        print('iteration %s of %s' % (i, total_iteration))

    # optimization
    best_loss = float('inf')
    song = None
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(iterations):
            last_step = (i == iterations - 1)
            print_progress(i, iterations)
            train_step.run()

            this_loss = loss.eval()
            if this_loss < best_loss:
                best_loss = this_loss
                song = result.eval()

    return (model, song, result)

