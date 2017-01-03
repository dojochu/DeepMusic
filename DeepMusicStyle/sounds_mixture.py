import tensorflow as tf
import musicCNN.music_tagger_cnn as mtc
import musicCNN.music_tagger_crnn as mtcr
import musicCNN.audio_processor as ap
import numpy as np
#from keras import models
import keras


def mix_music(network_path, initial_song, content_song_path, style_song_path, iterations,
        content_weight, style_weight, style_blend_weights, tv_weight,
        learning_rate, print_iterations=None, checkpoint_iterations=None):

    image = None

#Create Melgram Spectrogram feature input
    content_melgram = ap.compute_melgram(content_song_path)
    style_melgram = ap.compute_melgram(style_song_path)

#Create tensor flow model

    tensor_graph = tf.get_default_graph()
    tensor_run = tf.get_default_session()

    content_input = tf.placeholder(dtype=tf.float32, shape = np.shape(content_melgram), name='content_input_placeholder')
    style_input = tf.placeholder(dtype=tf.float32, shape=np.shape(style_melgram), name='style_input_placeholder')

#    load music tagger weights from h5 file
    model = mtcr.MusicTaggerCRNN('msd', include_top=False)
    with tensor_run as sess:
        features = model.layers[5].output.eval(feed_dict={content_input:content_melgram})
        print(features)
    return (model, image)