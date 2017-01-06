
from .sounds_mixture import mix_music
import numpy as np
import tensorflow as tf
import os

audio_path = 'DeepMusicStyle/music_data/Sparks.wav'
style_path = 'DeepMusicStyle/music_data/Hallelujah-Pentatonix.wav'
model_path = 'musicCNN/data/music_tagger_crnn_weights_tensorflow.h5'
output_path = 'DeepMusicStyle/music_data/output.wav'
#,
#              'NeuralStyle/examples/Heartbeats.wav',
#              'NeuralStyle/examples/.wav',
#              'NeuralStyle/Children (Club Radio Edit).wav',
#              'NeuralStyle/The Wanted - Glad You Came (Instrumental Version).wav'

CONTENT_WEIGHT = 10
STYLE_WEIGHT = 10
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
SR = 12000

(model, result_song, features) =  mix_music(
    network_path=os.path.abspath(model_path),
    initial_song=None,
    content_song_path=os.path.abspath(audio_path),
    style_song_path=os.path.abspath(style_path),
    iterations=ITERATIONS,
    content_weight=CONTENT_WEIGHT,
    style_weight=STYLE_WEIGHT,
    style_blend_weights=1.0,
    tv_weight=TV_WEIGHT,
    learning_rate=LEARNING_RATE,
    SR = SR
)

import librosa
librosa.output.write_wav(os.path.abspath(output_path), result_song, sr = SR)