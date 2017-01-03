
from .sounds_mixture import mix_music
import numpy as np
import tensorflow as tf
import os

audio_path = 'DeepMusicStyle/music_data/The Wanted - Glad You Came (Instrumental Version).wav'
style_path = 'DeepMusicStyle/music_data/Hallelujah-Pentatonix.wav'
model_path = 'musicCNN/data/music_tagger_crnn_weights_tensorflow.h5'
#,
#              'NeuralStyle/examples/Heartbeats.wav',
#              'NeuralStyle/examples/.wav',
#              'NeuralStyle/Children (Club Radio Edit).wav',
#              'NeuralStyle/The Wanted - Glad You Came (Instrumental Version).wav'

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000

(model, result_image) =  mix_music(
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
    print_iterations=None,
    checkpoint_iterations=None
)


