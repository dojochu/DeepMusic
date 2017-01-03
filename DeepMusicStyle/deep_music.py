import musicCNN.music_tagger_cnn as mtc
import musicCNN.music_tagger_crnn as mtcr
import musicCNN.audio_processor as ap
import numpy as np
import tensorflow as tf

audio_path = 'music_data/Sparks.wav'
#,
#              'NeuralStyle/examples/Heartbeats.wav',
#              'NeuralStyle/examples/.wav',
#              'NeuralStyle/Children (Club Radio Edit).wav',
#              'NeuralStyle/The Wanted - Glad You Came (Instrumental Version).wav'

music_tagger_model = mtcr.MusicTaggerCRNN(weights='msd', include_top=True)
#music_tagger_model = mtc.MusicTaggerCNN(weights='msd', include_top=True)

#melgrams = np.zeros((0, 1, 96, 1366))
melgrams = ap.compute_melgram(audio_path)
#for path in audio_path:
#    melgrams = np.concatenate(melgrams,ap.compute_melgram(path))

sess = tf.Session()

layer0Output = music_tagger_model.get_layer('bn_0_freq').get_output_at(0).eval(melgrams, sess)
print(layer0Output)

#features = music_tagger_model.predict(x=melgrams)

#print(features)
