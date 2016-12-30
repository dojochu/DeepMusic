import musicCNN.music_tagger_cnn as mtc
import musicCNN.music_tagger_cnn as mtcr
import musicCNN.audio_processor as ap
import numpy as np

audio_path = ['NeuralStyle/examples/Sparks.wav']
#,
#              'NeuralStyle/examples/Heartbeats.wav',
#              'NeuralStyle/examples/.wav',
#              'NeuralStyle/Children (Club Radio Edit).wav',
#              'NeuralStyle/The Wanted - Glad You Came (Instrumental Version).wav'

music_tagger_model = mtc.MusicTaggerCNN(weights='msd', include_top=False)

melgrams = np.zeros((0, 1, 96, 1366))
for path in audio_path:
    melgrams = np.concatenate(melgrams, ap.compute_melgram(path), axis = 0)

features = music_tagger_model.predict(x=melgrams)



