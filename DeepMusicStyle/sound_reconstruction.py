import numpy as np
import scipy.optimize as so
import DeepMusicStyle.reverse_melspectrogram as rm
import os

spark = rm.compute_melgram(rm.read_audio(os.path.abspath('DeepMusicStyle/music_data/Sparks.wav')))
initial = rm.read_audio(os.path.abspath('DeepMusicStyle/music_data/white_noise.wav'))

def optim(initial_sound, maxiter):
    #return so.minimize(objective, initial_sound, method = 'Nelder-Mead')
    return so.fmin_l_bfgs_b(func=objective, x0=initial_sound, approx_grad=True, maxiter=maxiter, iprint=2)

def objective(x):
    return np.sum((spark- rm.compute_melgram(x))**2)

def gradient(x, spark):
    return np.sum(2*(spark - rm.compute_melgram(x)))


maxiter = 100

opt = optim(initial, maxiter)

import librosa
librosa.output.write_wav('DeepMusicStyle/music_data/output.wav', opt[0], sr = 1000)