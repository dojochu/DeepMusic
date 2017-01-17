import numpy as np
import musicCNN.audio_processor as ap
import pandas as pd
import matplotlib.pyplot as pyp



def plot_spectrogram_grid(audio_files):

    music=[]
    for audio_file in audio_files:
        music.append(ap.compute_melgram(audio_file))

    fig, ax = pyp.subplots(2, int(len(music)/2+1))

    for ind in range(0,len(music)):
        pl = ax.flat[ind].pcolormesh(music[ind][0][0])

    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85,0.15, 0.05, 0.7])
    fig.colorbar(pl, cax = cbar)

    pyp.show()