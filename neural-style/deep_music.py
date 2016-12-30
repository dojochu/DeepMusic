from pyAudioAnalysis import audioBasicIO as io

musicfile = '/examples/03 - Sparks.mp3'

io.convertDirMP3ToWav('/examples/', 32000, 2, True)


