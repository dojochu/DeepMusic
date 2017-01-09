import librosa
import numpy as np
import tensorflow as tf

def read_audio(audio_path):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample - n_sample_fit) / 2):int((n_sample + n_sample_fit) / 2)]
    return src

def compute_melgram(src, SR=12000, N_FFT=512, N_MELS=96, HOP_LEN=256, DURA=29.0):


    ret = logamplitude(melspectrogram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret

def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, **kwargs):

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=2)
    # Build a Mel filter
    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)

def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length)) ** power

    return S, n_fft

def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):

 #   if amin <= 0:
 #       raise ParameterError('amin must be strictly positive')
    import six

    magnitude = np.abs(S)

    if six.callable(ref_power):
        # User supplied a function to calculate reference power
        __ref = ref_power(magnitude)
    else:
        __ref = np.abs(ref_power)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, __ref))

    if top_db is not None:
#        if top_db < 0:
#            raise ParameterError('top_db must be non-negative positive')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - fftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights

def fft_frequencies(sr=22050, n_fft=2048):

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):

    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def hz_to_mel(frequencies, htk=False):

    frequencies = np.atleast_1d(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels

def mel_to_hz(mels, htk=False):


    mels = np.atleast_1d(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    import scipy
    import six
    from librosa import util
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft
        #win_length = tf.constant(n_fft)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length.value() / 4)
        #hop_length = win_length/4
        #hop_length.to_int64()

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)
        #fft_window = tf.constant(scipy.signal.hann(convertTFtoNP(win_length), sym=False))

    elif six.callable(window):
        # User supplied a window function

        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
#        if fft_window.size != n_fft:
#           raise ParameterError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)
    #fft_window.assign(util.pad_center(convertTFtoNP(fft_window), n_fft))

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))
    #tf.reshape(fft_window, (-1,1))

    # Pad the time series so that frames are centered
    if center:
         util.valid_audio(y)
         y_ = np.pad(convertTFtoNP(y), int(n_fft // 2), mode='reflect')
    #    padding = int(n_fft // 2)
    #    y_frames = tf.pad(y, [[padding, padding],[padding,padding]], mode='REFLECT')

    # Window the time series.
    y_frames = util.frame(y_, frame_length=n_fft, hop_length=hop_length)
    #y_frames.assign(util.frame(convertTFtoNP(y_frames), frame_length=n_fft, hop_length=hop_length))

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                          order='F')
    #stft_matrix = tf.zeros((int(1 + n_fft // 2), y_frames.get_shape()[1]._value),
    #                      dtype=dtype,
    #                      order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    #n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.get_shape()[0]._value *
    #                                      convertTFtoNP(stft_matrix).itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
    #for bl_s in range(0, stft_matrix.get_shape()[1]._value, n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        #bl_t = min(bl_s + n_columns, stft_matrix.get_shape()[1]._value)
        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()
        #tf.scatter_update(stft_matrix, tf.constant(range(bl_s,bl_t)), tf.conj(tf.slice(tf.fft(
        #                                    fft_window * tf.slice(
        #                                    y_frames, [0,bl_s],[y_frames.get_shape()[0]._value,bl_t-bl_s])),
        #                                    [0],[stft_matrix.get_shape()[0]._value])))


    return stft_matrix

def convertTFtoNP(x):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess.run(x)
    #return x

def test(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    import scipy
    import six
    from librosa import util
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft
        # win_length = tf.constant(n_fft)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)
        # hop_length = win_length/4
        # hop_length.to_int64()

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)
        # fft_window = tf.constant(scipy.signal.hann(convertTFtoNP(win_length), sym=False))

    elif six.callable(window):
        # User supplied a window function

        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        #        if fft_window.size != n_fft:
        #           raise ParameterError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)
    # fft_window.assign(util.pad_center(convertTFtoNP(fft_window), n_fft))

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))
    # tf.reshape(fft_window, (-1,1))

    if center:
        #util.valid_audio(y)
        #y_ = np.pad(convertTFtoNP(y), int(n_fft // 2), mode='reflect')
        padding = int(n_fft // 2)
        y_frames = tf.Variable(tf.pad(y, [[padding,padding]], mode='REFLECT'))

    # Window the time series.
    #y_frames = util.frame(y_, frame_length=n_fft, hop_length=hop_length)
    #y_frames.assign(librosa.util.frame(convertTFtoNP(y_frames), frame_length=n_fft, hop_length=1))

    y_frames = frame(y_frames, n_fft, hop_length)

    # Pre-allocate the STFT matrix
    #stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
    #                       dtype=dtype,
    #                      order='F')
    stft_matrix = tf.Variable(tf.zeros(y_frames.get_shape()[1]._value,(int(1 + n_fft // 2)),dtype='float32'))

    # how many columns can we fit within MAX_MEM_BLOCK?
    #n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
    #                                      stft_matrix.itemsize))

    n_columns = int(librosa.util.MAX_MEM_BLOCK / (stft_matrix.get_shape()[1]._value *
                                          convertTFtoNP(stft_matrix).itemsize))

    #for bl_s in range(0, stft_matrix.shape[1], n_columns):
    for bl_s in range(0, stft_matrix.get_shape()[0]._value, n_columns):
        #bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        bl_t = min(bl_s + n_columns, stft_matrix.get_shape()[0]._value)
        # RFFT and Conjugate here to match phase from DPWE code
        #stft_matrix[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window *
        #                                    y_frames[:, bl_s:bl_t],
        #                                    axis=0)[:stft_matrix.shape[0]].conj()

        stft_matrix = tf.scatter_update(stft_matrix, tf.constant(list(range(bl_s, bl_t,1))), tf.conj(tf.slice(tf.fft(
                                            fft_window * tf.slice(
                                            y_frames, [0,bl_s], [y_frames.get_shape()[0]._value,bl_t-bl_s])),
                                            [0],[stft_matrix.get_shape()[0]._value])))

    return stft_matrix

def frame(y, frame_length=2048, hop_length=512):
    n_frames = 1 + int((len(convertTFtoNP(y)) - frame_length) / hop_length)
    new_frame = tf.Variable(tf.zeros([n_frames, frame_length]))
    for i in range(0, n_frames):
        ind = tf.constant([i])
        update = tf.slice(y, [i*hop_length,], [frame_length,])
        new_frame = tf.scatter_update(new_frame, ind, [update])
    return tf.transpose(new_frame)