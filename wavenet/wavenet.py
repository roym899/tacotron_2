import math
import numpy as np
import wave
import scipy
import array
import os
from os.path import expanduser
import scipy.io.wavfile

def reconstruct_signal(mag_spectogram, fftsize, hops, iterations):
    """
    Function to reconstruct a time signal from a given spectogram using the Griffin-Lim algorithm
    :param spectogram: 2D numpy array containing the MAGNITUDE spectogram of the sequence that should be reconstructed.
                       The rows correspond to time steps and the columns to frequency bins.
    :param fftsize: Stepsize with which the FFT transform was performed
    :param hops:    Number of samples the window is shifted after computing the FFT.
    :param iterations: Number of iterations the Griffin_lim algorithm should perform.
    :return: Returns the reconstructed (audio) signal as a 1D numpy array
    """

    time_steps = mag_spectogram.shape[0]
    len_samples = int(time_steps*hops+fftsize)
    reconstructed = np.random.randn(len_samples)

    i = iterations
    while i > 0:
        print(i)
        i -= 1
        stft_reconstructed = calculate_stft(reconstructed, fftsize, hops)
        stft_reconstructed_angle = np.angle(stft_reconstructed)
        # Replace the magnitude part of the STFT of the reconstructed signal by the given magnitude spectogram
        stft_reconstructed = mag_spectogram * np.exp(1.0j*stft_reconstructed_angle)

        #prev_reconstructed = reconstructed
        reconstructed = calculate_inverse_stft(stft_reconstructed, fftsize, hops)


        # Calculation of the error for debug purposes, should decrease over timer
        # squared_difference = (reconstructed-prev_reconstructed) ** 2
        # sqrt_error = math.sqrt(sum(squared_difference)/reconstructed.size)
    return reconstructed


def calculate_stft(time_signal,fftsize,hops):
    """
    Fucntion to calculate the Short Time Fourier Transform of a given time signal
    :param time_signal: 1D-numpy-array of time signal
    :param fftsize: Size of the SFTF (int)
    :param hops: Number of samples the window is shifted after computing the FFT
    :return:    Returns The STFT of a time signal as  a 2D numpy array, where the rows represent time steps and the
                columns represent frequency bins.
    """
    hann_window = np.hanning(fftsize)
    stft = np.array([np.fft.fft(hann_window * time_signal[i:(i + fftsize)])
                     for i in range(0, len(time_signal) - fftsize, hops)])
    return stft

def calculate_mag(spectogramm,cutoff = 128):
    """
    Function takes the input spectogram calculates the magnitude and cutsoff the high frequencies

    :param spectogramm: input spectorgram, size: seq_length, frequency_bins
    :param cutoff: cutoff frequency at which the magnitude spectrogramm is cut off
    :return: cutoff magnitude array of size: seq_length x (0:cutoff)
    """
    magnitude = abs(spectogramm)**2;
    cut_off_magnitude = np.copy(magnitude[:,0:cutoff])
    return cut_off_magnitude

def pad_magnitude(cut_off_magnitude,desired_frequency_bins = 2048):
    """
    Takes a magnitude spectogramm, that was cut off, and restores it to it's original amount of frequency bins
    by zero padding the higher frequencies
    :param cut_off_magnitude: input magnitude spectromgramm that was cut off; size: seq_length, frequency_bins
    :param desired_frequency_bins: Amount of frequency bins we want to have
    :return: return the same cut_off_magnitude spectogramm with additional zero padded high frequencies;
    size: seq_length, desired_frequency_bins
    """
    shape = np.shape(cut_off_magnitude)
    output_length = shape[0]
    frequency_bins = shape[1]
    padded_magnitude = np.zeros((output_length,desired_frequency_bins))
    padded_magnitude[:,0:frequency_bins] = cut_off_magnitude
    return padded_magnitude

def calculate_inverse_stft(spectogram,fftsize,hops):
    """
    Calculates the time signal from a given spectogram using an inverse STFT.
    :param spectogram: 2D numpy-array, where the rows represent time_steps and the columns frequency bins
    :param fftsize: Size of the FFT
    :param hops: Number of samples the window is shifted after computing the FFT
    :return: Return a 1D numpy array which contains the time signal
    """
    hann_window = np.hanning(fftsize)
    time_steps = spectogram.shape[0]
    len_samples = int(time_steps*hops+fftsize)
    istft = np.zeros(len_samples)
    for n, i in enumerate(range(0, len(istft)-fftsize, hops)):
        istft[i:i+fftsize] += hann_window * np.real(np.fft.ifft(spectogram[n]))
    return istft


def load_audio(path, expected_samplerate=44100):
    """
    Loads in wav-file and transforms it into a 1D numpy array.
    :param path: Path to the wav-file
    :param expected_samplerate: Expected samplerate of the wav file
    :return: 1D numpy-array containing the time signal
    """
    fs, y = scipy.io.wavfile.read(path)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y * (1.0 / 32768)
    elif num_type == 'int32':
        y = y * (1.0 / 2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expected_samplerate:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)


def save_audio(time_signal, samplerate, output_file = 'output.wav'):
    """
    Saves the given time signal as a wav-file
    :param time_signal: 1D-numpy array, should be normalized to the range [-1,1]
    :param samplerate: samplerate of the signal (Hz)
    :param output_file: Name of the file to save
    :return:
    """
    if max(time_signal) > 1 or min(time_signal) < -1:
        print('Input values out of range')
        return 0
    time_signal *= 32767.0
    data = array.array('h')
    for i in range(0,len(time_signal)):
        data.append(int(round(time_signal[i])))
    f = wave.open(output_file,'w')
    f.setparams((1, 2, samplerate, 0, 'NONE', 'Uncompressed'))
    f.writeframes(data.tostring())
    f.close()


### Test procedure ###

path = 'C:\\Users\\Thomas\Desktop\\KTH\Period 4\\deep_learning\\project\\wavenet\\speech.wav'
input_signal = load_audio(path,expected_samplerate=16000)
fftsize = 2048;
hops = 512;
#hops = fftsize//8

#print(max(input_signal))
stft_input = calculate_stft(input_signal, fftsize, hops)

cutoff_mag_input = calculate_mag(stft_input)
mag_input = pad_magnitude(cutoff_mag_input)


reconstructed = reconstruct_signal(mag_input, fftsize, hops, 100)
# Scale the reconstructed signal
max_value = np.max(abs(reconstructed))
if max_value > 1.0:
    reconstructed = reconstructed/max_value

save_audio(reconstructed, 16000,output_file='output_128.wav')
print('Done')
