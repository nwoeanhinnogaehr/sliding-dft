import numpy as np
from scipy.misc import imsave
from scipy.io import wavfile

# Perform the sliding DFT of a given size (frequency domain resolution) on an array of samples.
# Since this is a time-recursive transform, the spectrum from the last time step of the previous
# block can be optionally passed to preserve coherence between blocks.
def forward(samples, size, prev=None):
    damping = 0.99 # used to prevent floating point errors from accumulating
    spectrum = np.zeros((size + 1, size), dtype=np.complex128)
    if prev is not None:
        spectrum[0] = prev
    twiddle = np.exp(2*np.pi*1j*np.arange(size)/size)
    for t in range(size):
        spectrum[t+1] = (damping * spectrum[t] - damping**size * samples[t] + samples[t+size]) * twiddle
    return spectrum[1:,:]

# Apply a Hann window to the spectrum.
# TODO handle border cases properly
def hann(spectrum):
    out = np.zeros(spectrum.shape, dtype=np.complex128)
    #out[0] = 0.5*spectrum[0] - 0.25*(spectrum[1] + spectrum[-1])
    #out[-1] = 0.5*spectrum[-1] - 0.25*(spectrum[-2] + spectrum[0])
    for t in range(1, out.shape[0] - 1):
        out[t] = 0.5*spectrum[t] - 0.25*(spectrum[t-1] + spectrum[t+1])
    return out

# Perform the inverse sliding DFT on a spectrum, and return the reconstructed (real) samples.
def backward(spectrum):
    block_size = spectrum.shape[0]
    size = spectrum.shape[1]
    signal = np.zeros(block_size)
    for t in range(block_size):
        signal[t] = sum(spectrum[t].real)/size
    return signal

# function to transform spectrum for visualization
def specfn(spec):
    half = spec[:,:spec.shape[1]//2]
    return np.abs(half)+np.array([half.real*0, abs(np.angle(half)), half.real*0])

size = 1024 # DFT size
length = 8192 # total length of sound to generate
sample_rate = 44100
spectrum = [0] # spectrum of current block
t = 0 # current processing time
image = None # output image
# input samples
sampin = np.sin(np.arange(length+size)/44100*2*np.pi*440)
# output samples
sampout = np.array([])

while t < length:
    # forward
    samples = sampin[t:t+size*2]
    spectrum = forward(samples, size, spectrum[-1])

    # window
    windowed = hann(spectrum)
    if image is None:
        image = specfn(windowed)
    else:
        image = np.concatenate((image, specfn(windowed)), axis=1)

    # backward
    recon = backward(spectrum)
    sampout = np.append(sampout, recon)

    t += size

# write output
print("in: ", sampin[:length].shape)
print("out: ", sampout.shape)
print(image.shape)
image = np.swapaxes(image, 0, 2)
image = np.swapaxes(image, 0, 1)
imsave("out.png", image/np.max(image))
sampout /= np.max(np.abs(sampout))
sampin /= np.max(np.abs(sampin))
wavfile.write("out.wav", sample_rate, sampout[size:])
wavfile.write("in.wav", sample_rate, sampin[size:length])
