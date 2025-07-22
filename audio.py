import sounddevice as sd
import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import time

# 1. RECORD AUDIO
duration = 5  # seconds
fs = 16000  # Sampling rate (standard for audio analysis)

print("Recording...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = recording.flatten()
print("Recording complete.")

# 2. APPLY BUTTERWORTH FILTER
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut=20, highcut=2000, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

filtered_audio = apply_filter(audio)

# 3. MFCC EXTRACTION
mfccs = librosa.feature.mfcc(y=filtered_audio, sr=fs, n_mfcc=13)

# 4. APPLY DCT TO MFCCs (optional, if needed for compression/processing)
mfccs_dct = dct(mfccs, type=2, axis=1, norm='ortho')

# 5. PLOTTING
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=fs)
plt.colorbar()
plt.title('MFCCs (Filtered Audio)')
plt.tight_layout()
plt.show()
