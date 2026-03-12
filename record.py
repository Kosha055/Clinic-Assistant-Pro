import os
import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import butter, filtfilt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================
# CONFIGURATION
# ================================================
OUTPUT_PATH = r"C:\Users\Kosha\OneDrive\Desktop\SSIP\Code\Recorded"

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 5
CHANNELS = 1

# Filter parameters
ORDER = 4
LOW_CUT = 50
HIGH_CUT = 1000

# Mel-spectrogram parameters
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Fixed output size
FIXED_SAMPLES = SAMPLE_RATE * DURATION
FIXED_TIME_FRAMES = int(np.ceil(FIXED_SAMPLES / HOP_LENGTH))

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ================================================
# BUTTERWORTH BANDPASS FILTER
# ================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter"""
    data = data - np.mean(data)
    
    if np.all(data == 0) or not np.isfinite(data).all():
        return data
    
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    else:
        return data
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0 or low >= 1 or high <= 0 or high >= 1 or low >= high:
        return data * max_val
    
    try:
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        
        if not np.isfinite(y).all():
            return data * max_val
        
        return y * max_val
    except:
        return data * max_val

# ================================================
# FIX AUDIO LENGTH
# ================================================
def fix_audio_length(signal, target_length):
    current_length = len(signal)
    
    if current_length < target_length:
        pad_length = target_length - current_length
        signal = np.pad(signal, (0, pad_length), mode='constant', constant_values=0)
    elif current_length > target_length:
        signal = signal[:target_length]
    
    return signal

# ================================================
# EXTRACT MEL-SPECTROGRAM ARRAY
# ================================================
def extract_melspec_array(signal, sr):
    """Extract mel-spectrogram from audio signal"""
    try:
        signal, _ = librosa.effects.trim(signal, top_db=20)
        signal = fix_audio_length(signal, FIXED_SAMPLES)
        signal = signal - np.mean(signal)
        
        if not np.isfinite(signal).all():
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
        
        filtered_signal = butter_bandpass_filter(signal, LOW_CUT, HIGH_CUT, sr, ORDER)
        
        if not np.isfinite(filtered_signal).all():
            filtered_signal = signal
        
        mel_spec = librosa.feature.melspectrogram(
            y=filtered_signal,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmin=LOW_CUT,
            fmax=HIGH_CUT
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if not np.isfinite(mel_spec_db).all():
            mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
        
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        if mel_spec_normalized.shape[1] < FIXED_TIME_FRAMES:
            pad_width = FIXED_TIME_FRAMES - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :FIXED_TIME_FRAMES]
        
        return mel_spec_normalized
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# ================================================
# RECORD AUDIO
# ================================================
def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE, device=None):
    """Record audio from microphone"""
    print(f"Recording {duration}s...")
    
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype='float32',
            device=device
        )
        sd.wait()
        
        if CHANNELS == 1:
            recording = recording.flatten()
        
        return recording
        
    except Exception as e:
        print(f"Recording failed: {e}")
        return None

# ================================================
# SAVE AUDIO FILE
# ================================================
def save_audio_wav(signal, sr, filepath):
    """Save audio as WAV file"""
    try:
        import soundfile as sf
        sf.write(filepath, signal, sr)
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False

# ================================================
# PROCESS AND SAVE
# ================================================
def process_and_save(audio_signal, timestamp):
    """Process audio and save WAV and NPY files"""
    wav_filename = f"recording_{timestamp}.wav"
    npy_filename = f"melspec_{timestamp}.npy"
    
    wav_path = os.path.join(OUTPUT_PATH, wav_filename)
    npy_path = os.path.join(OUTPUT_PATH, npy_filename)
    
    # Save WAV
    save_audio_wav(audio_signal, SAMPLE_RATE, wav_path)
    
    # Extract mel-spectrogram
    print("Processing...")
    mel_spec = extract_melspec_array(audio_signal, SAMPLE_RATE)
    
    if mel_spec is not None:
        np.save(npy_path, mel_spec)
        print(f"Saved: {npy_filename}")
        
        return {
            'wav_path': wav_path,
            'npy_path': npy_path,
            'mel_spec': mel_spec
        }
    else:
        print("Processing failed")
        return None

# ================================================
# CLASSIFY (PLACEHOLDER FOR YOUR MODEL)
# ================================================
def classify_audio(mel_spec_array):
    """
    Classify using trained model
    
    TODO: Replace with your model
    """
    return "Unknown", 0.0

# ================================================
# MAIN FUNCTION
# ================================================
def main():
    """Main recording loop"""
    
    while True:
        print("\n1. Record  2. Exit")
        choice = input("Choice: ").strip()
        
        if choice == '1':
            # Record
            audio_data = record_audio()
            
            if audio_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result = process_and_save(audio_data, timestamp)
                
                if result:
                    # Classify
                    prediction, confidence = classify_audio(result['mel_spec'])
                    print(f"Result: {prediction}")
                    if confidence > 0:
                        print(f"Confidence: {confidence:.1%}")
        
        elif choice == '2':
            print("Exiting...")
            break

# ================================================
# SINGLE RECORDING MODE (FOR AUTOMATION)
# ================================================
def record_once():
    """Single recording without interaction"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    audio_data = record_audio()
    
    if audio_data is not None:
        result = process_and_save(audio_data, timestamp)
        
        if result:
            prediction, confidence = classify_audio(result['mel_spec'])
            print(f"{prediction}")
            return result['npy_path']
    
    return None

# ================================================
# RUN
# ================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Single recording mode
        record_once()
    else:
        # Interactive mode
        try:
            main()
        except KeyboardInterrupt:
            print("\nStopped")