import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================================================
# CONFIGURATION
# ================================================
DATASET_PATH = r"C:\Users\Kosha\OneDrive\Desktop\SSIP\Code\Audio Samples"
OUTPUT_PATH = r"C:\Users\Kosha\OneDrive\Desktop\SSIP\Code\MelSpectrograms"
SAMPLE_RATE = 22050
ORDER = 4
LOW_CUT = 50
HIGH_CUT = 1000
N_MELS = 128          # Number of mel frequency bins
N_FFT = 2048
HOP_LENGTH = 512

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ================================================
# ROBUST BUTTERWORTH BANDPASS FILTER
# ================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter with safety checks
    """
    # Remove DC offset
    data = data - np.mean(data)
    
    # Check for valid signal
    if np.all(data == 0) or not np.isfinite(data).all():
        return data
    
    # Normalize input to prevent overflow
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    else:
        return data
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure filter parameters are valid
    if low <= 0 or low >= 1 or high <= 0 or high >= 1 or low >= high:
        print(f"  Warning: Invalid filter range ({lowcut}-{highcut}Hz). Skipping filter.")
        return data * max_val
    
    try:
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        
        # Check for NaN or Inf in output
        if not np.isfinite(y).all():
            print("  Warning: Filter produced non-finite values. Using original signal.")
            return data * max_val
        
        # Restore original scale
        y = y * max_val
        return y
    
    except Exception as e:
        print(f"  Warning: Filter error ({e}). Using original signal.")
        return data * max_val

# ================================================
# PREPROCESS AUDIO WITH MULTIPLE SAFETY CHECKS
# ================================================
def preprocess_audio(signal, sr):
    """
    Clean and preprocess audio signal
    """
    # Remove silence/zero padding at start and end
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    # Check signal length
    if len(signal) < sr * 0.1:  # Less than 0.1 seconds
        print(f"  Warning: Signal too short ({len(signal)} samples)")
        return None
    
    # Remove DC offset
    signal = signal - np.mean(signal)
    
    # Check for non-finite values
    if not np.isfinite(signal).all():
        print("  Warning: Signal contains NaN or Inf")
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    
    return signal

# ================================================
# PROCESS AUDIO FILE AND SAVE MEL-SPECTROGRAM
# ================================================
def process_audio(file_path, save_path):
    try:
        # Load audio
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Preprocess
        signal = preprocess_audio(signal, sr)
        if signal is None:
            return False
        
        # Apply Butterworth bandpass filter
        filtered_signal = butter_bandpass_filter(signal, LOW_CUT, HIGH_CUT, sr, ORDER)
        
        # Additional check after filtering
        if not np.isfinite(filtered_signal).all():
            print(f"  Warning: Non-finite values after filtering. Using unfiltered signal.")
            filtered_signal = signal
        
        # Extract Mel-Spectrogram with error handling
        try:
            # Generate mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=filtered_signal,
                sr=sr,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                fmin=LOW_CUT,   # Use same frequency range as filter
                fmax=HIGH_CUT
            )
            
            # Convert to decibel scale (log scale, more suitable for visualization)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Check validity
            if not np.isfinite(mel_spec_db).all():
                print("  Warning: Mel-spectrogram contains non-finite values")
                mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            mel_spec_db = librosa.util.normalize(mel_spec_db)
            
        except Exception as e:
            print(f"  Mel-spectrogram extraction failed: {e}")
            return False
        
        # Save Mel-Spectrogram as an image
        plt.figure(figsize=(6, 4), dpi=100)
        librosa.display.specshow(
            mel_spec_db, 
            sr=sr, 
            x_axis='time',
            y_axis='mel',
            hop_length=HOP_LENGTH,
            fmin=LOW_CUT,
            fmax=HIGH_CUT,
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {e}")
        return False

# ================================================
# MAIN PROCESSING LOOP
# ================================================
labels = [d for d in os.listdir(DATASET_PATH) 
          if os.path.isdir(os.path.join(DATASET_PATH, d))]

total_processed = 0
total_failed = 0

for label in labels:
    input_folder = os.path.join(DATASET_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    print(f"\n Processing {label} ({len(files)} files)")
    
    processed = 0
    failed = 0
    
    for file in tqdm(files, desc=f"Mel-Spec Extraction - {label}", unit="file"):
        file_path = os.path.join(input_folder, file)
        save_path = os.path.join(output_folder, file.replace(".wav", ".png"))
        
        success = process_audio(file_path, save_path)
        if success:
            processed += 1
        else:
            failed += 1
    
    print(f"  [OK] Processed: {processed}/{len(files)}")
    if failed > 0:
        print(f"  [FAIL] Failed: {failed}/{len(files)}")
    
    total_processed += processed
    total_failed += failed

print("\n" + "="*50)
print(" MEL-SPECTROGRAM EXTRACTION COMPLETE!")
print("="*50)
print(f"Total Processed: {total_processed}")
print(f"Total Failed: {total_failed}")
print(f"Success Rate: {total_processed/(total_processed+total_failed)*100:.1f}%")
print(f"\nMel-Spectrogram images saved in: {OUTPUT_PATH}")