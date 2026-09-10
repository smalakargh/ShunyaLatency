import os
import numpy as np
import librosa

DATA_DIR = "dataset"
CLASSES = ["background", "mykeyword"]
SAMPLE_RATE = 16000
DURATION = 1.0   # 1 second
SAMPLES_PER_FILE = int(SAMPLE_RATE * DURATION)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Pad or truncate to exact duration
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)), 'constant')
    else:
        audio = audio[:SAMPLES_PER_FILE]

    # Extract 40-channel log-Mel filterbank energies
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=40, n_fft=512, hop_length=160)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel.T  # Shape: (Time, Features)

X, y = [], []
for label_idx, label_name in enumerate(CLASSES):
    class_folder = os.path.join(DATA_DIR, label_name)
    for file_name in os.listdir(class_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(class_folder, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)
np.save("X_data.npy", X)
np.save("y_data.npy", y)
print("Preprocessing complete. Data saved to disk.")