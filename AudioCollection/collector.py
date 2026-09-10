import os
import time
import serial

from scipy.io.wavfile import write
import numpy as np

# Configuration
PORT = "COM9"          # Change to your ESP32-S3 serial port
BAUD = 500000
SAMPLE_RATE = 16000
LABEL = "mykeyword"    # Change to "background" for noise samples
TOTAL_SAMPLES = 50

output_dir = os.path.join("dataset", LABEL)
os.makedirs(output_dir, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)

print(f"--- Recording via INMP441 for: {LABEL} ---")
existing_files = len(os.listdir(output_dir))

for i in range(1, TOTAL_SAMPLES + 1):
    file_idx = existing_files + i
    filename = os.path.join(output_dir, f"{file_idx:03d}.wav")

    input(f"[{i}/{TOTAL_SAMPLES}] Press [Enter] to record '{LABEL}'...")

    ser.reset_input_buffer()
    ser.write(b'R')

    audio_bytes = bytearray()
    timeout = time.time() + 3

    while len(audio_bytes) < 32000 and time.time() < timeout:
        chunk = ser.read(32000 - len(audio_bytes))
        if not chunk:
            break
        audio_bytes.extend(chunk)

    if len(audio_bytes) < 32000:
        print("Warning: incomplete audio. Retrying...")
        continue

    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    write(filename, SAMPLE_RATE, audio_data)
    print(f"Saved: {filename}\n")

ser.close()
print("Hardware audio collection complete!")