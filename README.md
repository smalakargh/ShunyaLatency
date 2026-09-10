# SunnyaLatency: ESP32-S3 Wake Word Detection

This project builds a lightweight keyword spotting system on an ESP32-S3. It records audio from an INMP441 microphone, converts the signal into log-mel features, runs a tiny TensorFlow Lite model locally, and triggers a cloud ASR endpoint when the keyword is detected.

The project is designed around the following flow:

1. Record 1-second audio clips from the hardware mic
2. Preprocess them into log-mel features
3. Train a small CNN keyword spotting model
4. Convert the model to TFLite
5. Export the embedded model header for the ESP32
6. Run inference on-device and trigger an HTTP event when the keyword is detected

## Verified status

The model pipeline was validated in Python with the project data:

- Input shape: (400, 101, 40)
- Output shape: (400, 2)
- Accuracy: 0.9925 (99.25%)

This confirms the dataset and trained model are valid. The remaining real-world issue is matching the live microphone input to the exact training feature pipeline on the hardware side.

---

## Project structure

```text
SunnyaLatency/
├── README.md
├── main.ino                     # ESP32-S3 firmware
├── preprocess.py                # feature extraction from WAV files
├── train.py                     # CNN training script
├── convert.py                   # TFLite conversion with int8 quantization
├── convertToCPP.py              # exports model.h from model.tflite
├── model.h                      # generated embedded TFLite model header
├── model.tflite                 # optimized TFLite model
├── kws_model.h5                 # trained Keras model
├── X_data.npy                   # generated feature dataset
├── y_data.npy                   # generated labels
├── dataset/
│   ├── background/
│   └── mykeyword/
├── AudioCollection/
│   ├── collectionESP.ino         # ESP32 recorder for mic samples
│   └── collector.py             # Python script to save WAV samples
└── other project files
```

---

## Hardware setup

### Microphone

- INMP441 I2S microphone
- Connected to ESP32-S3

### Pin mapping used in the firmware

- BCK = GPIO 4
- WS = GPIO 5
- SD = GPIO 6

This matches the firmware in `main.ino` and `AudioCollection/collectionESP.ino`.

### Display

- SSD1306 OLED via I2C
- SDA = GPIO 8
- SCL = GPIO 9

---

## Data collection workflow

The project expects 1-second audio samples at 16 kHz.

### 1. Record keyword samples

Edit `AudioCollection/collector.py` and set:

```python
LABEL = "mykeyword"
TOTAL_SAMPLES = 50
```

Then run:

```bash
python AudioCollection/collector.py
```

Press Enter for each sample and speak the keyword clearly.

### 2. Record background samples

Change:

```python
LABEL = "background"
TOTAL_SAMPLES = 50
```

Then run the script again.

Collect room noise, silence, fan noise, desk noise, etc.

### 3. Save to dataset folders

The script saves samples into:

```text
dataset/mykeyword/
dataset/background/
```

---

## Training workflow

### 1. Preprocess WAV files

```bash
python preprocess.py
```

This script loads each WAV, pads/truncates to 1 second, computes a 40-band log-mel spectrogram, and saves training data to `X_data.npy` and `y_data.npy`.

### 2. Train the model

```bash
python train.py
```

This trains a small CNN for the 2-class problem:

- background
- mykeyword

It saves the trained model as `kws_model.h5`.

---

## Model export workflow

### 1. Convert to TFLite

```bash
python convert.py
```

This creates `model.tflite` using int8 quantization for embedded inference.

### 2. Generate C array model header

```bash
python convertToCPP.py
```

This creates `model.h`, which is then included by the ESP32 firmware.

---

## Firmware workflow

The main firmware is in `main.ino`.

### Key responsibilities

- initialize I2S microphone input
- sample one second of audio
- build the log-mel feature map
- fill the TensorFlow Lite input tensor
- run inference
- inspect output scores
- trigger cloud ASR endpoint when keyword confidence passes threshold
- update OLED status

### Important model contract

The firmware must match the same feature extraction contract used in Python:

- 16 kHz audio
- 1 second capture
- 40 mel bands
- 101 time frames
- input shape: [1, 101, 40, 1]
- output shape: [1, 2]

If the live audio is not normalized and shaped consistently with the training pipeline, the model will predict background instead of the keyword.

---

## Cloud trigger behavior

The firmware includes a Wi-Fi + HTTP trigger path:

```cpp
const char* kAsrEndpoint = "http://192.168.1.50:5000/asr/trigger";
```

When the keyword confidence crosses the threshold, the ESP32 sends a JSON payload to the configured endpoint.

You should replace this with your real ASR server endpoint before deployment.

---

## Arduino setup notes

To compile the firmware in the Arduino IDE, you must install:

- ESP32 board package
- Arduino core for ESP32
- SSD1306 library
- Adafruit GFX library
- TensorFlow Lite for Microcontrollers library

The project also depends on the generated `model.h` embedded model.

---

## Common troubleshooting

### Model always predicts background

Check:

- input audio matches 16 kHz / 1 second
- mic gain is not too low or clipping
- preprocessing is identical between training and live audio
- DC offset and amplitude normalization are handled correctly
- keyword samples are collected with the same hardware setup

### Serial reads fail during collection

Check:

- COM port is correct
- baud rate matches
- ESP32 is connected and powered
- I2S pins are correct
- the ESP32 recorder sketch is uploaded properly

### TFLite compile problems on ESP32

Check:

- correct board selected
- Arduino libraries installed
- model header regenerated after model export
- TensorFlow Lite library version is compatible with your build

---

## Recommended workflow for best results

1. Collect keyword samples with the real mic
2. Collect background samples with the real mic
3. Run `preprocess.py`
4. Run `train.py`
5. Inspect validation accuracy
6. Run `convert.py`
7. Run `convertToCPP.py`
8. Upload `main.ino` to the ESP32
9. Test serial output and adjust threshold/gain

---

## Summary

This project is a compact wake-word detector built for an ESP32-S3 using a CNN trained on log-mel features. The workflow is intended to keep inference lightweight and local while enabling a cloud ASR trigger after a valid wake-word event.

The most important real-world requirement is matching the live mic signal with the exact feature representation used during training.
