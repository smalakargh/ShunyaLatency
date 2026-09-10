#include <Arduino.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// TensorFlow Lite for Microcontrollers headers (Chirale Library)
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h" // Generated TFLM model header

// OLED Pin & Display Configuration (I2C)
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
#define I2C_SDA 8
#define I2C_SCL 9
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// INMP441 Microphone Pin Configuration (I2S)
#define I2S_PORT I2S_NUM_0
#define I2S_SCK 4
#define I2S_WS 5
#define I2S_SD 6

constexpr int kAudioSampleRate = 16000;
constexpr int kWindowSamples = 512;
constexpr int kHopSamples = 160;
constexpr int kNumMelBands = 40;
constexpr int kNumFrames = 101;
constexpr int kFftBins = 257;
constexpr float kInputScale = 0.3137255f;
constexpr int8_t kInputZeroPoint = 127;
constexpr float kOutputScale = 0.00390625f;
constexpr int8_t kOutputZeroPoint = -128;
constexpr float kKeywordThreshold = 0.85f;

const char* kKeywordName = "mykeyword";
const char* kWifiSsid = "ZINDAGI CHUNO ENGINEERING NAHI";
const char* kWifiPassword = "rishi1234";
const char* kAsrEndpoint = "http://192.168.1.50:5000/asr/trigger";

const int kArenaSize = 96 * 1024;
uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

bool keywordDetected = false;
bool wifiConnected = false;
static int16_t audio_buffer[kAudioSampleRate];
static float mel_edges[kNumMelBands + 2];
static float mel_filter_weights[kNumMelBands][kFftBins];
static float frame_mels[kNumMelBands];

void initMelFilterBank() {
    const float mel_min = hzToMel(0.0f);
    const float mel_max = hzToMel(8000.0f);

    for (int i = 0; i < kNumMelBands + 2; ++i) {
        float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(kNumMelBands + 1);
        mel_edges[i] = melToHz(mel);
    }

    for (int band = 0; band < kNumMelBands; ++band) {
        const float low = mel_edges[band];
        const float center = mel_edges[band + 1];
        const float high = mel_edges[band + 2];

        for (int bin = 0; bin < kFftBins; ++bin) {
            const float hz = (float)bin * (float)kAudioSampleRate / 2.0f / 256.0f;
            float weight = 0.0f;

            if (hz >= low && hz <= center) {
                weight = (center > low) ? ((hz - low) / (center - low)) : 0.0f;
            } else if (hz > center && hz <= high) {
                weight = (high > center) ? ((high - hz) / (high - center)) : 0.0f;
            }

            mel_filter_weights[band][bin] = weight;
        }
    }
}

bool connectWiFi() {
    if (strlen(kWifiSsid) == 0 || strcmp(kWifiSsid, "YOUR_WIFI_SSID") == 0) {
        Serial.println("WiFi not configured. Cloud handoff disabled.");
        return false;
    }

    WiFi.begin(kWifiSsid, kWifiPassword);
    Serial.print("Connecting to WiFi");
    for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; ++i) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("WiFi connected: ");
        Serial.println(WiFi.localIP());
        return true;
    }

    Serial.println("WiFi connection failed.");
    return false;
}

void sendCloudTrigger() {
    if (!wifiConnected || strlen(kAsrEndpoint) == 0 || strcmp(kAsrEndpoint, "http://192.168.1.50:5000/asr/trigger") == 0) {
        return;
    }

    WiFiClient client;
    const String endpoint = String(kAsrEndpoint);
    int pathStart = endpoint.indexOf("//") + 2;
    int pathSlash = endpoint.indexOf('/', pathStart);
    String host = endpoint.substring(pathStart, pathSlash == -1 ? endpoint.length() : pathSlash);
    String path = pathSlash == -1 ? "/" : endpoint.substring(pathSlash);
    int port = 80;

    int colonPos = host.indexOf(':');
    if (colonPos != -1) {
        port = host.substring(colonPos + 1).toInt();
        host = host.substring(0, colonPos);
    }

    if (!client.connect(host.c_str(), port)) {
        Serial.println("ASR trigger connection failed");
        return;
    }

    String payload = "{\"keyword\":\"";
    payload += kKeywordName;
    payload += "\",\"event\":\"wake_word_detected\"}";

    client.println("POST " + path + " HTTP/1.1");
    client.println("Host: " + host);
    client.println("Content-Type: application/json");
    client.print("Content-Length: ");
    client.println(payload.length());
    client.println();
    client.print(payload);
    delay(100);
    client.stop();
    Serial.println("Wake-word event sent to ASR endpoint");
}

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = kAudioSampleRate,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_start(I2S_PORT);
}

void readOneSecondAudio(int16_t* audio_buffer, size_t& samples_read) {
    samples_read = 0;
    while (samples_read < kAudioSampleRate) {
        int16_t chunk[256];
        size_t bytesRead = 0;
        i2s_read(I2S_PORT, chunk, sizeof(chunk), &bytesRead, portMAX_DELAY);

        size_t chunk_samples = bytesRead / sizeof(int16_t);
        for (size_t i = 0; i < chunk_samples && samples_read < kAudioSampleRate; ++i) {
            audio_buffer[samples_read++] = chunk[i];
        }
    }

    int32_t sum = 0;
    int16_t max_abs = 0;
    for (size_t i = 0; i < samples_read; ++i) {
        const int32_t v = audio_buffer[i];
        sum += v;
        int16_t abs_v = (v < 0) ? (int16_t)(-v) : (int16_t)v;
        if (abs_v > max_abs) {
            max_abs = abs_v;
        }
    }

    const int32_t mean = sum / (int32_t)samples_read;
    const float gain = (max_abs > 2000) ? 18000.0f / (float)max_abs : 1.0f;

    for (size_t i = 0; i < samples_read; ++i) {
        int32_t centered = (int32_t)audio_buffer[i] - mean;
        centered = (int32_t)lrintf((float)centered * gain);
        if (centered > 32767) centered = 32767;
        if (centered < -32768) centered = -32768;
        audio_buffer[i] = (int16_t)centered;
    }
}

void populateInputTensorFromAudio(const int16_t* audio_buffer, size_t sample_count, int8_t* input_data) {
    const int kPad = kWindowSamples / 2;
    float frame_power[kNumFrames][kNumMelBands];
    float max_mel_power = 1.0e-12f;

    for (int frame = 0; frame < kNumFrames; ++frame) {
        for (int band = 0; band < kNumMelBands; ++band) {
            frame_power[frame][band] = 0.0f;
        }

        const int start_index = frame * kHopSamples - kPad;

        for (int bin = 0; bin < kFftBins; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;

            for (int n = 0; n < kWindowSamples; ++n) {
                const int sample_index = start_index + n;
                float sample = 0.0f;

                if (sample_index >= 0 && sample_index < (int)sample_count) {
                    sample = (float)audio_buffer[sample_index] / 32768.0f;
                }

                const float window = 0.54f - 0.46f * cosf((2.0f * 3.14159265358979323846f * n) / (float)(kWindowSamples - 1));
                const float x = sample * window;
                const float angle = -2.0f * 3.14159265358979323846f * (float)bin * (float)n / (float)kWindowSamples;
                real += x * cosf(angle);
                imag += x * sinf(angle);
            }

            const float magnitude_sq = real * real + imag * imag;
            for (int band = 0; band < kNumMelBands; ++band) {
                const float weight = mel_filter_weights[band][bin];
                if (weight > 0.0f) {
                    frame_power[frame][band] += magnitude_sq * weight;
                }
            }
        }

        for (int band = 0; band < kNumMelBands; ++band) {
            if (frame_power[frame][band] > max_mel_power) {
                max_mel_power = frame_power[frame][band];
            }
        }
    }

    for (int frame = 0; frame < kNumFrames; ++frame) {
        for (int band = 0; band < kNumMelBands; ++band) {
            float power = frame_power[frame][band];
            if (power < 1.0e-12f) {
                power = 1.0e-12f;
            }
            if (max_mel_power < 1.0e-12f) {
                max_mel_power = 1.0e-12f;
            }

            const float log_mel = 10.0f * log10f(power / max_mel_power);
            const float clamped_db = constrain(log_mel, -80.0f, 0.0f);
            const int idx = frame * kNumMelBands + band;
            const int8_t q = (int8_t)constrain((int)lrintf((clamped_db / kInputScale) + kInputZeroPoint), -128, 127);
            input_data[idx] = q;
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n--- ESP32-S3 KWS Booting ---");

    Wire.begin(I2C_SDA, I2C_SCL);
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("SSD1306 allocation failed!"));
        while (1) {
            delay(1);
        }
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("System Booting...");
    display.display();

    Serial.println("Initializing I2S Microphone...");
    setupI2S();
    initMelFilterBank();

    wifiConnected = connectWiFi();

    Serial.println("Initializing TFLM Interpreter...");
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch error!");
        return;
    }

    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddStridedSlice();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;

    Serial.println("Allocating Tensors...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        return;
    }

    input = interpreter->input(0);

    Serial.print("Input Shape Confirmed: [");
    for (int i = 0; i < input->dims->size; ++i) {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1) {
            Serial.print(", ");
        }
    }
    Serial.println("]");

    Serial.println("Initialization Complete!");

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Listening...");
    display.display();
}

void loop() {
    if (!keywordDetected) {
        size_t sample_count = 0;
        readOneSecondAudio(audio_buffer, sample_count);
        populateInputTensorFromAudio(audio_buffer, sample_count, input->data.int8);

        if (interpreter->Invoke() == kTfLiteOk) {
            int8_t* output_data = interpreter->output(0)->data.int8;
            const float class0 = (float)(output_data[0] - kOutputZeroPoint) * kOutputScale;
            const float class1 = (float)(output_data[1] - kOutputZeroPoint) * kOutputScale;
            const float keyword_confidence = class1;

            Serial.print("Scores: [");
            Serial.print(class0, 4);
            Serial.print(", ");
            Serial.print(class1, 4);
            Serial.println("]");

            if (keyword_confidence >= kKeywordThreshold) {
                keywordDetected = true;
                sendCloudTrigger();

                display.clearDisplay();
                display.setCursor(0, 0);
                display.println("Detected Keyword");
                display.display();

                delay(2000);

                display.clearDisplay();
                display.setCursor(0, 0);
                display.println("Listening...");
                display.display();

                keywordDetected = false;
            }
        }
    }
}