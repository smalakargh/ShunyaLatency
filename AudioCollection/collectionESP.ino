#include <driver/i2s.h>

// I2S Pin Definitions for INMP441
#define I2S_WS 5
#define I2S_SD 6
#define I2S_SCK 4
#define I2S_PORT I2S_NUM_0

// Audio parameters
const int sampleRate = 16000;
const int durationSeconds = 1;
const int numSamples = sampleRate * durationSeconds;
int16_t buffer[numSamples];

void setup() {
  Serial.begin(500000); // High baud rate for fast data transfer
  delay(1000);

  // Configure I2S peripheral
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = sampleRate,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .fixed_mclk = 0
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
  i2s_zero_dma_buffer(I2S_PORT);
}

void loop() {
  // Wait for a command character 'R' from the computer to record one sample
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'R') {
      size_t bytesRead = 0;
      i2s_read(I2S_PORT, (void*)buffer, sizeof(buffer), &bytesRead, portMAX_DELAY);
      
      // Send raw binary PCM data over Serial to the PC script
      Serial.write((uint8_t*)buffer, sizeof(buffer));
    }
  }
}