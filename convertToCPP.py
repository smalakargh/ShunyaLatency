with open("model.tflite", "rb") as f:
    tflite_bytes = f.read()

header_content = "// Automatically generated model.h for ESP32-S3 TFLM\n"
header_content += "alignas(16) const unsigned char g_model[] = {\n"
header_content += ", ".join([hex(b) for b in tflite_bytes])
header_content += "\n};\n"
header_content += f"const int g_model_len = {len(tflite_bytes)};\n"

with open("model.h", "w") as f:
    f.write(header_content)

print("model.h generated successfully!")