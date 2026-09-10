import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import layers, models  # type: ignore
except ImportError:
    from keras import layers, models

from sklearn.model_selection import train_test_split

# Load preprocessed data
X = np.load("X_data.npy")
y = np.load("y_data.npy")

# Add channel dimension for CNN input: (Samples, Time, Features, Channels)
X = np.expand_dims(X, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build lightweight Keras model (< 50KB parameters)
model = models.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2], 1)),
    layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax') # 2 classes: background, my_keyword
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=16)

# Save model
model.save("kws_model.h5")
print("Model training complete and saved as kws_model.h5")