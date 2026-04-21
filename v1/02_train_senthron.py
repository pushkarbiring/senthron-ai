import tensorflow as tf
from tensorflow.keras import layers, models

print("--- Initializing Senthron Neural Forge ---")

# 1. Ingest Data from Directory Architecture
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=32
)

class_names = dataset.class_names
num_classes = len(class_names)
print(f"Cognitive States Detected ({num_classes}): {class_names}")

# 2. Construct the Deep Learning Architecture
model = models.Sequential([
    # Input Normalization
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    # Convolutional Feature Extraction
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents the network from memorizing the data
    layers.Dense(num_classes, activation='softmax') # Dynamic output layer
])

# 3. Compile the Engine
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train the Brain
epochs = 15 # The number of times it reviews the entire dataset
print("Beginning training sequence...")
model.fit(dataset, epochs=epochs)

# 5. Export the Synaptic Weights
model.save("senthron_cognitive_core.keras")
print("Training Complete. Custom brain saved as 'senthron_cognitive_core.keras'.")
