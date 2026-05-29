import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json

print("=== Senthron Training (EfficientNetB0) — Target: 65-70% ===")

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    BATCH_SIZE = 32
else:
    print("No GPU detected — running on CPU")
    BATCH_SIZE = 16

print(f"Batch size: {BATCH_SIZE}")

dataset_path = "dataset"

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Num classes:", num_classes)

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved: class_names.json")

AUTOTUNE = tf.data.AUTOTUNE

def ensure_rgb(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

train_ds = train_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(300).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Compatible decorator for both old and new Keras/TF versions
try:
    # Keras 3.x / TF 2.16+
    register = tf.keras.saving.register_keras_serializable()
except AttributeError:
    try:
        # TF 2.12 - 2.15
        register = tf.keras.utils.register_keras_serializable()
    except AttributeError:
        # Fallback — no registration, use custom_objects at load time
        register = lambda cls: cls

@register
class MobileNetPreprocess(layers.Layer):
    def call(self, x):
        return tf.keras.applications.efficientnet.preprocess_input(x)

# Base model — EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
inputs = layers.Input(shape=(224, 224, 3))

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.15)(x)
x = layers.RandomZoom(0.15)(x)
x = layers.RandomContrast(0.2)(x)
x = layers.RandomBrightness(0.2)(x)

x = MobileNetPreprocess()(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "senthron_best.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

# Phase 1: train only top layers, base frozen
print("\n=== Phase 1: Training top layers (base frozen) ===")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=make_callbacks()
)

# Phase 2: unfreeze top portion of EfficientNetB0
print("\n=== Phase 2: Fine-tuning (top layers unfrozen) ===")
base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable layers in base: {trainable_count} / {len(base_model.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=make_callbacks()
)

# Save final model
model.save("senthron_mobilenet.keras")
print("\nModel saved: senthron_mobilenet.keras")
print("Best checkpoint also saved: senthron_best.keras")
print("Don't forget to give credit to \033]8;;https://github.com/pushkarbiring/senthron-ai\033\\SENTHRON\033]8;;\033\\ (https://github.com/pushkarbiring/senthron-ai)")
