import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from model import create_cnn_model


# ============================================================
# 1. Random Seed
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 5

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fashion_mnist_cnn.keras"
)


# ============================================================
# 3. Class Names
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# 4. Load FashionMNIST
# ============================================================

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Original training shape:", x_train.shape)
print("Original test shape:", x_test.shape)


# ============================================================
# 5. Preprocessing
# ============================================================

# Convert 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# FashionMNIST normalization
mean = 0.2860
std = 0.3530

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


print("Processed training shape:", x_train.shape)
print("Processed test shape:", x_test.shape)


# ============================================================
# 6. Validation Split
# ============================================================

validation_size = 6000

x_val = x_train[-validation_size:]
y_val = y_train[-validation_size:]

x_train_new = x_train[:-validation_size]
y_train_new = y_train[:-validation_size]


print("Training:", x_train_new.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)


# ============================================================
# 7. Create CNN
# ============================================================

model = create_cnn_model()

model.summary()


# ============================================================
# 8. Compile Model
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    
    loss="sparse_categorical_crossentropy",
    
    metrics=["accuracy"]
)


# ============================================================
# 9. Train Model
# ============================================================

history = model.fit(
    x_train_new,
    y_train_new,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=1
)


# ============================================================
# 10. Test Evaluation
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print()
print("====================================")
print("Test Results")
print("====================================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# 11. Save Model
# ============================================================

model.save(MODEL_PATH)

print()
print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# 12. Training Loss Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 13. Training Accuracy Graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid()

plt.show()


print()
print("Training completed successfully.")