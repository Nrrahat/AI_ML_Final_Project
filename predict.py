import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras


# ============================================================
# 1. Paths
# ============================================================

DATASET_DIR = "dataset"

MODEL_PATH = os.path.join(
    "model",
    "fashion_mnist_cnn.keras"
)


# ============================================================
# 2. Class Names
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
# 3. Expected Classes
# ============================================================

expected_classes = {

    "tshirt.jpg": "T-shirt/top",

    "trouser.jpg": "Trouser",

    "pullover.jpg": "Pullover",

    "dress.jpg": "Dress",

    "coat.jpg": "Coat",

    "sandal.jpg": "Sandal",

    "shirt.jpg": "Shirt",

    "sneaker.jpg": "Sneaker",

    "bag.jpg": "Bag",

    "ankle_boot.jpg": "Ankle boot"
}


# ============================================================
# 4. Load Model
# ============================================================

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 5. Normalization Values
# ============================================================

mean = 0.2860
std = 0.3530


# ============================================================
# 6. Prediction Function
# ============================================================

def predict_image(image_path):

    # Open image
    image = Image.open(
        image_path
    ).convert("L")

    # Resize
    image = image.resize(
        (28, 28)
    )

    # Convert to NumPy
    image_array = np.array(
        image
    ).astype("float32")

    # Scale 0-255 → 0-1
    image_array = image_array / 255.0

    # Normalize
    image_array = (
        image_array - mean
    ) / std

    # Add channel dimension
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = np.argmax(
        probabilities
    )

    confidence = probabilities[
        predicted_index
    ]

    return (
        predicted_index,
        confidence
    )


# ============================================================
# 7. Find Images
# ============================================================

image_files = sorted([
    file
    for file in os.listdir(DATASET_DIR)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])


print()
print("Found images:", len(image_files))


# ============================================================
# 8. Prediction Gallery
# ============================================================

results = []


plt.figure(figsize=(15, 10))


for i, filename in enumerate(image_files):

    image_path = os.path.join(
        DATASET_DIR,
        filename
    )

    predicted_index, confidence = predict_image(
        image_path
    )

    predicted_class = class_names[
        predicted_index
    ]

    expected_class = expected_classes.get(
        filename,
        "Unknown"
    )

    results.append({
        "filename": filename,
        "predicted": predicted_class,
        "expected": expected_class,
        "confidence": confidence
    })

    # Display image
    image = Image.open(
        image_path
    ).convert("RGB")

    plt.subplot(2, 5, i + 1)

    plt.imshow(image)

    plt.title(
        f"Pred: {predicted_class}\n"
        f"Confidence: {confidence * 100:.1f}%\n"
        f"True: {expected_class}",
        fontsize=9
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 9. Print Results
# ============================================================

print()
print("============================================")
print("CUSTOM IMAGE PREDICTIONS")
print("============================================")


correct = 0


for result in results:

    print(
        f"{result['filename']} | "
        f"True: {result['expected']} | "
        f"Pred: {result['predicted']} | "
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    if (
        result["expected"]
        == result["predicted"]
    ):
        correct += 1


# ============================================================
# 10. Custom Accuracy
# ============================================================

if len(results) > 0:

    custom_accuracy = (
        correct / len(results)
    )

    print()
    print(
        f"Custom Image Accuracy: "
        f"{custom_accuracy * 100:.2f}%"
    )

    print(
        f"Correct: {correct}/{len(results)}"
    )