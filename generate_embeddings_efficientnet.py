import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from numpy.linalg import norm

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential

# ---------------- CONFIG ----------------
IMAGE_DIR = "images"
EMB_FILE = "embeddings_effnet.pkl"
FILE_FILE = "filenames_effnet.pkl"

# ---------------- MODEL ----------------
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D()
])

# ---------------- LOAD IMAGE FILES ----------------
filenames = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

features = []

for fp in tqdm(filenames):
    img = image.load_img(fp, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    vec = model.predict(arr, verbose=0).flatten()
    vec = vec / (norm(vec) + 1e-10)
    features.append(vec.astype("float32"))

features = np.array(features)

# ---------------- SAVE ----------------
pickle.dump(features, open(EMB_FILE, "wb"))
pickle.dump(filenames, open(FILE_FILE, "wb"))

print("✅ EfficientNet embeddings saved")
print("Embedding shape:", features.shape)
