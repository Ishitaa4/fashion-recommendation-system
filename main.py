import streamlit as st
import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("Fashion Recommender System")

IMAGE_DIR = "images"
UPLOAD_DIR = "uploads"

# ResNet files
RESNET_EMB = "embeddings.pkl"
RESNET_NAMES = "filenames.pkl"

# EfficientNet files
EFF_EMB = "embeddings_efficientnet.pkl"
EFF_NAMES = "filenames_efficientnet.pkl"
FAISS_INDEX = "faiss_hnsw.index"

IMAGES_CSV = "images.csv"
STYLES_CSV = "styles.csv"

# ---------------- LOAD METADATA ----------------
@st.cache_resource
def load_metadata():
    images = pd.read_csv(IMAGES_CSV)
    styles = pd.read_csv(STYLES_CSV)

    styles = styles[["id", "gender", "masterCategory", "articleType", "baseColour"]]
    images["id"] = images["filename"].str.replace(".jpg", "", regex=False).astype(int)

    return images.merge(styles, on="id", how="left")

meta_df = load_metadata()

# ---------------- MODEL SELECTION ----------------
model_choice = st.selectbox(
    "Choose Model",
    ["EfficientNetB0 + FAISS", "ResNet50 + KNN"]
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_resnet():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    return tf.keras.Sequential([base, GlobalMaxPooling2D()])

@st.cache_resource
def load_effnet():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    return tf.keras.Sequential([base, GlobalAveragePooling2D()])

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_resnet_data():
    with open(RESNET_EMB, "rb") as f:
        emb = np.array(pickle.load(f))
    with open(RESNET_NAMES, "rb") as f:
        names = pickle.load(f)
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(emb)
    return emb, names, knn

@st.cache_resource
def load_effnet_data():
    with open(EFF_EMB, "rb") as f:
        emb = np.array(pickle.load(f)).astype("float32")
    with open(EFF_NAMES, "rb") as f:
        names = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX)
    return emb, names, index

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img_path, model_type):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    if model_type == "ResNet50":
        arr = resnet_preprocess(arr)
        model = load_resnet()
    else:
        arr = effnet_preprocess(arr)
        model = load_effnet()

    vec = model.predict(arr, verbose=0).flatten()
    return vec / np.linalg.norm(vec)

# ---------------- SEARCH FUNCTIONS ----------------
def search_resnet(query_vec):
    emb, names, knn = load_resnet_data()
    start = time.time()
    dist, idx = knn.kneighbors([query_vec])
    elapsed = (time.time() - start) * 1000
    return dist[0][1:], idx[0][1:], elapsed

def search_effnet(query_vec):
    _, names, index = load_effnet_data()
    start = time.time()
    D, I = index.search(np.expand_dims(query_vec, axis=0), 6)
    elapsed = (time.time() - start) * 1000
    return D[0][1:], I[0][1:], elapsed

# ---------------- UI FILTERS ----------------
c1, c2, c3, c4, c5 = st.columns(5)

gender = c1.selectbox("Gender", ["All"] + sorted(meta_df["gender"].dropna().unique()))
category = c2.selectbox("Category", ["All"] + sorted(meta_df["masterCategory"].dropna().unique()))
article = c3.selectbox("Article Type", ["All"] + sorted(meta_df["articleType"].dropna().unique()))
colour = c4.selectbox("Base Colour", ["All"] + sorted(meta_df["baseColour"].dropna().unique()))
uploaded_file = c5.file_uploader("Upload Image")

# ---------------- FILTER DATA ----------------
filtered_df = meta_df.copy()
if gender != "All": filtered_df = filtered_df[filtered_df["gender"] == gender]
if category != "All": filtered_df = filtered_df[filtered_df["masterCategory"] == category]
if article != "All": filtered_df = filtered_df[filtered_df["articleType"] == article]
if colour != "All": filtered_df = filtered_df[filtered_df["baseColour"] == colour]

# ---------------- BROWSE ----------------
st.subheader("Browse Products")
cols = st.columns(5)

for i, row in filtered_df.head(20).iterrows():
    with cols[i % 5]:
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        st.image(img_path, use_container_width=True)
        if st.button("View Similar", key=row["filename"]):
            st.session_state["browse_img"] = img_path

# ---------------- RECOMMEND (BROWSE MODE) ----------------
if "browse_img" in st.session_state:
    query_vec = extract_features(
        st.session_state["browse_img"],
        "ResNet50" if "ResNet" in model_choice else "EfficientNet"
    )

    if "ResNet" in model_choice:
        dist, idx, t = search_resnet(query_vec)
        names = load_resnet_data()[1]
    else:
        dist, idx, t = search_effnet(query_vec)
        names = load_effnet_data()[1]

    st.subheader("Recommended for You")
    st.caption(f"Search Time: {round(t,2)} ms")

    rcols = st.columns(5)
    for j, iidx in enumerate(idx[:5]):
        sim = 1 / (1 + dist[j])
        with rcols[j]:
            st.image(names[iidx], use_container_width=True)
            st.caption(f"Similarity: {round(sim*100,2)}%")

# ---------------- UPLOAD MODE ----------------
if uploaded_file:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    upath = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(upath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Uploaded Image")
    st.image(Image.open(uploaded_file), width=250)

    query_vec = extract_features(
        upath,
        "ResNet50" if "ResNet" in model_choice else "EfficientNet"
    )

    if "ResNet" in model_choice:
        dist, idx, t = search_resnet(query_vec)
        names = load_resnet_data()[1]
    else:
        dist, idx, t = search_effnet(query_vec)
        names = load_effnet_data()[1]

    st.subheader("Recommended for You")
    st.caption(f"Search Time: {round(t,2)} ms")

    rcols = st.columns(5)
    for j, iidx in enumerate(idx[:5]):
        sim = 1 / (1 + dist[j])
        with rcols[j]:
            st.image(names[iidx], use_container_width=True)
            st.caption(f"Similarity: {round(sim*100,2)}%")