# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image
# model_paths = {
#     "Baseline CNN": "models/baseline_cnn_model.h5",
#     "ResNet50": "models/resnet50_final.h5",
#     "EfficientNetB0": "models/effnetb0_final.h5"
# }

# st.set_page_config(page_title="Deepfake Detection", layout="wide")

# st.title("Real vs Fake Face Detection")

# tab1, tab2 = st.tabs(["📊 Model Comparison", "🧪 Model Demo"])

# # ---------- TAB 1: DASHBOARD ----------
# with tab1:
#     df = pd.read_csv("metrics.csv")
#     st.subheader("Model Performance Metrics")
#     st.dataframe(df)

#     st.subheader("Accuracy Comparison")
#     st.bar_chart(df.set_index("Model")["Accuracy"])

#     st.subheader("F1 Score Comparison")
#     st.bar_chart(df.set_index("Model")["F1_Score"])

#     st.markdown("""
#     **Key Insights:**
#     - Custom CNN outperforms transfer learning models.
#     - ResNet shows moderate generalization.
#     - EfficientNet struggles with deepfake-specific artifacts.
#     """)

# # ---------- TAB 2: INFERENCE ----------
# with tab2:
#     st.subheader("Upload an Image for Prediction")

#     model = load_model("models/baseline_cnn_model.h5")

#     uploaded = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

#     if uploaded:
#         # Model-specific input size
#        input_sizes = {
#           "Baseline CNN": (224, 224),
#           "ResNet50": (224, 224),
#           "EfficientNetB0": (224, 224)
#          }

#        target_size = input_sizes[model_choice]

#        image = Image.open(uploaded).convert("RGB")
#        image = image.resize(target_size)

#        img_array = np.array(image) / 255.0
#        img_array = np.expand_dims(img_array, axis=0)

#         # image = Image.open(uploaded).resize((256,256))
#         # st.image(image, caption="Uploaded Image", width=250)

#         # img_array = np.array(image) / 255.0
#         # img_array = np.expand_dims(img_array, axis=0)

#         prediction = model.predict(img_array)[0][0]

#         label = "Fake" if prediction > 0.5 else "Real"
#         confidence = prediction if prediction > 0.5 else 1 - prediction

#         st.success(f"Prediction: **{label}**")
#         st.info(f"Confidence: **{confidence:.2%}**")

# model_choice = st.selectbox(
#     "Select Model",
#     ["Baseline CNN", "ResNet50", "EfficientNetB0"]
# )

# model = load_model(model_paths[model_choice])
# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image

# # ----------------------------------
# # Page config
# # ----------------------------------
# st.info(
#     "⚠️ Disclaimer: All images and videos used in this application are sourced from "
#     "public datasets or AI-generated sources and are used strictly for academic "
#     "and research demonstration purposes."
# )

# st.set_page_config(
#     page_title="Deepfake Detection System",
#     layout="wide"
# )

# st.title("🕵️ Deepfake Detection System")

# # ----------------------------------
# # Cache models
# # ----------------------------------
# @st.cache_resource
# def load_models():
#     models = {
#         "Baseline CNN": load_model("models/baseline_cnn_model.h5"),
#         "ResNet50": load_model("models/resnet50_final.h5"),
#         "EfficientNetB0": load_model("models/effnetb0_final.h5")
#     }
#     return models

# models = load_models()

# # ----------------------------------
# # Model input sizes
# # ----------------------------------
# INPUT_SIZES = {
#     "Baseline CNN": (224, 224),
#     "ResNet50": (224, 224),
#     "EfficientNetB0": (224, 224)
# }

# # ----------------------------------
# # Tabs
# # ----------------------------------
# tab1, tab2 = st.tabs(["📊 Model Comparison", "🧪 Model Demo"])

# # ---------- TAB 1: DASHBOARD ----------
# with tab1:
#     st.subheader("Model Performance Metrics")

#     # Load metrics.csv
#     df = pd.read_csv("metrics.csv")
#     st.dataframe(df)

#     st.subheader("Accuracy Comparison")
#     st.bar_chart(df.set_index("Model")["Accuracy"])

#     st.subheader("F1 Score Comparison")
#     st.bar_chart(df.set_index("Model")["F1_Score"])

#     st.markdown("""
#     **Key Insights:**
#     - Custom CNN outperforms transfer learning models.
#     - ResNet shows moderate generalization.
#     - EfficientNet struggles with deepfake-specific artifacts.
#     """)

# # ---------- TAB 2: INFERENCE ----------
# with tab2:
#     st.subheader("Upload an Image for Prediction")
#     st.write("Upload an image to detect whether it is **Real or Fake**.")

#     # Model selection
#     model_choice = st.selectbox(
#         "Select Model",
#         list(models.keys())
#     )

#     model = models[model_choice]
#     target_size = INPUT_SIZES[model_choice]

#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload Face Image",
#         type=["jpg", "jpeg", "png"]
#     )

#     # Inference
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Preprocessing
#         image = image.resize(target_size)
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Prediction
#         prediction = model.predict(img_array)[0][0]

#         label = "FAKE" if prediction >= 0.5 else "REAL"
#         confidence = prediction if label == "FAKE" else 1 - prediction

#         st.markdown("---")
#         st.subheader("Prediction Result")
#         st.write(f"**Prediction:** {label}")
#         st.write(f"**Confidence:** {confidence:.2%}")
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------------
# Page config (must be first)
# ----------------------------------
st.set_page_config(
    page_title="Deepfake Detection System",
    layout="wide"
)

# ----------------------------------
# Disclaimer
# ----------------------------------
st.info(
    "⚠️ Disclaimer: All images and videos used in this application are sourced from "
    "public datasets or AI-generated sources and are used strictly for academic "
    "and research demonstration purposes."
)

st.title("🕵️ Deepfake Detection System")

# ----------------------------------
# Cache models
# ----------------------------------
@st.cache_resource
def load_models():
    return {
        "Baseline CNN": load_model("models/baseline_cnn_model.h5"),
        "ResNet50": load_model("models/resnet50_final.h5"),
        "EfficientNetB0": load_model("models/effnetb0_final.h5")
    }

models = load_models()

# ----------------------------------
# Model input sizes
# ----------------------------------
INPUT_SIZES = {
    "Baseline CNN": (224, 224),
    "ResNet50": (224, 224),
    "EfficientNetB0": (224, 224)
}

# ----------------------------------
# Tabs
# ----------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Model Comparison", "🧪 Image Demo", "🎥 Video Demo"]
)

# ======================================================
# TAB 1 — DASHBOARD
# ======================================================
with tab1:
    st.subheader("Model Performance Metrics")

    df = pd.read_csv("metrics.csv")
    st.dataframe(df)

    st.subheader("Accuracy Comparison")
    st.bar_chart(df.set_index("Model")["Accuracy"])

    st.subheader("F1 Score Comparison")
    st.bar_chart(df.set_index("Model")["F1_Score"])

    st.markdown("""
    **Key Insights:**
    - Custom CNN demonstrates the strongest performance.
    - ResNet50 provides moderate generalization.
    - EfficientNetB0 underperforms on deepfake-specific artifacts.
    """)

# ======================================================
# TAB 2 — IMAGE INFERENCE
# ======================================================
with tab2:
    st.subheader("Image-Based Deepfake Detection")

    st.write(
        "Upload a facial image and select a trained model to classify it as **Real** or **Fake**."
    )

    model_choice = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models[model_choice]
    target_size = INPUT_SIZES[model_choice]

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image = image.resize(target_size)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        label = "FAKE" if prediction >= 0.5 else "REAL"
        confidence = prediction if label == "FAKE" else 1 - prediction

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"**Selected Model:** {model_choice}")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2%}")

# ======================================================
# TAB 3 — VIDEO INFERENCE
# ======================================================
with tab3:
    st.subheader("Video-Based Deepfake Detection")

    st.write(
        "This module performs frame-wise analysis on uploaded videos "
        "to detect potential deepfake content."
    )

    model_choice_video = st.selectbox(
        "Select Model for Video Analysis",
        list(models.keys()),
        key="video_model"
    )

    model_video = models[model_choice_video]
    target_size = INPUT_SIZES[model_choice_video]

    uploaded_video = st.file_uploader(
        "Upload Video File",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_count = 0
        fake_count = 0
        MAX_FRAMES = 30  # limit for performance

        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)

            pred = model_video.predict(frame)[0][0]
            if pred >= 0.5:
                fake_count += 1

            frame_count += 1

        cap.release()

        if frame_count > 0:
            fake_ratio = fake_count / frame_count

            st.markdown("---")
            st.subheader("Video Analysis Result")
            st.write(f"Frames analyzed: {frame_count}")
            st.write(f"Fake frame ratio: {fake_ratio:.2%}")

            if fake_ratio > 0.5:
                st.error("⚠️ Video likely contains deepfake content")
            else:
                st.success("✅ Video appears authentic")

st.warning("Model performance may vary on images outside the training distribution.")
