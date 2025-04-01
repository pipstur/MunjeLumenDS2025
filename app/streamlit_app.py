import warnings

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore")

MODEL_PATH = "models/melanoma.onnx"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant"]


@st.cache_resource
def load_model() -> ort.InferenceSession:
    """Loads the ONNX model for inference."""
    return ort.InferenceSession(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses an image for the ONNX model."""
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image).unsqueeze(0).numpy()
    return image


def predict(image: Image.Image, session: ort.InferenceSession) -> tuple[str, float]:
    """Runs inference and returns the predicted class and confidence."""
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})

    prediction = torch.softmax(torch.tensor(output[0]), dim=1)
    confidence, label = torch.max(prediction, dim=1)

    return CLASS_NAMES[label.item()], confidence.item() * 100


def main():
    """Main function to run the Streamlit melanoma detection app."""
    st.title("🔬 Melanoma Detection App")
    st.write("Upload an image to check if it's **benign** or **malignant**.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        session = load_model()
        label, confidence = predict(image, session)

        st.write(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        if label == "Malignant":
            st.error("⚠️ This lesion is likely **malignant**. Please consult a dermatologist.")
        else:
            st.success("✅ This lesion is likely **benign**.")

    st.write(
        """
        **Disclaimer:**\n
        This model is for educational purposes only and should not be used for medical diagnosis.
        """
    )


if __name__ == "__main__":
    main()
