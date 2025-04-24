import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)

import os
import warnings
from typing import List

import numpy as np
import onnxruntime as ort
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image

from img_utils.preprocessing import apply_clahe, resize_image

warnings.filterwarnings("ignore")

MODEL_FOLDER = "models/included_models"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant"]
REFERENCE_IMAGE_URL = (
    "https://www.yashodahealthcare.com/blogs/wp-content/uploads/2021/07/melanoma-skin-cancer.jpeg"
)
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@st.cache_resource
def load_onnx_models(models_folder: str) -> List[ort.InferenceSession]:
    """Load ONNX models from a given folder."""
    sessions = []
    for model_name in os.listdir(models_folder):
        if model_name.endswith(".onnx"):
            model_path = os.path.join(models_folder, model_name)
            session = ort.InferenceSession(model_path, providers=PROVIDERS)
            sessions.append(session)
    return sessions


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the input image: resize and apply CLAHE."""
    image = resize_image(image, IMAGE_SIZE, padding_flag=True)
    image = apply_clahe(image, clip_limit=1.4, tile_grid_size=(8, 8))
    return image


def prepare_image_tensor(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized tensor suitable for model input."""
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform_pipeline(image).numpy()
    return np.expand_dims(tensor, axis=0)


def predict_image(
    sessions: List[ort.InferenceSession], image_tensor: np.ndarray
) -> List[np.ndarray]:
    """Run inference on the image using all loaded models."""
    predictions = []
    for session in sessions:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_tensor})
        prob = outputs[0].squeeze()
        predictions.append(prob)

    return predictions


def soft_vote(predictions: List[np.ndarray]) -> int:
    """Aggregate predictions using soft voting."""
    probabilities = [np.exp(logits) / np.sum(np.exp(logits)) for logits in predictions]
    avg_prob = np.mean(probabilities, axis=0)
    final_prediction = int(np.argmax(avg_prob))
    return final_prediction, avg_prob[final_prediction] * 100


def get_confidence_text(avg_prob: float) -> str:
    if 50 <= avg_prob < 65:
        return "**(low confidence)**"
    elif 65 <= avg_prob < 80:
        return "**(medium confidence)**"
    else:
        return "**(high confidence)**"


def main():
    st.title("🔬 Melanoma Detection App")
    st.write("Upload an image to check if it's **benign** or **malignant**.")

    st.image(
        REFERENCE_IMAGE_URL,
        caption="How to notice early signs of Melanoma",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        sessions = load_onnx_models(MODEL_FOLDER)

        preprocessed_image = preprocess_image(image)
        image_tensor = prepare_image_tensor(preprocessed_image)

        predictions = predict_image(sessions, image_tensor)
        final_prediction, avg_prob = soft_vote(predictions)

        label = CLASS_NAMES[final_prediction]
        added_text = get_confidence_text(avg_prob)

        st.write(f"### Prediction: **{label}**")
        st.write(f"Confidence: {avg_prob:.2f}%, {added_text}")

        if label == "Malignant":
            st.error(
                f"""⚠️ Based on the prediction {added_text}, this lesion is likely **malignant**.
                Please consult a dermatologist for further evaluation and diagnosis."""
            )

        else:
            st.success(
                f"""✅ Based on the prediction {added_text}, this lesion is likely **benign**.
                However, if you have any concerns, consider consulting a dermatologist."""
            )

    st.write(
        """
        **Disclaimer:**
        This model is for educational purposes only and should not be used for medical diagnosis.
        Our model has been trained and validated on many images, however
        we cannot guarantee its accuracy.
        Consult the diagram above and judge for yourself, or go to a dermatologist
        if you feel you are in risk of melanoma.
        """
    )


if __name__ == "__main__":
    main()
