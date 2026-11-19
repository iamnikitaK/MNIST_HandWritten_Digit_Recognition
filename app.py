
import streamlit as st
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageChops

from model import CNNModel

from torchvision import transforms


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #4F46E5;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: gray;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #E8F5E9;
    text-align: center;
}

.predicted-digit {
    font-size: 65px;
    font-weight: bold;
    color: #15803D;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# DEVICE
# --------------------------------

DEVICE = torch.device("cpu")


# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model = CNNModel()

    state_dict = torch.load(
        "mnist_cnn.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# --------------------------------
# IMAGE PREPROCESSING
# --------------------------------

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Detect whether background is white
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1))
    ]

    if np.mean(corners) > 127:
        image = ImageOps.invert(image)

    # Find digit boundaries
    mask = image.point(lambda p: 255 if p > 30 else 0)

    bbox = mask.getbbox()

    if bbox is None:
        return None

    image = image.crop(bbox)

    # Resize while preserving aspect ratio
    image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Place digit in a 28 x 28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    tensor = transform(canvas)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def predict_digit(image):

    tensor = preprocess_image(image)

    if tensor is None:
        return None, None

    with torch.no_grad():

        outputs = model(tensor.to(DEVICE))

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    return (
        predicted.item(),
        probabilities[0].cpu().numpy()
    )


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">'
    'MNIST Handwritten Digit Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning | Convolutional Neural Network | PyTorch'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------
# PROJECT INFORMATION
# --------------------------------

st.subheader("About the Project")

st.write("""
This application uses a Convolutional Neural Network (CNN)
trained on the MNIST dataset to recognize handwritten digits.

Upload an image containing a single handwritten digit
and the model will predict the corresponding number.
""")

st.info(
    "Upload a clear image containing one handwritten digit."
)


# --------------------------------
# IMAGE UPLOAD
# --------------------------------

import struct

st.subheader("Upload Your Handwritten Digit")

uploaded_file = st.file_uploader(
    "Choose an image or MNIST IDX dataset",
    type=None
)

if uploaded_file is not None:

    if uploaded_file.name.endswith("ubyte"):

        data = uploaded_file.read()

        magic, num_images, rows, cols = struct.unpack(
            ">IIII", data[:16]
        )

        if magic != 2051:
            st.error("Invalid MNIST image dataset!")

        else:

            images = np.frombuffer(
                data,
                dtype=np.uint8,
                offset=16
            ).reshape(num_images, rows, cols)

            st.success(f"Successfully loaded {num_images} images!")

            index = st.slider(
                "Select handwritten digit image",
                0,
                num_images - 1
            )

            image = Image.fromarray(images[index])

            st.image(
                image,
                caption=f"MNIST Image {index}",
                width=200
            )

    else:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=200
        )

    if st.button("Recognize Digit", type="primary"):

        predicted_digit, probabilities = predict_digit(image)

        if predicted_digit is None:
            st.error("No digit detected.")

        else:

            confidence = probabilities[predicted_digit] * 100

            st.success(f"Predicted Digit: {predicted_digit}")
            st.write(f"Confidence: {confidence:.2f}%")

            st.session_state["probabilities"] = probabilities
            st.session_state["prediction"] = predicted_digit
# --------------------------------
# PROBABILITY DISTRIBUTION
# --------------------------------

if "probabilities" in st.session_state:

    st.divider()

    st.subheader("Prediction Probability Distribution")

    probabilities = st.session_state["probabilities"]

    digits = list(range(10))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        digits,
        probabilities * 100
    )

    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability (%)")

    ax.set_xticks(digits)

    ax.set_title("CNN Prediction Probabilities")

    ax.set_ylim(0, 100)

    st.pyplot(fig)

    st.success(
        f"Successfully predicted digit: "
        f"{st.session_state['prediction']}"
    )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown(
    """
    <center>
    Developed using Python, PyTorch and Streamlit
    <br>
    MNIST Handwritten Digit Recognition Project
    </center>
    """,
    unsafe_allow_html=True
)
